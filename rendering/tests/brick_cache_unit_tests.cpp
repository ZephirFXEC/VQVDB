#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <list>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

#include "vqvdb/brick_cache.hpp"

namespace {

constexpr uint64_t kInvalidMorton = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kEmptyHashSlot = 0xFFFFFFFFu;

struct ReferenceAllocation {
	uint32_t slotIndex{0};
	bool wasCached{false};
	bool evicted{false};
	uint64_t evictedMorton{0};
};

class ReferenceLRU {
   public:
	explicit ReferenceLRU(uint32_t capacity) : capacity_(capacity), slotToMorton_(capacity, kInvalidMorton) {
		for (uint32_t i = capacity_; i > 0; --i) {
			freeSlots_.push_back(i - 1u);
		}
	}

	std::optional<uint32_t> lookup(uint64_t morton) {
		const auto it = map_.find(morton);
		if (it == map_.end()) return std::nullopt;
		touch(morton);
		return it->second;
	}

	ReferenceAllocation allocate(uint64_t morton) {
		const auto existing = map_.find(morton);
		if (existing != map_.end()) {
			touch(morton);
			return ReferenceAllocation{.slotIndex = existing->second, .wasCached = true, .evicted = false, .evictedMorton = 0};
		}

		ReferenceAllocation out{};
		if (!freeSlots_.empty()) {
			out.slotIndex = freeSlots_.back();
			freeSlots_.pop_back();
		} else {
			const uint64_t evictedMorton = lru_.back();
			lru_.pop_back();
			lruPos_.erase(evictedMorton);

			out.slotIndex = map_.at(evictedMorton);
			out.evicted = true;
			out.evictedMorton = evictedMorton;
			map_.erase(evictedMorton);
			slotToMorton_[out.slotIndex] = kInvalidMorton;
		}

		map_[morton] = out.slotIndex;
		slotToMorton_[out.slotIndex] = morton;
		lru_.push_front(morton);
		lruPos_[morton] = lru_.begin();
		return out;
	}

   private:
	void touch(uint64_t morton) {
		auto posIt = lruPos_.find(morton);
		if (posIt == lruPos_.end()) return;
		lru_.erase(posIt->second);
		lru_.push_front(morton);
		lruPos_[morton] = lru_.begin();
	}

	uint32_t capacity_{0};
	std::unordered_map<uint64_t, uint32_t> map_;
	std::vector<uint64_t> slotToMorton_;
	std::vector<uint32_t> freeSlots_;
	std::list<uint64_t> lru_;
	std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lruPos_;
};

std::optional<uint32_t> probeHashTable(std::span<const vqvdb::CacheHashEntry> table, uint64_t morton) {
	if (table.empty()) return std::nullopt;

	auto hashMorton = [](uint64_t value) -> uint32_t {
		uint64_t x = value;
		x ^= x >> 33;
		x *= 0xff51afd7ed558ccdULL;
		x ^= x >> 33;
		x *= 0xc4ceb9fe1a85ec53ULL;
		x ^= x >> 33;
		return static_cast<uint32_t>(x);
	};

	const uint32_t tableSize = static_cast<uint32_t>(table.size());
	const uint32_t mask = tableSize - 1u;
	uint32_t idx = hashMorton(morton) & mask;

	for (uint32_t probe = 0; probe < tableSize; ++probe) {
		const auto& e = table[idx];
		if (e.slotIndex == kEmptyHashSlot) {
			return std::nullopt;
		}
		const uint64_t entryMorton = (static_cast<uint64_t>(e.mortonHigh) << 32u) | static_cast<uint64_t>(e.mortonLow);
		if (entryMorton == morton) {
			return e.slotIndex;
		}
		idx = (idx + 1u) & mask;
	}

	return std::nullopt;
}

}  // namespace

TEST_CASE("Brick cache computes atlas dimensions and slot offsets") {
	vqvdb::BrickCache cache;
	vqvdb::BrickCacheConfig config;
	config.capacitySlots = 2048;
	config.brickSizeVoxels = 8;
	config.allocateAtlasTexture = false;

	const auto init = vqvdb::initBrickCache(cache, config);
	REQUIRE(init.has_value());

	CHECK(cache.slotGridDims.x == 16);
	CHECK(cache.slotGridDims.y == 16);
	CHECK(cache.slotGridDims.z == 8);
	CHECK(cache.atlasDimsVoxels.x == 128);
	CHECK(cache.atlasDimsVoxels.y == 128);
	CHECK(cache.atlasDimsVoxels.z == 64);

	const auto off0 = vqvdb::slotToAtlasOffset(cache, 0);
	REQUIRE(off0.has_value());
	CHECK(off0->x == 0);
	CHECK(off0->y == 0);
	CHECK(off0->z == 0);

	const auto off17 = vqvdb::slotToAtlasOffset(cache, 17);
	REQUIRE(off17.has_value());
	CHECK(off17->x == 8);
	CHECK(off17->y == 8);
	CHECK(off17->z == 0);

	const auto offLast = vqvdb::slotToAtlasOffset(cache, 2047);
	REQUIRE(offLast.has_value());
	CHECK(offLast->x == 120);
	CHECK(offLast->y == 120);
	CHECK(offLast->z == 56);

	vqvdb::shutdownBrickCache(cache);
}

TEST_CASE("Brick cache evicts LRU entry after access updates recency") {
	vqvdb::BrickCache cache;
	vqvdb::BrickCacheConfig config;
	config.capacitySlots = 3;
	config.allocateAtlasTexture = false;
	REQUIRE(vqvdb::initBrickCache(cache, config).has_value());

	const uint64_t a = vqvdb::encodeMorton64(0, 0, 0);
	const uint64_t b = vqvdb::encodeMorton64(8, 0, 0);
	const uint64_t c = vqvdb::encodeMorton64(16, 0, 0);
	const uint64_t d = vqvdb::encodeMorton64(24, 0, 0);

	REQUIRE(vqvdb::allocateSlot(cache, a).has_value());
	REQUIRE(vqvdb::allocateSlot(cache, b).has_value());
	REQUIRE(vqvdb::allocateSlot(cache, c).has_value());

	// Touch A so B becomes LRU.
	const auto hitA = vqvdb::lookupBrick(cache, a);
	REQUIRE(hitA.has_value());
	REQUIRE(hitA->has_value());

	const auto allocD = vqvdb::allocateSlot(cache, d);
	REQUIRE(allocD.has_value());
	CHECK(allocD->evicted);
	CHECK(allocD->evictedMorton == b);

	const auto lookupB = vqvdb::lookupBrick(cache, b);
	REQUIRE(lookupB.has_value());
	CHECK_FALSE(lookupB->has_value());

	const auto lookupA = vqvdb::lookupBrick(cache, a);
	REQUIRE(lookupA.has_value());
	CHECK(lookupA->has_value());

	const auto stats = vqvdb::getCacheStats(cache);
	CHECK(stats.capacitySlots == 3);
	CHECK(stats.residentBricks == 3);
	CHECK(stats.evictionCount == 1);

	vqvdb::shutdownBrickCache(cache);
}

TEST_CASE("Brick cache randomized operations match reference LRU model") {
	std::mt19937 rng(1337);
	std::uniform_int_distribution<uint32_t> capDist(2, 32);
	std::uniform_int_distribution<uint32_t> opDist(0, 99);

	for (int run = 0; run < 20; ++run) {
		const uint32_t capacity = capDist(rng);

		vqvdb::BrickCache cache;
		vqvdb::BrickCacheConfig config;
		config.capacitySlots = capacity;
		config.allocateAtlasTexture = false;
		REQUIRE(vqvdb::initBrickCache(cache, config).has_value());

		ReferenceLRU ref(capacity);
		std::uniform_int_distribution<uint32_t> keyDist(0, capacity * 4);

		for (int step = 0; step < 500; ++step) {
			const uint64_t key = vqvdb::encodeMorton64(static_cast<int32_t>(keyDist(rng)), static_cast<int32_t>(keyDist(rng)),
			                                           static_cast<int32_t>(keyDist(rng)));
			if (opDist(rng) < 65) {
				const auto expected = ref.allocate(key);
				const auto actual = vqvdb::allocateSlot(cache, key);
				REQUIRE(actual.has_value());
				CHECK(actual->slotIndex == expected.slotIndex);
				CHECK(actual->wasCached == expected.wasCached);
				CHECK(actual->evicted == expected.evicted);
				CHECK(actual->evictedMorton == expected.evictedMorton);
			} else {
				const auto expected = ref.lookup(key);
				const auto actual = vqvdb::lookupBrick(cache, key);
				REQUIRE(actual.has_value());
				CHECK(actual->has_value() == expected.has_value());
				if (actual->has_value()) {
					CHECK(actual->value() == expected.value());
				}
			}
		}

		vqvdb::shutdownBrickCache(cache);
	}
}

TEST_CASE("Brick cache builds open-address hash table for morton->slot lookups") {
	vqvdb::BrickCache cache;
	vqvdb::BrickCacheConfig config;
	config.capacitySlots = 8;
	config.allocateAtlasTexture = false;
	REQUIRE(vqvdb::initBrickCache(cache, config).has_value());

	std::vector<uint64_t> mortonCodes;
	for (int i = 0; i < 6; ++i) {
		const uint64_t m = vqvdb::encodeMorton64(i * 8, i * 4, i * 2);
		mortonCodes.push_back(m);
		REQUIRE(vqvdb::allocateSlot(cache, m).has_value());
	}

	const std::vector<vqvdb::CacheHashEntry> table = vqvdb::buildCacheHashTable(cache);
	REQUIRE(table.size() == 16);  // nextPow2(capacity * 2)

	for (const uint64_t morton : mortonCodes) {
		const auto slot = vqvdb::lookupBrick(cache, morton);
		REQUIRE(slot.has_value());
		REQUIRE(slot->has_value());

		const auto hashedSlot = probeHashTable(table, morton);
		REQUIRE(hashedSlot.has_value());
		CHECK(hashedSlot.value() == slot->value());
	}

	vqvdb::shutdownBrickCache(cache);
}
