//
// Created by zphrfx on 07/08/2025.
//

#ifndef DWTVDB_COMPRESSIONBACKEND_HPP
#define DWTVDB_COMPRESSIONBACKEND_HPP

#include <zlib.h>

#include <stdexcept>
#include <string>
#include <vector>

std::string zerr(int code) {
	switch (code) {
		case Z_MEM_ERROR:
			return "Z_MEM_ERROR";
		case Z_BUF_ERROR:
			return "Z_BUF_ERROR";
		case Z_DATA_ERROR:
			return "Z_DATA_ERROR";
		default:
			return "Zlib error " + std::to_string(code);
	}
}

std::vector<char> zcompress(const std::vector<int16_t>& src) {
	if (src.empty()) return {};
	const uLong srcBytes = src.size() * sizeof(int16_t);
	uLongf bound = compressBound(srcBytes);
	std::vector<Bytef> tmp(bound);
	uLongf compBytes = bound;
	int res = compress2(tmp.data(), &compBytes, reinterpret_cast<const Bytef*>(src.data()), srcBytes, Z_BEST_COMPRESSION);
	if (res != Z_OK) throw std::runtime_error("zlib compress: " + zerr(res));
	std::vector<char> blob(sizeof(uint32_t) + compBytes);
	uint32_t szLE = static_cast<uint32_t>(srcBytes);
	memcpy(blob.data(), &szLE, sizeof(uint32_t));
	memcpy(blob.data() + sizeof(uint32_t), tmp.data(), compBytes);
	return blob;
}

std::vector<int16_t> zdecompress(const std::vector<char>& blob) {
	if (blob.size() < sizeof(uint32_t)) throw std::runtime_error("blob too small");
	uint32_t dstBytes;
	memcpy(&dstBytes, blob.data(), sizeof(uint32_t));
	if (dstBytes == 0) return {};
	std::vector<int16_t> dst(dstBytes / sizeof(int16_t));
	uLongf dstLen = dstBytes;
	const Bytef* src = reinterpret_cast<const Bytef*>(blob.data() + sizeof(uint32_t));
	const uLong srcLen = blob.size() - sizeof(uint32_t);
	int res = uncompress(reinterpret_cast<Bytef*>(dst.data()), &dstLen, src, srcLen);
	if (res != Z_OK || dstLen != dstBytes) throw std::runtime_error("zlib uncompress: " + zerr(res));
	return dst;
}

#endif  // DWTVDB_COMPRESSIONBACKEND_HPP