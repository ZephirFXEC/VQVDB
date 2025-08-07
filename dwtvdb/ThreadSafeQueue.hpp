//
// Created by zphrfx on 07/08/2025.
//

#ifndef DWTVDB_THREADSAFEQUEUE_HPP
#define DWTVDB_THREADSAFEQUEUE_HPP

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class ThreadSafeQueue {
public:
	explicit ThreadSafeQueue(size_t capacity = 0)
		: capacity_(capacity), closed_(false) {
	}

	// Blocking push. Returns false if queue is closed while waiting.
	bool push(T&& item) {
		std::unique_lock<std::mutex> lock(m_);
		cv_not_full_.wait(lock, [&] {
			return closed_ || capacity_ == 0 || q_.size() < capacity_;
		});
		if (closed_) return false;
		q_.push(std::move(item));
		cv_not_empty_.notify_one();
		return true;
	}

	// Blocking pop. Returns false on closed and empty.
	bool pop(T& out) {
		std::unique_lock<std::mutex> lock(m_);
		cv_not_empty_.wait(lock,
		                   [&] { return closed_ || !q_.empty(); });
		if (q_.empty()) return false;
		out = std::move(q_.front());
		q_.pop();
		cv_not_full_.notify_one();
		return true;
	}

	void close() {
		std::lock_guard<std::mutex> lock(m_);
		closed_ = true;
		cv_not_empty_.notify_all();
		cv_not_full_.notify_all();
	}

	bool empty() const {
		std::lock_guard<std::mutex> lock(m_);
		return q_.empty();
	}

private:
	mutable std::mutex m_;
	std::condition_variable cv_not_empty_;
	std::condition_variable cv_not_full_;
	std::queue<T> q_;
	size_t capacity_;
	bool closed_;
};

#endif //DWTVDB_THREADSAFEQUEUE_HPP