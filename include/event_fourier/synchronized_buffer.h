// -*-c++-*--------------------------------------------------------------------
// Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef EVENT_FOURIER__SYNCHRONIZED_BUFFER_H_
#define EVENT_FOURIER__SYNCHRONIZED_BUFFER_H_

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace event_fourier
{
template <class T>
class SynchronizedBuffer
{
public:
  // need to define these to make compiler errors related to
  // mutex not being copyable go away
  SynchronizedBuffer(size_t reserveSize = 1) { buffer_.reserve(reserveSize); }
  SynchronizedBuffer(const SynchronizedBuffer & b) { buffer_.reserve(b.buffer_.capacity()); }
  SynchronizedBuffer & operator=(const SynchronizedBuffer & b)
  {
    buffer_.reserve(b.buffer_.capacity());
  }

  // add to buffer
  void add(const T & e) { buffer_.push_back(e); }

  const std::vector<T> & get_elements() { return (buffer_); }

  void writer_done()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    isWriting_ = false;
    cv_.notify_all();
  }

  void wait_until_write_complete()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (isWriting_) {
      cv_.wait(lock);
    }
  }
  void reader_done()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    buffer_.clear();    // should not change the capacity
    isWriting_ = true;  // it's the writer's turn now
    cv_.notify_all();
  }

  void wait_until_read_complete()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    // reader will clear isWriting when done
    while (!isWriting_) {
      cv_.wait(lock);
    }
  }

private:
  std::vector<T> buffer_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool isWriting_{false};
};
}  // namespace event_fourier
#endif  // EVENT_FOURIER__SYNCHRONIZED_BUFFER_H_
