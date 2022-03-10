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

#ifndef EVENT_FOURIER__EVENT_FOURIER_H_
#define EVENT_FOURIER__EVENT_FOURIER_H_

#include <event_array_msgs/msg/event_array.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <opencv2/core/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace event_fourier
{
class EventFourier : public rclcpp::Node
{
public:
  explicit EventFourier(const rclcpp::NodeOptions & options);
  ~EventFourier();

  EventFourier(const EventFourier &) = delete;
  EventFourier & operator=(const EventFourier &) = delete;

private:
  using EventArray = event_array_msgs::msg::EventArray;
  using EventArrayConstPtr = EventArray::ConstSharedPtr;
  typedef std::complex<double> complex_t;
  void readEventsFromBag(const std::string & bagName);
  bool initialize();
  void callbackEvents(EventArrayConstPtr msg);
  void resetState(uint32_t width, uint32_t height);
  void updateState(
    const size_t f_idx, const uint16_t x, const uint16_t y, uint64_t t, bool polarity);
  void publishImage();

  // ------ variables ----
  rclcpp::Time lastTime_{0};
  uint64_t sliceTime_{0};
  bool useSensorTime_;
  image_transport::Publisher imagePub_;
  rclcpp::Subscription<EventArray>::SharedPtr eventSub_;
  rclcpp::TimerBase::SharedPtr pubTimer_;

  std::vector<double> freq_;
  complex_t * state_{0};
  uint32_t width_;
  uint32_t height_;
  uint64_t eventCount_{0};
  uint64_t lastCount_{0};
  uint64_t totTime_{0};
  std_msgs::msg::Header header_;
};
}  // namespace event_fourier
#endif  // EVENT_FOURIER__EVENT_FOURIER_H_
