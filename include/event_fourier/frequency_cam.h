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

#ifndef EVENT_FOURIER__FREQUENCY_CAM_H_
#define EVENT_FOURIER__FREQUENCY_CAM_H_

#include <stdlib.h>

#include <cstdlib>
#include <event_array_msgs/msg/event_array.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/core/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace event_fourier
{
class FrequencyCam : public rclcpp::Node
{
public:
  explicit FrequencyCam(const rclcpp::NodeOptions & options);
  ~FrequencyCam();

  FrequencyCam(const FrequencyCam &) = delete;
  FrequencyCam & operator=(const FrequencyCam &) = delete;

private:
  typedef float variable_t;
  struct State  // per-pixel filter state
  {
    variable_t t_flip{0};     // time of last flip
    bool upper_half{false};   // whether signal is in upper or lower half
    variable_t t{0};          // last time stamp
    variable_t p{0};          // lagged polarity of events
    variable_t x[2]{0, 0};    // current and lagged signal x
    variable_t dt_avg{-1.0};  // average sample time (time between events)
  };
  using EventArray = event_array_msgs::msg::EventArray;
  using EventArrayConstPtr = EventArray::ConstSharedPtr;
  void readEventsFromBag(const std::string & bagName);
  bool initialize();
  void initializeState(uint32_t width, uint32_t height, uint64_t t);
  void callbackEvents(EventArrayConstPtr msg);
  void publishImage();
  void statistics();
  void updateState(const uint16_t x, const uint16_t y, uint64_t t, bool polarity);
  cv::Mat makeRawFrequencyImage() const;
  // ------ variables ----
  rclcpp::Time lastTime_{0};
  uint64_t sliceTime_{0};
  bool useSensorTime_;
  image_transport::Publisher imagePub_;
  rclcpp::Subscription<EventArray>::SharedPtr eventSub_;
  rclcpp::TimerBase::SharedPtr pubTimer_;
  rclcpp::TimerBase::SharedPtr statsTimer_;

  std::vector<uint32_t> roi_;
  uint32_t ixStart_;
  uint32_t ixEnd_;
  uint32_t iyStart_;
  uint32_t iyEnd_;
  State * state_{0};
  double freq_[2]{-1.0, -1.0};  // frequency range
  uint32_t width_;
  uint32_t height_;
  uint64_t eventCount_{0};
  uint64_t msgCount_{0};
  uint64_t lastCount_{0};
  uint64_t totTime_{0};
  uint64_t lastEventTime_;
  int64_t lastSeq_{0};
  int64_t droppedSeq_{0};
  std_msgs::msg::Header header_;
  // ---------- coefficients for state update
  variable_t c_[2];
  variable_t c_p_{0};
  variable_t dtMix_{1.0 / 100.0};
  variable_t dtDecay_{1 - dtMix_};
  variable_t resetThreshold_{5};
  // ------------------ debugging stuff
  uint16_t debugX_{0};
  uint16_t debugY_{0};
};
}  // namespace event_fourier
#endif  // EVENT_FOURIER__FREQUENCY_CAM_H_
