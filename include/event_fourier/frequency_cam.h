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
#include <iostream>
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
  struct Event  // event representation, needed for filtering
  {
    uint64_t t;
    uint16_t x;
    uint16_t y;
    bool polarity;
  };
  friend std::ostream & operator<<(std::ostream & os, const Event & e);
  struct State  // per-pixel filter state
  {
    variable_t t_flip{0};     // time of last flip
    bool upper_half{false};   // whether signal is in upper or lower half
    variable_t t{0};          // last time stamp
    variable_t p{0};          // lagged polarity of events
    variable_t x[2]{0, 0};    // current and lagged signal x
    variable_t dt_avg{-1.0};  // average sample time (time between events)
    uint8_t skip{4};          // counter for noise filter
    Event e[4];               // buffer of events for noise filter
    uint8_t idx{0};           // index pointer into noise event buffer
  };
  using EventArray = event_array_msgs::msg::EventArray;
  using EventArrayConstPtr = EventArray::ConstSharedPtr;
  void readEventsFromBag(const std::string & bagName);
  bool initialize();
  void initializeState(uint32_t width, uint32_t height, uint64_t t);
  void callbackEvents(EventArrayConstPtr msg);
  void publishImage();
  void statistics();
  void updateState(State * state, const Event & e);
  struct NoTF
  {
    static double tf(double f) { return (f); }
  };
  struct LogTF
  {
    static double tf(double f) { return (std::log10(f)); }
  };
  template <class T>
  cv::Mat makeTransformedFrequencyImage() const
  {
    const double lastEventTime = 1e-9 * lastEventTime_;
    cv::Mat rawImg(height_, width_, CV_32FC1, 0.0);
    const double maxDt = 1.0 / freq_[0] * 2.0;
    const double minFreq = T::tf(freq_[0]);
    for (uint32_t iy = iyStart_; iy < iyEnd_; iy++) {
      for (uint32_t ix = ixStart_; ix < ixEnd_; ix++) {
        const size_t offset = iy * width_ + ix;
        const State & state = state_[offset];
        const double dt = lastEventTime - state.t;
        const double f = 1.0 / std::max(state.dt_avg, 1e-6f);
        // filter out any pixels that have not been updated
        // for more than two periods of the minimum allowed
        // frequency or two periods of the actual estimated
        // period
        if (dt < maxDt && dt * f < 2) {
          rawImg.at<float>(iy, ix) = std::max(T::tf(f), minFreq);
        } else {
          rawImg.at<float>(iy, ix) = minFreq;
        }
      }
    }
    return (rawImg);
  }

  cv::Mat makeRawFrequencyImage()
  {
    if (useLogFrequency_) {
      return (makeTransformedFrequencyImage<LogTF>());
    }
    return (makeTransformedFrequencyImage<NoTF>());
  }

  bool filterNoise(State * state, const Event & newEvent, Event * e_f);

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
  double freq_[2]{-1.0, -1.0};   // frequency range
  double tfFreq_[2]{0, 1.0};     // transformed frequency range
  bool useLogFrequency_{false};  // visualize log10(frequency)
  int numClusters_{0};           // number of freq clusters for image
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
  // ---------- dark noise filtering
  uint64_t noiseFilterDtPass_{0};
  uint64_t noiseFilterDtDead_{0};
  // ------------------ debugging stuff
  uint16_t debugX_{0};
  uint16_t debugY_{0};
};
std::ostream & operator<<(std::ostream & os, const FrequencyCam::Event & e);
}  // namespace event_fourier
#endif  // EVENT_FOURIER__FREQUENCY_CAM_H_
