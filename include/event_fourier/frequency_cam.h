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

#include <event_fourier/synchronized_buffer.h>
#include <stdlib.h>

#include <cstdlib>
#include <event_array_msgs/msg/event_array.hpp>
#include <image_transport/image_transport.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <thread>
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
  struct PackedVar
  {
    void setPolarity(bool p) { packed = (packed & 0x7F) | (p << 7); }
    void startSkip() { packed = (packed & ~0x7) | 0x4; }
    void decSkip() { packed = (packed & ~0x7) | ((packed & 0x7) - 1); }
    inline bool p() const { return (static_cast<bool>(packed & 0x80)); }
    inline uint8_t idx() const { return ((packed >> 3) & 0x3); }
    void setIdx(uint8_t i) { packed = (packed & ~0x18) | (i << 3); }
    inline uint8_t skip() const { return (packed & 0x7); }
    uint8_t packed{0};
  };
  struct TimeAndPolarity  // keep time and polarity in one 32 bit variable
  {
    void set(uint32_t t_usec, bool p) { t_and_p = (t_usec & 0x7FFFFFFF) | (p << 31); }
    inline bool p() const { return (static_cast<bool>(t_and_p & ~(0x7FFFFFFF))); }
    inline uint32_t t() const { return (t_and_p & 0x7FFFFFFF); }
    uint32_t t_and_p;
  };

  struct Event  // event representation for convenience
  {
    Event(uint32_t ta = 0, uint16_t xa = 0, uint16_t ya = 0, bool p = false)
    : t(ta), x(xa), y(ya), polarity(p)
    {
    }
    inline void setTimeAndPolarity(const TimeAndPolarity & tp)
    {
      polarity = tp.p();
      t = tp.t();
    }
    uint32_t t;
    uint16_t x;
    uint16_t y;
    bool polarity;
  };
  friend std::ostream & operator<<(std::ostream & os, const Event & e);
  struct State  // per-pixel filter state
  {
    // 4 * 4  = 16 bytes
    TimeAndPolarity tp[4];  // circular buffer for noise filter
    //
    uint32_t t_flip;    // time of last flip
    uint32_t t;         // last time stamp
    variable_t x[2];    // current and lagged signal x
    variable_t dt_avg;  // average sample time (time between events)
    // could collapse 6 bytes into 1, reducing
    // total to 32 + 21 = 53 bytes
    // these 4 bytes could be collapsed into 1 bit
    PackedVar p_skip_idx;  // polarity, skip cnt, idx
  };

  using EventArray = event_array_msgs::msg::EventArray;
  using EventArrayConstPtr = EventArray::ConstSharedPtr;
  void readEventsFromBag(const std::string & bagName);
  bool initialize();
  void initializeState(uint32_t width, uint32_t height, uint32_t t);
  void callbackEvents(EventArrayConstPtr msg);
  std::vector<float> findLegendValuesAndText(
    const double minVal, const double maxVal, const std::vector<float> & centers,
    std::vector<std::string> * text) const;

  void addLegend(
    cv::Mat * img, const double minVal, const double maxVal,
    const std::vector<float> & centers) const;
  void publishImage();
  void statistics();
  void updateState(State * state, const Event & e);
  struct NoTF
  {
    static double tf(double f) { return (f); }
    static double inv(double f) { return (f); }
  };
  struct LogTF
  {
    static double tf(double f) { return (std::log10(f)); }
    static double inv(double f) { return (std::pow(10.0, f)); }
  };
  struct EventFrameUpdater
  {
    static void update(cv::Mat * img, int ix, int iy, double dt, double dtMax)
    {
      if (dt < dtMax) {
        img->at<uint8_t>(iy, ix) = 255;
      }
    }
  };
  struct NoEventFrameUpdater
  {
    static void update(cv::Mat *, int, int, double, double){};
  };
  template <class T, class U>
  cv::Mat makeTransformedFrequencyImage(cv::Mat * eventFrame) const
  {
    cv::Mat rawImg(height_, width_, CV_32FC1, 0.0);
    const double maxDt = 1.0 / freq_[0] * 2.0;
    const double minFreq = T::tf(freq_[0]);
    for (uint32_t iy = iyStart_; iy < iyEnd_; iy++) {
      for (uint32_t ix = ixStart_; ix < ixEnd_; ix++) {
        const size_t offset = iy * width_ + ix;
        const State & state = state_[offset];
        const double dt = (lastEventTime_ - state.t) * 1e-6;
        const double f = 1.0 / std::max(state.dt_avg, 1e-6f);
        // filter out any pixels that have not been updated
        // for more than two periods of the minimum allowed
        // frequency or two periods of the actual estimated
        // period
        U::update(eventFrame, ix, iy, dt, eventImageDt_);
        if (dt < maxDt && dt * f < 2) {
          rawImg.at<float>(iy, ix) = std::max(T::tf(f), minFreq);
        } else {
          rawImg.at<float>(iy, ix) = 0;  // mark as invalid
        }
      }
    }
    return (rawImg);
  }

  cv::Mat makeFrequencyAndEventImage(cv::Mat * eventImage);

  bool filterNoise(State * state, const Event & newEvent, Event * e_f);
  void startThreads();
  void worker(unsigned int id);
  uint32_t updateMultiThreaded(uint64_t timeBase, const std::vector<uint8_t> & events);
  uint32_t updateSingleThreaded(uint64_t timeBase, const std::vector<uint8_t> & events);

  // ------ variables ----
  rclcpp::Time lastTime_{0};
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
  double tfFreq_[2]{0, 1.0};    // transformed frequency range
  uint32_t width_;              // image width
  uint32_t height_;             // image height
  uint64_t eventCount_{0};
  uint64_t msgCount_{0};
  uint64_t lastCount_{0};
  uint64_t totTime_{0};
  uint32_t lastEventTime_;
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
  uint32_t noiseFilterDtPass_{0};
  uint32_t noiseFilterDtDead_{0};
  // ---------- multithreading
  std::vector<std::thread> threads_;
  std::atomic<bool> keepRunning_{true};
  using EventBuffer = SynchronizedBuffer;
  std::vector<EventBuffer> eventBuffer_;
  // ---------- visualization
  bool useLogFrequency_{false};                   // visualize log10(frequency)
  int numClusters_{0};                            // number of freq clusters for image
  bool printClusterCenters_{false};               // if cluster centers should be printed
  int legendWidth_{0};                            // width of legend in pixels
  std::vector<double> legendValues_;              // frequency values for which to show legend
  size_t legendBins_;                             // # of bins if legend values are not given
  cv::ColormapTypes colorMap_{cv::COLORMAP_JET};  // colormap for freq
  double eventImageDt_{0};                        // time slice for event visualization
  bool overlayEvents_{false};
  //
  // ------------------ debugging stuff
  uint16_t debugX_{0};
  uint16_t debugY_{0};
};
std::ostream & operator<<(std::ostream & os, const FrequencyCam::Event & e);
}  // namespace event_fourier
#endif  // EVENT_FOURIER__FREQUENCY_CAM_H_
