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

// The simple event image was necessary to compare with other data sources such
// as the metavision SDK's analytics modules. It means that the events drawn are
// based on a simple frame rather than the more sophisticated criterion of
// having had an event within the currently estimated frequency period.
// NOTE: using this option decreases performance substantially!

#define SIMPLE_EVENT_IMAGE

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
    //  7  65  43  210
    //  p   a   i    s
    //  o   v   d    k
    //  l   g   x    i
    //               p

    // polarity of previous event (filtering)
    void setPolarity(bool p) { packed = (packed & 0x7F) | (p << 7); }
    inline bool p() const { return (static_cast<bool>(packed & 0x80)); }

    // index into previous events (noise filtering)
    void setIdx(uint8_t i) { packed = (packed & ~0x18) | (i << 3); }
    inline uint8_t idx() const { return ((packed >> 3) & 0x3); }

    // skipping event (noise filtering)
    void startSkip() { packed = (packed & ~0x7) | 0x4; }
    void decSkip() { packed = (packed & ~0x7) | ((packed & 0x7) - 1); }
    inline uint8_t skip() const { return (packed & 0x7); }

    // for counting down #of events in average
    void resetBadDataCount() { packed = (packed & ~0x60); }
    void incBadDataCount() { packed = (packed & ~0x60) | ((((packed >> 5) & 0x3) + 1) << 5); }
    inline bool badDataCountLimitReached() const { return (((packed >> 5) & 0x3) == 3); }
    inline bool hasBadData() const { return (((packed >> 5) & 0x3) != 0); }
    uint8_t getBadDataCount() const { return ((packed >> 5) & 0x3); }
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
    variable_t x[2];    // current and lagged signal x
    variable_t dt_avg;  // average sample time (time between events)
    PackedVar p_skip_idx;  // polarity, skip cnt, idx
    uint8_t avg_cnt{0};    // number of dt til good average
#ifdef SIMPLE_EVENT_IMAGE
    uint32_t last_update;  // shortened time of last update
#endif
  };

  using EventArray = event_array_msgs::msg::EventArray;
  using EventArrayConstPtr = EventArray::ConstSharedPtr;
  void playEventsFromBag(const std::string & bagName);
  void playEventsFromBagExtTimeStamps(const std::string & bagName);
  bool initialize();
  void initializeState(uint32_t width, uint32_t height, uint32_t t);
  void callbackEvents(EventArrayConstPtr msg);
  std::vector<float> findLegendValuesAndText(
    const double minVal, const double maxVal, const std::vector<float> & centers,
    std::vector<std::string> * text) const;

  void addLegend(
    cv::Mat * img, const double minVal, const double maxVal,
    const std::vector<float> & centers) const;
  cv::Mat makeImage() const;
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
        // state.avg_cnt is zero when enough updates
        // have been compounded into the average
        const double dt = (lastEventTime_ - state.t_flip) * 1e-6;
#ifdef SIMPLE_EVENT_IMAGE
        (void)minFreq;  // suppress compiler warning/error
        const double dtEv = (lastEventTime_ - state.last_update) * 1e-6;
        U::update(eventFrame, ix, iy, dtEv, eventImageDt_);
#else
        U::update(eventFrame, ix, iy, dt, eventImageDt_);
#endif
        if (!state.avg_cnt) {
          const double f = 1.0 / std::max(state.dt_avg, 1e-6f);
          // filter out any pixels that have not been updated
          // for more than two periods of the minimum allowed
          // frequency or two periods of the actual estimated
          // period
          if (dt < maxDt && dt * f < 2) {
            rawImg.at<float>(iy, ix) = std::max(T::tf(f), minFreq);
          } else {
            rawImg.at<float>(iy, ix) = 0;  // mark as invalid
          }
        }
      }
    }
    return (rawImg);
  }

  cv::Mat makeFrequencyAndEventImage(cv::Mat * eventImage) const;

  bool filterNoise(State * state, const Event & newEvent, Event * e_f);
  void startThreads();
  void worker(unsigned int id);
  uint32_t updateMultiThreaded(uint64_t timeBase, const std::vector<uint8_t> & events);
  uint32_t updateSingleThreaded(uint64_t timeBase, const std::vector<uint8_t> & events);
  inline void filterAndUpdateState(State * s, const Event & e)
  {
    if (useTemporalNoiseFilter_) {
      Event e_f(e);  // filtered event, from the past
      if (filterNoise(s, e, &e_f)) {
        updateState(s, e_f);
      }
    } else {
      updateState(s, e);
    }
  }
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
  // ---------- variables for state update
  variable_t c_[2];
  variable_t c_p_{0};
  variable_t dtMix_{1.0 / 100.0};
  variable_t dtDecay_{1 - dtMix_};
  variable_t resetThreshold_{0.2};
  variable_t stalePixelThreshold_{10.0};
  variable_t dtMin_{0};
  variable_t dtMax_{1.0};
  uint8_t numGoodCyclesRequired_{3};
  // ---------- dark noise filtering
  uint32_t noiseFilterDtPass_{0};
  uint32_t noiseFilterDtDead_{0};
  bool useTemporalNoiseFilter_{false};
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
