// -*-c++-*----------------------------------------------------------------------------------------
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

#include "event_fourier/frequency_cam_hilbert.h"

#include <cv_bridge/cv_bridge.h>
#include <event_array_msgs/decode.h>

#include <fstream>
#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>

//#define DEBUG
//#define DEBUG_FREQ
#define PRODUCTION
#define TAKE_LOG

namespace event_fourier
{
#ifdef DEBUG
std::ofstream debug_file("debug_cam.txt");
#endif
#ifdef DEBUG_FREQ
std::ofstream debug_freq("freq_cam.txt");
#endif
FrequencyCamHilbert::FrequencyCamHilbert(const rclcpp::NodeOptions & options)
: Node("event_fourier", options)
{
  c_[0] = 1.95;
  c_[1] = -0.9504;
  c_p_ = 0.98;
  if (!initialize()) {
    RCLCPP_ERROR(this->get_logger(), "frequency cam  startup failed!");
    throw std::runtime_error("startup of FrequencyCamHilbert node failed!");
  }
}

FrequencyCamHilbert::~FrequencyCamHilbert() { delete[] state_; }

bool FrequencyCamHilbert::initialize()
{
  rmw_qos_profile_t qosProf = rmw_qos_profile_default;
#if 0  
  qosProf.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
  qosProf.depth = imageQueueSize;  // keep at most this number of images
  qosProf.reliability = RMW_QOS_POLICY_RELIABILITY_SYSTEM_DEFAULT;
  qosProf.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;  // sender does not have to store
  qosProf.deadline.sec = 5;                                 // max expect time between msgs pub
  qosProf.deadline.nsec = 0;
  qosProf.lifespan.sec = 1;  // how long until msg are considered expired
  qosProf.lifespan.nsec = 0;
  qosProf.liveliness_lease_duration.sec = 10;  // time to declare client dead
  qosProf.liveliness_lease_duration.nsec = 0;
#endif
  imagePub_ = image_transport::create_publisher(this, "~/frequency_image", qosProf);
  sliceTime_ =
    static_cast<uint64_t>((std::abs(this->declare_parameter<double>("slice_time", 0.025) * 1e9)));
  const size_t EVENT_QUEUE_DEPTH(1000);
  auto qos = rclcpp::QoS(rclcpp::KeepLast(EVENT_QUEUE_DEPTH)).best_effort().durability_volatile();
  useSensorTime_ = this->declare_parameter<bool>("use_sensor_time", true);
  const std::string bag = this->declare_parameter<std::string>("bag_file", "");
  freq_[0] = this->declare_parameter<double>("min_frequency", 0.0);
  freq_[1] = this->declare_parameter<double>("max_frequency", 0.0);
  RCLCPP_INFO_STREAM(this->get_logger(), "minimum frequency: " << freq_[0]);
  if (freq_[1] > 0) {
    RCLCPP_INFO_STREAM(this->get_logger(), "maximum frequency: " << freq_[1]);
  }
  dtMix_ = static_cast<float>(this->declare_parameter<double>("dt_averaging", 1e-2));
  dtDecay_ = 1.0 - dtMix_;
  omegaMix_ = static_cast<float>(this->declare_parameter<double>("omega_averaging", 1e-2));
  omegaDecay_ = 1.0 - omegaMix_;
  const std::vector<long> def_roi = {0, 0, 100000, 100000};
  const std::vector<long> roi = this->declare_parameter<std::vector<long>>("roi", def_roi);
  roi_ = std::vector<uint32_t>(roi.begin(), roi.end());  // convert to uint32_t
  if (
    roi_[0] != def_roi[0] || roi_[1] != def_roi[1] || roi_[2] != def_roi[2] ||
    roi_[3] != def_roi[3]) {
    RCLCPP_INFO_STREAM(
      this->get_logger(),
      "using roi: (" << roi_[0] << ", " << roi_[1] << ") w: " << roi_[2] << " h: " << roi_[3]);
  }
  if (bag.empty()) {
    eventSub_ = this->create_subscription<EventArray>(
      "~/events", qos,
      std::bind(&FrequencyCamHilbert::callbackEvents, this, std::placeholders::_1));
    double T = 1.0 / std::max(this->declare_parameter<double>("publishing_frequency", 20.0), 1.0);
    pubTimer_ = rclcpp::create_timer(
      this, this->get_clock(), rclcpp::Duration::from_seconds(T), [=]() { this->publishImage(); });
    statsTimer_ = rclcpp::create_timer(
      this, this->get_clock(), rclcpp::Duration::from_seconds(2.0), [=]() { this->statistics(); });
  } else {
    // reading from bag is only for debugging...
    readEventsFromBag(bag);
  }
  return (true);
}

cv::Mat FrequencyCamHilbert::makeRawFrequencyImage() const
{
  const double lastEventTime = 1e-9 * lastEventTime_;
  cv::Mat rawImg(height_, width_, CV_32FC1, 0.0);
  constexpr double pi2_inv = 1.0 / (2.0 * M_PI);
  // copy data into raw image
  size_t n_stale(0);
  for (uint32_t iy = iyStart_; iy < iyEnd_; iy++) {
    for (uint32_t ix = ixStart_; ix < ixEnd_; ix++) {
      const size_t offset = iy * width_ + ix;
      const State & state = state_[offset];
      // dt: time of no update for this pixel
#ifdef PRODUCTION
      const double freq_clamped = state.omega * (pi2_inv / std::max(state.dt_avg, 1.e-8f));
      const double dt = lastEventTime - state.t;
      // if no events for certain time, no valid!
      const bool isStale = dt * std::max(freq_clamped, freq_[0]) > 2;
      //const double dt_avg_clamped = std::max(state.dt_avg, 1.0e-6f);
      //const double relVar =
      //(state.dt2_avg - state.dt_avg * state.dt_avg) / (dt_avg_clamped * dt_avg_clamped);

      //const bool isValid = state.omega > 0 && relVar < 10.0 && !isStale;
      const bool isValid = state.omega > 0 && !isStale;
      if (isStale) {
        n_stale++;
      }
#ifdef TAKE_LOG
      rawImg.at<float>(iy, ix) = isValid ? std::log10(freq_clamped) : std::log10(freq_[0]);
#else
      rawImg.at<float>(iy, ix) = isValid ? freq_clamped : freq_[0];
#endif
#else
      (void)lastEventTime;
      rawImg.at<float>(iy, ix) =
        (state.dt2_avg - state.dt_avg * state.dt_avg) / std::max(state.dt_avg, 1.0e-6f);
#endif
    }
  }
  //std::cout << "num stale: " << n_stale << std::endl;
  return (rawImg);
}

void FrequencyCamHilbert::publishImage()
{
  if (imagePub_.getNumSubscribers() != 0 && height_ != 0) {
    cv::Mat rawImg = makeRawFrequencyImage();
    cv::Mat scaled;
    double range;
    double minVal;
    if (freq_[0] >= 0 && freq_[1] > freq_[0]) {
      // fixed frequency-to-color mapping
#ifdef PRODUCTION
#ifdef TAKE_LOG
      minVal = log10(freq_[0]);
      range = log10(freq_[1]) - log10(freq_[0]);
#else
      minVal = freq_[0];
      range = freq_[1] - freq_[0];
#endif
#else
      // auto-scale frequency-to-color mapping
      double maxVal;
      cv::Point minLoc, maxLoc;
      cv::minMaxLoc(rawImg, &minVal, &maxVal, &minLoc, &maxLoc);
      range = maxVal - minVal;
#endif
    } else {
      // auto-scale frequency-to-color mapping
      double maxVal;
      cv::Point minLoc, maxLoc;
      cv::minMaxLoc(rawImg, &minVal, &maxVal, &minLoc, &maxLoc);
      std::cout << "min: " << minVal << " max: " << maxVal << std::endl;
      if (freq_[0] > 0) {
        minVal = freq_[0];
      }
      if (freq_[1] > 0) {
        maxVal = freq_[0];
      }
      range = maxVal - minVal;
    }
    cv::convertScaleAbs(rawImg, scaled, 255.0 / range, -minVal * 255.0 / range);
    //const float alpha = 255.0 / (2 * M_PI * (freq_[1] - freq_[0]));
    //cv::convertScaleAbs(rawImg, scaled, alpha, -freq_[0] * 2 * M_PI * alpha);
    cv::Mat colorImg;
    cv::applyColorMap(scaled, colorImg, cv::COLORMAP_JET);
    header_.stamp = lastTime_;
    imagePub_.publish(cv_bridge::CvImage(header_, "bgr8", colorImg).toImageMsg());
  }
}

void FrequencyCamHilbert::readEventsFromBag(const std::string & bagName)
{
  rclcpp::Time t0;
  rosbag2_cpp::Reader reader;
  reader.open(bagName);
  rclcpp::Serialization<event_array_msgs::msg::EventArray> serialization;
#ifdef DEBUG_FREQ
  size_t lastCount = 0;
#endif
  while (reader.has_next()) {
    auto bagmsg = reader.read_next();
    // if (bagmsg->topic_name == topic)
    rclcpp::SerializedMessage serializedMsg(*bagmsg->serialized_data);
    EventArray::SharedPtr msg(new EventArray());
    serialization.deserialize_message(&serializedMsg, &(*msg));
    callbackEvents(msg);
#ifdef DEBUG
    if (eventCount_ > 10000) {
      break;
    }
#endif
#ifdef DEBUG_FREQ
    if (eventCount_ > lastCount + 10000) {
      cv::Mat rawImg = makeRawFrequencyImage();
      for (uint32_t iy = iyStart_; iy < iyEnd_; iy++) {
        for (uint32_t ix = ixStart_; ix < ixEnd_; ix++) {
          debug_freq << " " << rawImg.at<float>(iy, ix);
        }
      }
      debug_freq << std::endl;
      debug_freq.flush();
      lastCount = eventCount_;
    }
#endif
  }
}

void FrequencyCamHilbert::initializeState(uint32_t width, uint32_t height, uint64_t t)
{
  width_ = width;
  height_ = height;
  const variable_t t_sec = t * 1e-9;
  state_ = new State[width * height];
  for (size_t i = 0; i < width * height; i++) {
    state_[i].t = t_sec;
  }
  ixStart_ = std::max(0u, roi_[0]);
  iyStart_ = std::max(0u, roi_[1]);
  ixEnd_ = std::min(width_, roi_[0] + roi_[2]);
  iyEnd_ = std::min(height_, roi_[1] + roi_[3]);
}

void FrequencyCamHilbert::updateState(const uint16_t x, const uint16_t y, uint64_t t, bool polarity)
{
  const size_t offset = y * width_ + x;
  State & s = state_[offset];
  // prefiltering (detrend, accumulate, high pass)
  // x_k has the current filtered signal (log(illumination))
  //
  const auto p = polarity ? 1.0 : -1.0;
  const auto dp = p - s.p;  // raw change in polarity
  const auto x_k = c_[0] * s.x[0] + c_[1] * s.x[1] + c_p_ * dp;
  // compute imaginary arm of quadrature filter
  const auto y_i_0 = h0Filter0_.apply(x_k);  // first section
  const auto y_i = h0Filter1_.apply(y_i_0);  // second section
  // compute real arm of quadrature filter
  const auto y_r_0 = h1Filter0_.apply(x_k);  // first section
  const auto y_r = h1Filter1_.apply(y_r_0);  // second section
  // compute z * conjugate(z[-1]) and phase difference via atan
  const auto dz_r = y_r * s.y_lag_r + y_i * s.y_lag_i;
  const auto dz_i = y_i * s.y_lag_r - y_r * s.y_lag_i;
  const auto omega = std::atan2(dz_i, dz_r);
  // compute how much the difference in angle is w.r.t
  // the predicted angle
  const auto e_omega = omega - s.omega;
  // could avoid conditional by another complex multiplication
  const auto e_omega_wrapped = (e_omega < -M_PI) ? (e_omega + 2 * M_PI) : e_omega;
  // update frequency: new omega measured is s.omega + e_omega_wrapped
  s.omega = s.omega * omegaDecay_ + omegaMix_ * (s.omega + e_omega_wrapped);
  // update rate
  const double t_sec = 1e-9 * t;
  const double dt = t_sec - s.t;
  s.dt_avg = s.dt_avg * dtDecay_ + dtMix_ * dt;
  const double dt2 = dt * dt;
  s.dt2_avg = s.dt2_avg * dtDecay_ + dtMix_ * dt2;
  // advance lagged state of filter
  s.x[1] = s.x[0];
  s.x[0] = x_k;
  s.p = p;
  s.y_lag_r = y_r;
  s.y_lag_i = y_i;
  s.t = t_sec;
#ifdef DEBUG
  debug_file << t << " " << x_k << " " << y_r << " " << y_i << " " << omega << " " << s.omega
             << std::endl;
#endif
}

void FrequencyCamHilbert::callbackEvents(EventArrayConstPtr msg)
{
  const auto t_start = std::chrono::high_resolution_clock::now();
  const auto time_base =
    useSensorTime_ ? msg->time_base : rclcpp::Time(msg->header.stamp).nanoseconds();
  lastTime_ = rclcpp::Time(msg->header.stamp);
  const size_t BYTES_PER_EVENT = 8;

  if (state_ == 0 && !msg->events.empty()) {
    const uint8_t * p = &msg->events[0];
    uint64_t t;
    uint16_t x, y;
    (void)event_array_msgs::mono::decode_t_x_y_p(p, time_base, &t, &x, &y);
    (void)x;
    (void)y;
    initializeState(msg->width, msg->height, t - 1000L /* - 1usec */);
    header_ = msg->header;  // copy frame id
    lastSeq_ = msg->seq - 1;
  }
  const uint8_t * p_base = &msg->events[0];

  uint64_t lastEventTime(0);
  for (const uint8_t * p = p_base; p < p_base + msg->events.size(); p += BYTES_PER_EVENT) {
    uint64_t t;
    uint16_t x, y;
    const bool polarity = event_array_msgs::mono::decode_t_x_y_p(p, time_base, &t, &x, &y);
    lastEventTime = t;
#ifdef DEBUG
    if (x == 319 && y == 239) {
      updateState(x, y, t, polarity);
    }
#else
    updateState(x, y, t, polarity);
#endif
  }
  lastEventTime_ = lastEventTime;
  eventCount_ += msg->events.size() >> 3;
  msgCount_++;
  droppedSeq_ += static_cast<int64_t>(msg->seq) - lastSeq_ - 1;
  lastSeq_ = static_cast<int64_t>(msg->seq);

  const auto t_end = std::chrono::high_resolution_clock::now();
  totTime_ += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
}

void FrequencyCamHilbert::statistics()
{
  if (eventCount_ > 0 && totTime_ > 0) {
    const double usec = static_cast<double>(totTime_);
    RCLCPP_INFO(
      this->get_logger(), "%6.2f Mev/s, %8.2f msgs/s, %8.2f usec/ev  %6.0f usec/msg, drop: %3ld",
      double(eventCount_) / usec, msgCount_ * 1.0e6 / usec, usec / (double)eventCount_,
      usec / msgCount_, droppedSeq_);
    eventCount_ = 0;
    totTime_ = 0;
    msgCount_ = 0;
    droppedSeq_ = 0;
  }
}

}  // namespace event_fourier

RCLCPP_COMPONENTS_REGISTER_NODE(event_fourier::FrequencyCamHilbert)
