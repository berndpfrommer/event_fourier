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

#include "event_fourier/event_fourier.h"

#include <cv_bridge/cv_bridge.h>
#include <event_array_msgs/decode.h>

#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>

enum FIELDS { S_OMEGA, S_T, S_X, S_SUM, S_NUM_FIELDS };

namespace event_fourier
{
EventFourier::EventFourier(const rclcpp::NodeOptions & options) : Node("event_fourier", options)
{
  if (!initialize()) {
    RCLCPP_ERROR(this->get_logger(), "event_fourier startup failed!");
    throw std::runtime_error("startup of EventFourier node failed!");
  }
}

EventFourier::~EventFourier() { delete[] state_; }

bool EventFourier::initialize()
{
  const size_t imageQueueSize = 4;
  rmw_qos_profile_t qosProf = rmw_qos_profile_default;
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
  imagePub_ = image_transport::create_publisher(this, "~/image", qosProf);
  sliceTime_ =
    static_cast<uint64_t>((std::abs(this->declare_parameter<double>("slice_time", 0.025) * 1e9)));
  const size_t EVENT_QUEUE_DEPTH(1000);
  auto qos = rclcpp::QoS(rclcpp::KeepLast(EVENT_QUEUE_DEPTH)).best_effort().durability_volatile();
  useSensorTime_ = this->declare_parameter<bool>("use_sensor_time", true);
  alpha_ = 1.0 / this->declare_parameter<double>("window_time", 10.0);  // decay constant
  alpha2_ = alpha_ * alpha_;
  const std::string bag = this->declare_parameter<std::string>("bag_file", "");
  std::vector<double> default_freq = {7.0, 14.0};
  freq_ = this->declare_parameter<std::vector<double>>("frequencies", default_freq);
  if (freq_.size() != 2) {
    RCLCPP_ERROR(this->get_logger(), "must specify exactly 2 frequencies!");
    return (false);
  }
  if (bag.empty()) {
    eventSub_ = this->create_subscription<EventArray>(
      "~/events", qos, std::bind(&EventFourier::callbackEvents, this, std::placeholders::_1));
    double T = 1.0 / std::max(this->declare_parameter<double>("publishing_frequency", 20.0), 1.0);
    pubTimer_ = rclcpp::create_timer(
      this, this->get_clock(), rclcpp::Duration::from_seconds(T), [=]() { this->publishImage(); });

  } else {
    // reading from bag is for debugging...
    readEventsFromBag(bag);
  }
  return (true);
}

void EventFourier::publishImage()
{
  if (imagePub_.getNumSubscribers() != 0 && height_ != 0) {
    std::vector<cv::Mat> images(3);
    images[0] = cv::Mat::zeros(height_, width_, CV_8U);  // blue channel is off
    for (size_t f_idx = 0; f_idx < freq_.size(); f_idx++) {
      cv::Mat rawImg(height_, width_, CV_64FC1);
      // copy data into raw image
      for (uint32_t iy = 0; iy < height_; iy++) {
        for (uint32_t ix = 0; ix < width_; ix++) {
          const size_t offset = ((iy * width_ + ix) * freq_.size() + f_idx) * S_NUM_FIELDS;
          // const complex_t & X = state_[offset + S_X];
          const complex_t & X = state_[offset + S_SUM];
          // approximate detrend correction
          const complex_t X_detrend = state_[offset + S_X] / state_[offset + S_OMEGA];
          const complex_t X_d = X - X_detrend;
          rawImg.at<double>(iy, ix) = std::sqrt(X_d.real() * X_d.real() + X_d.imag() * X_d.imag());
        }
      }
      cv::Mat normImg;
      cv::normalize(rawImg, normImg, 0, 255, cv::NORM_MINMAX, CV_8U);
      cv::equalizeHist(normImg, images[1 + f_idx]);
    }
    // merge images
    header_.stamp = lastTime_;
    cv::Mat merged;
    cv::merge(images, merged);
    imagePub_.publish(cv_bridge::CvImage(header_, "bgr8", merged).toImageMsg());
  }
}

void EventFourier::readEventsFromBag(const std::string & bagName)
{
  rclcpp::Time t0;
  rosbag2_cpp::Reader reader;
  reader.open(bagName);
  rclcpp::Serialization<event_array_msgs::msg::EventArray> serialization;
  while (reader.has_next()) {
    auto bagmsg = reader.read_next();
    // if (bagmsg->topic_name == topic)
    rclcpp::SerializedMessage serializedMsg(*bagmsg->serialized_data);
    EventArray::SharedPtr msg(new EventArray());
    serialization.deserialize_message(&serializedMsg, &(*msg));
    callbackEvents(msg);
  }
}

void EventFourier::resetState(uint32_t width, uint32_t height)
{
  width_ = width;
  height_ = height;
  state_ = new complex_t[width * height * freq_.size() * S_NUM_FIELDS];
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      for (size_t f_idx = 0; f_idx < freq_.size(); f_idx++) {
        const size_t offset = ((i * width + j) * freq_.size() + f_idx) * S_NUM_FIELDS;
        for (size_t k = 0; k < S_NUM_FIELDS; k++) {
          state_[offset + k] = complex_t(0, 0);
        }
        // write frequency and decay constant into image
        state_[offset + S_OMEGA] = complex_t(alpha_, -2 * M_PI * freq_[f_idx]);
      }
    }
  }
}

void EventFourier::updateState(
  const size_t f_idx, const uint16_t x, const uint16_t y, uint64_t t, bool polarity)
{
  const size_t offset = ((y * width_ + x) * freq_.size() + f_idx) * S_NUM_FIELDS;
  complex_t & t_s = state_[offset + S_T];
  const double t_d = 1e-9 * t;
  const double dt = t_d - t_s.real();
  t_s.real(t_d);          // remember time stamp for next time
  if (t_s.imag() == 0) {  // first event for this pixel
    t_s.imag(t_d);        // remember the very first timestamp (for detrending)
    return;               // must wait for second event to have valid dt
  }
  const double dt2 = dt * dt;
  const complex_t amjw = state_[offset + S_OMEGA];  // alpha - j * omega
  const double w = -amjw.imag();                    // omega
  const double w2 = w * w;
  const double a2pw2 = alpha2_ + w2;
  const double a2pw2dt2 = a2pw2 * dt2;  // (alpha^2 + omega^2) * dt^2
  const double adt = alpha_ * dt;
  const double wdt = w * dt;  // omega * dt
  const double dx = (int)polarity * 2 - 1.0;
  const double expma = exp(-adt);  // TODO: approximate for small adt
  const double two_aw = 2.0 * alpha_ * w;
  const double a2mw2 = alpha2_ - w2;
  const double x_km1 = state_[offset + S_X].real();  // x[k - 1]
  const complex_t expiwdt = complex_t(cos(wdt), sin(wdt));
  if (a2pw2dt2 < 1e-4) {
    // we are using some complex arithmetic here
    const complex_t s = amjw * dt;  // (alpha - j * omega) * dt
    const complex_t s2 = s * s;
    const complex_t g = complex_t(1.0, 0) - 0.5 * s + 0.16666666667 * s2;
    const complex_t h = complex_t(0.5, 0) - 0.16666666667 * s + 0.04166666667 * s2;
    const complex_t d = dt * (x_km1 * g + dx * h);
    state_[offset + S_SUM] = d + expma * expiwdt * state_[offset + S_SUM];
  } else {
    // use mostly real arithmetic here
    const double a2pw2_inv = 1.0 / a2pw2;
    const double a2pw2dt_inv = 1.0 / (a2pw2 * dt);
    const double sinwdt = sin(wdt);
    const double coswdt = cos(wdt);
    const double expma_sinwdt = expma * sinwdt;
    const double expma_coswdt = expma * coswdt;
    const double expma_sinwdt_m_wdt = expma_sinwdt - wdt;
    const double expma_coswdt_p_adt_m_1 = expma_coswdt + adt - 1;
    const complex_t gdt = a2pw2_inv * complex_t(
                                        alpha_ + w * expma_sinwdt - alpha_ * expma_coswdt,
                                        w - alpha_ * expma_sinwdt - w * expma * coswdt);
    const complex_t hdt = a2pw2_inv * a2pw2dt_inv *
                          complex_t(
                            a2mw2 * expma_coswdt_p_adt_m_1 - two_aw * expma_sinwdt_m_wdt,
                            a2mw2 * expma_sinwdt_m_wdt + two_aw * expma_coswdt_p_adt_m_1);
    const complex_t d = x_km1 * gdt + dx * hdt;
    state_[offset + S_SUM] = d + expma * expiwdt * state_[offset + S_SUM];
  }
  state_[offset + S_X] += dx;  // update x[k - 1] -> x[k]
}

void EventFourier::callbackEvents(EventArrayConstPtr msg)
{
  const auto time_base =
    useSensorTime_ ? msg->time_base : rclcpp::Time(msg->header.stamp).nanoseconds();
  lastTime_ = rclcpp::Time(msg->header.stamp);
  const size_t BYTES_PER_EVENT = 8;
  size_t start_event = 0;
  if (state_ == 0) {
    resetState(msg->width, msg->height);
    uint64_t t;
    uint16_t x, y;
    (void)event_array_msgs::mono::decode_t_x_y_p(&msg->events[0], time_base, &t, &x, &y);
    start_event = BYTES_PER_EVENT;  // skip first event of first packet
    header_ = msg->header;          // copy frame id
  }

  const auto t_start = std::chrono::high_resolution_clock::now();
  const uint8_t * p_base = &msg->events[0];

  for (const uint8_t * p = p_base + start_event; p < p_base + msg->events.size();
       p += BYTES_PER_EVENT) {
    uint64_t t;
    uint16_t x, y;
    const bool polarity = event_array_msgs::mono::decode_t_x_y_p(p, time_base, &t, &x, &y);
    for (size_t f_idx = 0; f_idx < freq_.size(); f_idx++) {
      updateState(f_idx, x, y, t, polarity);
    }
  }
  const auto t_end = std::chrono::high_resolution_clock::now();
  eventCount_ += msg->events.size();
  totTime_ += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
  if (eventCount_ > lastCount_ + 10000000) {
    std::cout << "avg perf: " << double(eventCount_) / double(totTime_) << std::endl;
    lastCount_ = eventCount_;
  }
}

}  // namespace event_fourier

RCLCPP_COMPONENTS_REGISTER_NODE(event_fourier::EventFourier)
