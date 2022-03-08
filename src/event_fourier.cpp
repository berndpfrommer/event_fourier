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
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>

enum FIELDS { S_OMEGA, S_X, S_SUM, S_NUM_FIELDS };

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
  if (bag.empty()) {
    eventSub_ = this->create_subscription<EventArray>(
      "~/events", qos, std::bind(&EventFourier::callbackEvents, this, std::placeholders::_1));
  } else {
    readEventsFromBag(bag);
  }
  return (true);
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
  state_ = new complex_t[width * height * S_NUM_FIELDS];
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      const size_t offset = (i * width + j) * S_NUM_FIELDS;
      for (size_t k = 0; k < S_NUM_FIELDS; k++) {
        state_[offset + k] = complex_t(0, 0);
      }
      const double f = 7.0;  // frequency in hz
      //const double f = 0.0;  // frequency in hz
      state_[offset + S_OMEGA] = complex_t(alpha_, -2 * M_PI * f);
    }
  }
}

void EventFourier::callbackEvents(EventArrayConstPtr msg)
{
  const auto time_base =
    useSensorTime_ ? msg->time_base : rclcpp::Time(msg->header.stamp).nanoseconds();
  const size_t BYTES_PER_EVENT = 8;
  size_t start_event = 0;
  if (state_ == 0) {
    resetState(msg->width, msg->height);
    uint64_t t;
    uint16_t x, y;
    (void)event_array_msgs::mono::decode_t_x_y_p(&msg->events[0], time_base, &t, &x, &y);
    lastTime_ = t;
    start_event = BYTES_PER_EVENT;  // skip first event of first packet
  }

  const auto t_start = std::chrono::high_resolution_clock::now();
  const uint8_t * p_base = &msg->events[0];

  for (const uint8_t * p = p_base + start_event; p < p_base + msg->events.size();
       p += BYTES_PER_EVENT) {
    uint64_t t;
    uint16_t x, y;
    const bool polarity = event_array_msgs::mono::decode_t_x_y_p(p, time_base, &t, &x, &y);
    const uint32_t offset = (y * width_ + x) * S_NUM_FIELDS;
    const double dt = (t - lastTime_) * 1e-9;
    lastTime_ = t;
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
