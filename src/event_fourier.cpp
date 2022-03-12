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

static const uint8_t NUM_FREQ = 2;  // number of frequencies to keep

#define COMPUTE_IMAGE_HIST

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
  const std::string bag = this->declare_parameter<std::string>("bag_file", "");
  std::vector<double> default_freq = {3.0, 100.0};
  freq_ = this->declare_parameter<std::vector<double>>("frequencies", default_freq);
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
  if (freq_.size() != 2) {
    RCLCPP_ERROR(this->get_logger(), "must specify exactly 2 frequencies!");
    return (false);
  }
  RCLCPP_INFO_STREAM(this->get_logger(), "frequencies: " << freq_[0] << " - " << freq_[1]);
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
    const uint8_t f_idx = 0;
    cv::Mat rawImg(height_, width_, CV_32FC1, static_cast<float>(freq_[0] * 2 * M_PI));
    // copy data into raw image
    const uint32_t ix_start = std::max(0u, roi_[0]);
    const uint32_t iy_start = std::max(0u, roi_[1]);
    const uint32_t ix_end = std::min(width_, roi_[0] + roi_[2]);
    const uint32_t iy_end = std::min(height_, roi_[1] + roi_[3]);
    for (uint32_t iy = iy_start; iy < iy_end; iy++) {
      for (uint32_t ix = ix_start; ix < ix_end; ix++) {
        const size_t offset = ((iy * width_ + ix) * NUM_FREQ + f_idx) * S_NUM_FIELDS;
        // copy dominant frequency into image
        rawImg.at<float>(iy, ix) = -state_[offset + S_OMEGA].imag();
      }
    }

    cv::Mat scaled;
    const float alpha = 255.0 / (2 * M_PI * (freq_[1] - freq_[0]));
    cv::convertScaleAbs(rawImg, scaled, alpha, -freq_[0] * 2 * M_PI * alpha);
    cv::Mat colorImg;
    cv::applyColorMap(scaled, colorImg, cv::COLORMAP_JET);
    header_.stamp = lastTime_;
    imagePub_.publish(cv_bridge::CvImage(header_, "bgr8", colorImg).toImageMsg());
#ifdef COMPUTE_IMAGE_HIST
    std::cout << "----------------------" << std::endl;
    int channels[1] = {0};  // only one channel
    cv::MatND hist;
    const int hbins = 50;
    const int histSize[1] = {hbins};
    const float franges[2] = {
      static_cast<float>(freq_[0] * 2 * M_PI), static_cast<float>(freq_[1] * 2 * M_PI)};
    //const double franges[2] = {freq_[0], freq_[1]};
    const float * ranges[1] = {franges};
    cv::calcHist(
      &rawImg, 1 /* num images */, channels, mask_, hist, 1 /* dim of hist */,
      histSize /* num bins */, ranges /* frequency range */, true /* histogram is uniform */,
      false /* don't accumulate */);
    for (int i = 0; i < hbins; i++) {
      printf(
        "%6.2f %8.0f\n", freq_[0] + (freq_[1] - freq_[0]) * (float)i / hbins, hist.at<float>(i));
    }
#endif
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

void EventFourier::initializeState(uint32_t x, uint32_t y, uint8_t f_idx, double f)
{
  const size_t offset = ((y * width_ + x) * NUM_FREQ + f_idx) * S_NUM_FIELDS;
  // TODO(Bernd): use memset for this
  for (size_t k = 0; k < S_NUM_FIELDS; k++) {
    state_[offset + k] = complex_t(0, 0);
  }
  // write frequency and decay constant into image
  const double alpha = 0.1 * f;  // 10 cycles to establish frequency
  state_[offset + S_OMEGA] = complex_t(alpha, -2 * M_PI * f);
}

void EventFourier::copyState(uint32_t x, uint32_t y, uint8_t f_src, uint8_t f_dest)
{
  const size_t offset_src = ((y * width_ + x) * NUM_FREQ + f_src) * S_NUM_FIELDS;
  const size_t offset_dest = ((y * width_ + x) * NUM_FREQ + f_dest) * S_NUM_FIELDS;
  // TODO (Bernd): use memcpy for this
  for (size_t k = 0; k < S_NUM_FIELDS; k++) {
    state_[offset_dest + k] = state_[offset_src + k];
  }
}

void EventFourier::resetState(uint32_t width, uint32_t height)
{
  width_ = width;
  height_ = height;
  state_ = new complex_t[width * height * NUM_FREQ * S_NUM_FIELDS];
  for (size_t iy = 0; iy < height; iy++) {
    for (size_t ix = 0; ix < width; ix++) {
      for (uint8_t f_idx = 0; f_idx < NUM_FREQ; f_idx++) {
        initializeState(ix, iy, f_idx, getRandomFreq());
      }
    }
  }
#ifdef COMPUTE_IMAGE_HIST
  if (roi_[0] != 0 || roi_[1] != 0 || roi_[0] + roi_[2] != width_ || roi_[1] + roi_[3] != height_) {
    // set mask
    mask_ = cv::Mat::zeros(height_, width_, CV_8U);
    cv::Rect rect(
      roi_[0], roi_[1], std::min(roi_[2], width_ - roi_[0]), std::min(roi_[3], height_ - roi_[1]));
    mask_(rect) = 1;
  }
#endif
}

void EventFourier::updateState(
  const uint8_t f_idx, const uint16_t x, const uint16_t y, uint64_t t, bool polarity)
{
  const size_t offset = ((y * width_ + x) * NUM_FREQ + f_idx) * S_NUM_FIELDS;
  const double alpha = state_[offset + S_OMEGA].real();
  const double alpha2 = alpha * alpha;
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
  const double a2pw2 = alpha2 + w2;
  const double a2pw2dt2 = a2pw2 * dt2;  // (alpha^2 + omega^2) * dt^2
  const double adt = alpha * dt;
  const double wdt = w * dt;  // omega * dt
  const double dx = (int)polarity * 2 - 1.0;
  const double expma = exp(-adt);  // TODO: approximate for small adt
  const double two_aw = 2.0 * alpha * w;
  const double a2mw2 = alpha2 - w2;
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
                                        alpha + w * expma_sinwdt - alpha * expma_coswdt,
                                        w - alpha * expma_sinwdt - w * expma * coswdt);
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
    for (uint8_t f_idx = 0; f_idx < NUM_FREQ; f_idx++) {
      updateState(f_idx, x, y, t, polarity);
    }
    double amplitude[NUM_FREQ];
    double dta[NUM_FREQ];
    for (uint8_t f_idx = 0; f_idx < NUM_FREQ; f_idx++) {
      const size_t offset = ((y * width_ + x) * NUM_FREQ + f_idx) * S_NUM_FIELDS;
      const complex_t & X = state_[offset + S_SUM];
      //// approximate detrend correction
      const complex_t X_detrend = state_[offset + S_X] / state_[offset + S_OMEGA];
      const complex_t X_d = X - X_detrend;
      amplitude[f_idx] = X_d.real() * X_d.real() + X_d.imag() * X_d.imag();
      // T: total time elapsed
      const double T = state_[offset + S_T].real() - state_[offset + S_T].imag();
      dta[f_idx] = state_[offset + S_OMEGA].real() * T;
    }
    // if amplitude of test frequency is higher, use that
    if (dta[0] > 1.0 && dta[1] > 1.0 && amplitude[1] > amplitude[0]) {
      // transfer state from slot 1 -> 0
      copyState(x, y, 1, 0);
      // restart trial with random frequency;
      initializeState(x, y, 1, getRandomFreq());
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
