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

enum FIELDS { S_OMEGA, S_T, S_X, S_SUM, S_AMP2, S_NUM_FIELDS };

static const uint8_t NUM_FREQ = 2;  // number of frequencies to keep

//#define COMPUTE_IMAGE_HIST

namespace event_fourier
{
using complex_t = event_fourier::EventFourier::complex_t;

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
  windowSizeInCycles_ = this->declare_parameter<double>("window_size_in_cycles", 25.0);
  RCLCPP_INFO_STREAM(this->get_logger(), "window size in cycles: " << windowSizeInCycles_);
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

void EventFourier::printFrequencies()
{
  const uint32_t ix_start = std::max(0u, roi_[0]);
  const uint32_t iy_start = std::max(0u, roi_[1]);
  const uint32_t ix_end = std::min(width_, roi_[0] + roi_[2]);
  const uint32_t iy_end = std::min(height_, roi_[1] + roi_[3]);
  printf("event counter: %zu ------\n", eventCount_);
  for (uint32_t iy = iy_start; iy < iy_end; iy++) {
    for (uint32_t ix = ix_start; ix < ix_end; ix++) {
      printf("%3d %3d ", ix, iy);
      for (uint8_t f_idx = 0; f_idx < NUM_FREQ; f_idx++) {
        const size_t offset = ((iy * width_ + ix) * NUM_FREQ + f_idx) * S_NUM_FIELDS;
        printf(
          " f: %8.2f amp: %10.4e adt: %10.4e", -state_[offset + S_OMEGA].imag() / (2 * M_PI),
          state_[offset + S_AMP2].real(), state_[offset + S_AMP2].imag());
      }
      //std::cout << std::endl;
      printf("\n");
    }
  }
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
  const double alpha = f / windowSizeInCycles_;  // determines cycles to establish frequency
  // write frequency and decay constant into image
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

void EventFourier::copyState(
  uint32_t x_src, uint32_t y_src, uint8_t f_src, uint32_t x_dest, uint32_t y_dest, uint8_t f_dest)
{
  const size_t offset_src = ((y_src * width_ + x_src) * NUM_FREQ + f_src) * S_NUM_FIELDS;
  const size_t offset_dest = ((y_dest * width_ + x_dest) * NUM_FREQ + f_dest) * S_NUM_FIELDS;
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
#if 0        
        if (ix == 319 and iy == 239) {
          initializeState(ix, iy, f_idx, 2000.0);
        } else {
          initializeState(ix, iy, f_idx, getRandomFreq());
        }
#else
        initializeState(ix, iy, f_idx, getRandomFreq());
#endif
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

inline static complex_t dt_dx_h_exact(
  const double adt, const double a2mw2, const double two_aw, const double dx, const double dt,
  const double wdt, const double a2pw2_inv, const double expma_sinwdt, const double expma_coswdt)
{
  // use mostly real arithmetic here
  const double a2pw2dt_inv = a2pw2_inv / dt;
  const double expma_sinwdt_m_wdt = expma_sinwdt - wdt;
  const double expma_coswdt_p_adt_m_1 = expma_coswdt + adt - 1;
  const complex_t hdt = a2pw2_inv * a2pw2dt_inv *
                        complex_t(
                          a2mw2 * expma_coswdt_p_adt_m_1 - two_aw * expma_sinwdt_m_wdt,
                          a2mw2 * expma_sinwdt_m_wdt + two_aw * expma_coswdt_p_adt_m_1);
  return (dx * hdt);
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
#ifdef DEBUG
  if (f_idx == 0 && x == 319 && y == 239) {
    std::cout << "t: " << std::setprecision(8) << t_d << " dt: " << dt << " dx: " << dx
              << " adt: " << adt << " wdt: " << wdt;
  }
#endif
  if (a2pw2dt2 < 1e-4) {
    // we are using some complex arithmetic here
    const complex_t s = amjw * dt;  // (alpha - j * omega) * dt
    const complex_t s2 = s * s;
    const complex_t g = complex_t(1.0, 0) - 0.5 * s + 0.16666666667 * s2;
    const complex_t h = complex_t(0.5, 0) - 0.16666666667 * s + 0.04166666667 * s2;
    const complex_t d = dt * (x_km1 * g + dx * h);
    state_[offset + S_SUM] = d + expma * expiwdt * state_[offset + S_SUM];
#ifdef DEBUG
    if (f_idx == 0 && x == 319 && y == 239) {
      std::cout << " sum: " << state_[offset + S_SUM] << " d(app): " << d
                << " oldsum: " << expma * expiwdt * state_[offset + S_SUM] << std::endl;
    }
#endif
  } else {
    // use mostly real arithmetic here
    const double a2pw2_inv = 1.0 / a2pw2;  // common
    const double a2pw2dt_inv = 1.0 / (a2pw2 * dt);
    const double sinwdt = sin(wdt);
    const double coswdt = cos(wdt);
    const double expma_sinwdt = expma * sinwdt;  // common
    const double expma_coswdt = expma * coswdt;  // common
    const complex_t gdt = a2pw2_inv * complex_t(
                                        alpha + w * expma_sinwdt - alpha * expma_coswdt,
                                        w - alpha * expma_sinwdt - w * expma * coswdt);
    const double expma_sinwdt_m_wdt = expma_sinwdt - wdt;
    const double expma_coswdt_p_adt_m_1 = expma_coswdt + adt - 1;
    const complex_t hdt = a2pw2_inv * a2pw2dt_inv *
                          complex_t(
                            a2mw2 * expma_coswdt_p_adt_m_1 - two_aw * expma_sinwdt_m_wdt,
                            a2mw2 * expma_sinwdt_m_wdt + two_aw * expma_coswdt_p_adt_m_1);
    const complex_t d = x_km1 * gdt + dx * hdt;
    state_[offset + S_SUM] = d + expma * expiwdt * state_[offset + S_SUM];
#ifdef DEBUG
    if (f_idx == 0 && x == 319 && y == 239) {
      std::cout << " sum: " << state_[offset + S_SUM] << " d(exa): " << d
                << " oldsum: " << expma * expiwdt * state_[offset + S_SUM] << std::endl;
      std::cout << "   a2pw2_inv:  " << a2pw2_inv << std::endl;
      std::cout << "   a2p2dt_inv: " << a2pw2dt_inv << std::endl;
      std::cout << "   sinwdt: " << sinwdt << std::endl;
      std::cout << "   coswdt: " << coswdt << std::endl;
      std::cout << "   expma: " << expma << std::endl;
      std::cout << "   x_km1: " << x_km1 << std::endl;
      std::cout << "   dx: " << dx << std::endl;
      std::cout << " exact: gdt: " << gdt << " hdt: " << hdt << " d: " << d << std::endl;
    }
#endif
  }
  complex_t & x_k = state_[offset + S_X];
  x_k.real(x_k.real() + dx);  // update x[k - 1] -> x[k]
#ifdef DEBUG
  if (f_idx == 0 && x == 319 && y == 239) {
    std::cout << " updated x to " << state_[offset + S_X].real() << " by: " << dx << std::endl;
    std::cout << t_d << " updated STATE SUM to " << state_[offset + S_SUM] << std::endl;
  }
#endif
  //
  // -----------  detrend and compute amplitude squared ---------------
  //
  const complex_t & Xtilde = state_[offset + S_SUM];  // original ~X, no norm/phase yet
  // dT is total time elapsed
  const double dT = state_[offset + S_T].real() - state_[offset + S_T].imag();
  const double adT = alpha * dT;
  if (adT > 1e-2) {  // no point computing the amplitude w/o enough data
    const double wdT = w * dT;
    const double sinwdT = sin(wdT);
    const double coswdT = cos(wdT);
    const double expma_sinwdT = expma * sinwdT;
    const double expma_coswdT = expma * coswdT;
    const double a2pw2_inv = 1.0 / a2pw2;

    const complex_t X_detrend = dt_dx_h_exact(
      adT, a2mw2, two_aw, x_k.real(), dT, w * dT, a2pw2_inv, expma_sinwdT, expma_coswdT);

    const double nf = alpha / (1.0 - exp(-adT));
    const complex_t X_d = (Xtilde - X_detrend) * nf;
    // update amplitude
    state_[offset + S_AMP2].real(X_d.real() * X_d.real() + X_d.imag() * X_d.imag());
  }

  // store elapsed_time * alpha which counts number of cycles
  state_[offset + S_AMP2].imag(adT);

#ifdef DEBUG
  if (f_idx == 0 && x == 319 && y == 239) {
    //std::cout << "   amp2: " << state_[offset + S_AMP2] << std::endl;
    std::cout << t_d << " updated STATE SUM to " << state_[offset + S_SUM]
              << " amp2: " << state_[offset + S_AMP2].real() << std::endl;
  }
#endif
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
    header_ = msg->header;  // copy frame id
  }
  eventCount_ += msg->events.size();

  const auto t_start = std::chrono::high_resolution_clock::now();
  const uint8_t * p_base = &msg->events[0];

  bool frequencyChanged(false);
  for (const uint8_t * p = p_base + start_event; p < p_base + msg->events.size();
       p += BYTES_PER_EVENT) {
    uint64_t t;
    uint16_t x, y;
    const bool polarity = event_array_msgs::mono::decode_t_x_y_p(p, time_base, &t, &x, &y);
    for (uint8_t f_idx = 0; f_idx < NUM_FREQ; f_idx++) {
      updateState(f_idx, x, y, t, polarity);
    }
    const size_t off_0 = ((y * width_ + x) * NUM_FREQ + 0) * S_NUM_FIELDS;
    const complex_t & amp_0 = state_[off_0 + S_AMP2];
    if (amp_0.imag() > 1.0) {  // enough time has passed
      const size_t off_1 = ((y * width_ + x) * NUM_FREQ + 1) * S_NUM_FIELDS;
      const complex_t & amp_1 = state_[off_1 + S_AMP2];
      if (amp_1.imag() > 1.0) {  // test frequency is also valid
        frequencyChanged = true;
        if (amp_1.real() > amp_0.real()) {
          // amplitude of test freq is higher than reference freq, set ref_freq = test_freq
          // transfer state from slot 1 -> 0
#ifdef DEBUG
          if (x == 319 && y == 239) {
            std::cout << "XXX overwriting freq!" << std::endl;
          }
#endif
          copyState(x, y, 1, 0);
          // restart test frequency with random frequency;
          initializeState(x, y, 1, getRandomFreq());
        } else {
          // restart test frequency with random frequency;
          initializeState(x, y, 1, getRandomFreq());
        }
      }
    }

#if 0
      else {
        // if any of the neighboring ref frequencies has higher amplitude, use that one.
        for (uint32_t iy = std::max(static_cast<uint32_t>(y) - 1, 0u);
             iy < std::min(static_cast<uint32_t>(y) + 2, height_); iy++) {
          for (uint32_t ix = std::max(static_cast<uint32_t>(x) - 1, 0u);
               ix < std::min(static_cast<uint32_t>(x) + 2, width_); ix++) {
            const size_t off = ((iy * width_ + ix) * NUM_FREQ) * S_NUM_FIELDS;
            const complex_t & amp = state_[off + S_AMP2];
            if (amp.imag() > 1.0 && amp.real() > amp_0.real()) {
              // transfer state from neighboring pixel
              // copyState(ix, iy, 0, x, y, 0);
              // restart with neighbor's frequency
              initializeState(x, y, 0, -state_[off + S_OMEGA].imag() / (2.0 * M_PI));
            }
          }
        }
      }
#endif
  }
  if (frequencyChanged) {
    printFrequencies();
  }
  const auto t_end = std::chrono::high_resolution_clock::now();
  totTime_ += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
  if (eventCount_ > lastCount_ + 10000000) {
    std::cout << "avg perf: " << double(eventCount_) / double(totTime_) << std::endl;
    lastCount_ = eventCount_;
  }
}

}  // namespace event_fourier

RCLCPP_COMPONENTS_REGISTER_NODE(event_fourier::EventFourier)
