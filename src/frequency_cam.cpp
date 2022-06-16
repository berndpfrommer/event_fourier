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

#include "event_fourier/frequency_cam.h"

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

#ifdef DEBUG
std::ofstream debug("freq.txt");
std::ofstream debug_flip("flip.txt");
std::ofstream debug2("debug.txt");
#endif

namespace event_fourier
{
FrequencyCam::FrequencyCam(const rclcpp::NodeOptions & options) : Node("event_fourier", options)
{
  if (!initialize()) {
    RCLCPP_ERROR(this->get_logger(), "frequency cam  startup failed!");
    throw std::runtime_error("startup of FrequencyCam node failed!");
  }
}

FrequencyCam::~FrequencyCam() { delete[] state_; }

static double compute_alpha_prefilter(const double T_cut)
{
  const double omega_c = 2 * M_PI / T_cut;
  const double y = 2 * std::cos(omega_c);
  // set up b and c coeffients for quadratic equation in a
  const double b = 6 * y - 16;
  const double c = y * y - 16 * y + 32;
  const double a = -0.5 * (b + std::sqrt(b * b - 4 * c));
  return (0.5 * (a - std::sqrt(a * a - 4)));
}

// a1: 1.874160776773375 a2: -0.8781196542989451 a3: 0.9685401941933438

bool FrequencyCam::initialize()
{
  rmw_qos_profile_t qosProf = rmw_qos_profile_default;
  imagePub_ = image_transport::create_publisher(this, "~/frequency_image", qosProf);
  sliceTime_ =
    static_cast<uint64_t>((std::abs(this->declare_parameter<double>("slice_time", 0.025) * 1e9)));
  const size_t EVENT_QUEUE_DEPTH(1000);
  auto qos = rclcpp::QoS(rclcpp::KeepLast(EVENT_QUEUE_DEPTH)).best_effort().durability_volatile();
  useSensorTime_ = this->declare_parameter<bool>("use_sensor_time", true);
  const std::string bag = this->declare_parameter<std::string>("bag_file", "");
  freq_[0] = this->declare_parameter<double>("min_frequency", 0.0);
  freq_[0] = std::max(freq_[0], 0.1);
  freq_[1] = this->declare_parameter<double>("max_frequency", -1.0);
  RCLCPP_INFO_STREAM(this->get_logger(), "minimum frequency: " << freq_[0]);
  RCLCPP_INFO_STREAM(this->get_logger(), "maximum frequency: " << freq_[1]);
  dtMix_ = static_cast<float>(this->declare_parameter<double>("dt_averaging_alpha", 0.1));
  dtDecay_ = 1.0 - dtMix_;
  const double T_prefilter =
    std::max(1.0, this->declare_parameter<double>("prefilter_event_cutoff", 40));
  const double alpha_prefilter = compute_alpha_prefilter(T_prefilter);
  const double beta_prefilter = alpha_prefilter;
  c_[0] = alpha_prefilter + beta_prefilter;
  c_[1] = -alpha_prefilter * beta_prefilter;
  c_p_ = 0.5 * (1 + beta_prefilter);
  const double dt_pass = this->declare_parameter<double>("noise_dt_pass", 15.0e-6);
  const double dt_dead = this->declare_parameter<double>("noise_dt_dead", dt_pass);
  noiseFilterDtPass_ = static_cast<uint64_t>(std::abs(dt_pass) * 1e9);
  noiseFilterDtDead_ = static_cast<uint64_t>(std::abs(dt_dead) * 1e9);
  resetThreshold_ = this->declare_parameter<double>("reset_threshold", 5.0);
#ifdef DEBUG
  debugX_ = static_cast<uint16_t>(this->declare_parameter<int>("debug_x", 320));
  debugY_ = static_cast<uint16_t>(this->declare_parameter<int>("debug_y", 240));
#endif

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
      "~/events", qos, std::bind(&FrequencyCam::callbackEvents, this, std::placeholders::_1));
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

void FrequencyCam::readEventsFromBag(const std::string & bagName)
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
  RCLCPP_INFO(this->get_logger(), "finished playing bag");
}

void FrequencyCam::initializeState(uint32_t width, uint32_t height, uint64_t t)
{
  width_ = width;
  height_ = height;
  const variable_t t_sec = t * 1e-9;
  state_ = new State[width * height];
  for (size_t i = 0; i < width * height; i++) {
    state_[i].t = t_sec;
    state_[i].t_flip = t_sec;
  }
  ixStart_ = std::max(0u, roi_[0]);
  iyStart_ = std::max(0u, roi_[1]);
  ixEnd_ = std::min(width_, roi_[0] + roi_[2]);
  iyEnd_ = std::min(height_, roi_[1] + roi_[3]);
}

void FrequencyCam::updateState(State * state, const Event & e)
{
  State & s = *state;
  // prefiltering (detrend, accumulate, high pass)
  // x_k has the current filtered signal (log(illumination))
  //
  const double t_sec = 1e-9 * e.t;
  const auto p = e.polarity ? 1.0 : -1.0;
  const auto dp = p - s.p;  // raw change in polarity
  const auto x_k = c_[0] * s.x[0] + c_[1] * s.x[1] + c_p_ * dp;
#ifdef DEBUG_FULL
  if (e.x == debugX_ && e.y == debugY_) {
    std::cout << (polarity ? "ON" : "OFF") << " event " << t_sec << " avg: " << s.dt_avg
              << std::endl;
  }
#endif
  if (x_k < 0 && s.x[0] >= 0) {
    const double dt = t_sec - s.t_flip;
    if (s.dt_avg <= 0) {  // initialization phase
      if (s.dt_avg == 0) {
        s.dt_avg = std::min(dt, 1.0);
#ifdef DEBUG
        if (e.x == debugX_ && e.y == debugY_) {
          std::cout << "  init avg: " << s.dt_avg << std::endl;
        }
#endif
      } else {
        s.dt_avg = 0;  // signal that on next step dt can be computed
        // std::cout << "  setting dt_avg to zero at " << t_sec << std::endl;
      }
      s.t_flip = t_sec;
    } else {  // not in intialization phase
      if (dt > resetThreshold_ * s.dt_avg) {
        std::cout << t_sec << " restart avg dt: " << dt << " vs " << s.dt_avg
                  << " thresh: " << resetThreshold_ << std::endl;
        s.dt_avg = 0;  // restart
      } else {         // regular case: update
        s.dt_avg = s.dt_avg * dtDecay_ + dtMix_ * dt;
      }
    }
#ifdef DEBUG_FULL
    if (e.x == debugX_ && e.y == debugY_) {
      const double f = 1.0 / std::max(s.dt_avg, 1e-6f);
      std::cout << x_k << " " << (int)s.upper_half << "  dt = " << dt << " avg: " << s.dt_avg
                << " freq: " << f << std::endl;
    }
#endif
#ifdef DEBUG
    if (e.x == debugX_ && e.y == debugY_) {
      debug_flip << std::setprecision(10) << t_sec << " " << dt << " " << s.dt_avg << std::endl;
    }
#endif
    s.t_flip = t_sec;
  }
  s.t = t_sec;
  s.p = p;
  s.x[1] = s.x[0];
  s.x[0] = x_k;
#ifdef DEBUG
  if (e.x == debugX_ && e.y == debugY_) {
    const double dt = t_sec - s.t_flip;
    const double f = s.dt_avg < 1e-6f ? 0 : (1.0 / s.dt_avg);
    debug << t_sec << " " << x_k << " " << (int)s.upper_half << " " << dt << " " << s.dt_avg << " "
          << f << std::endl;
  }
#endif
}

cv::Mat FrequencyCam::makeRawFrequencyImage() const
{
  const double lastEventTime = 1e-9 * lastEventTime_;
  cv::Mat rawImg(height_, width_, CV_32FC1, 0.0);
  // copy data into raw image
  const double maxDt = 1.0 / freq_[0] * 2.0;
  const double logMinFreq = std::log10(freq_[0]);
  for (uint32_t iy = iyStart_; iy < iyEnd_; iy++) {
    for (uint32_t ix = ixStart_; ix < ixEnd_; ix++) {
      const size_t offset = iy * width_ + ix;
      const State & state = state_[offset];
      const double dt = lastEventTime - state.t;
      const double f = 1.0 / std::max(state.dt_avg, 1e-6f);
      if (dt < maxDt && dt * f < 2) {
        rawImg.at<float>(iy, ix) = std::max(std::log10(f), logMinFreq);
      } else {
        rawImg.at<float>(iy, ix) = logMinFreq;
      }
#if 0      
      if (ix == debugX_ && iy == debugY_) {
        std::cout << "raw image: f: " << f << " img: " << rawImg.at<float>(iy, ix)
                  << " lmf: " << logMinFreq << std::endl;
      }
#endif
    }
  }
  return (rawImg);
}

void FrequencyCam::publishImage()
{
  if (imagePub_.getNumSubscribers() != 0 && height_ != 0) {
    cv::Mat rawImg = makeRawFrequencyImage();
    cv::Mat scaled;
    double range;
    double minVal;
    double maxVal;
    if (freq_[1] < 0) {
      cv::Point minLoc, maxLoc;
      cv::minMaxLoc(rawImg, &minVal, &maxVal, &minLoc, &maxLoc);
    } else {
      maxVal = std::log10(freq_[1]);  // override upper bound
    }
    minVal = std::log10(freq_[0]);  // override lower bound no matter what

    range = maxVal - minVal;
    cv::convertScaleAbs(rawImg, scaled, 255.0 / range, -minVal * 255.0 / range);
#if 0    
    std::cout << "minval: " << minVal << " maxval: " << maxVal << std::endl;
    std::cout << "published: " << rawImg.at<float>(debugY_, debugX_)
              << " scaled: " << (int)scaled.at<uint8_t>(debugY_, debugX_) << std::endl;
#endif
    cv::Mat colorImg;

    cv::applyColorMap(scaled, colorImg, cv::COLORMAP_JET);
    header_.stamp = lastTime_;
    imagePub_.publish(cv_bridge::CvImage(header_, "bgr8", colorImg).toImageMsg());
  }
}

/*
def filter_noise(d, dt_pass, dt_dead):
    t = d[:, 0] * 1e-9
    p = d[:, 1].astype(np.int32) * 2 - 1
    d_f = []
    skip_counter = 0
    for i in range(4, d.shape[0]):
        if skip_counter == 0:
            d_f.append((d[i - 4, 0], d[i - 4, 1]))
        else:
            skip_counter -= 1
        if p[i - 2] < 0 and p[i - 1] > 0 and \
           t[i - 1] - t[i - 2] < dt_pass and \
           t[i - 2] - t[i - 3] > dt_dead:
            skip_counter = 4
    n_filt = d.shape[0] - len(d_f)
    print(f'filtered {n_filt} of {d.shape[0]} events ({n_filt/d.shape[0]}%)')
    return np.array(d_f)
*/

bool FrequencyCam::filterNoise(State * s, const Event & newEvent, Event * e_f)
{
  Event * e = s->e;
  bool eventAvailable(false);
  const uint8_t lag_1 = s->idx;
  const uint8_t lag_2 = (s->idx + 3) & 0x03;
  const uint8_t lag_3 = (s->idx + 2) & 0x03;
  const uint8_t lag_4 = (s->idx + 1) & 0x03;

  if (s->skip == 0) {
    eventAvailable = true;
    *e_f = e[lag_4];  // return the one that is 3 events old
#ifdef EXTRA_DEBUG
    //debug2 << e[lag_4].t << " " << (int)e[lag_4].polarity << std::endl;
    debug2 << "accept: " << e[lag_4] << " " << e[lag_3] << " " << e[lag_2] << " " << e[lag_1]
           << std::endl;
#endif
  } else {
    s->skip--;
#ifdef EXTRA_DEBUG
    debug2 << "skip:   " << e[lag_4] << " " << e[lag_3] << " " << e[lag_2] << " " << e[lag_1]
           << std::endl;
#endif
    //const Event & e_s = e[(s->idx + 1) & 0x03];
    //debug2 << "skipping " << e_s.t << " " << (int)e_s.polarity << std::endl;
  }
  // if a DOWN event is followed quickly by an UP event, and
  // if before the DOWN event a significant amount of time has passed,
  // the DOWN/UP is almost certainly a noise event that needs to be
  // filtered out
  if (
    (!e[lag_2].polarity && e[lag_1].polarity) && (e[lag_1].t - e[lag_2].t < noiseFilterDtPass_) &&
    (e[lag_2].t - e[lag_3].t > noiseFilterDtDead_)) {
#ifdef EXTRA_DEBUG
    debug2 << "SKIP:   " << e[lag_4] << " " << e[lag_3] << " " << e[lag_2] << " " << e[lag_1] << " "
           << e[lag_1].t - e[lag_2].t << " vs " << noiseFilterDtPass_ << " "
           << e[lag_2].t - e[lag_3].t << " vs " << noiseFilterDtDead_ << std::endl;
#endif
    s->skip = 4;
  } else {
#ifdef EXTRA_DEBUG
    debug2 << "GOOD:   " << e[lag_4] << " " << e[lag_3] << " " << e[lag_2] << " " << e[lag_1] << " "
           << e[lag_1].t - e[lag_2].t << " vs " << noiseFilterDtPass_ << " "
           << e[lag_2].t - e[lag_3].t << " vs " << noiseFilterDtDead_ << std::endl;
#endif
  }
  // advance circular buffer pointer and store latest event
  s->idx = (s->idx + 1) & 0x03;
  e[s->idx] = newEvent;
  // signal whether a filtered event was produced, i.e *e_f is valid
  return (eventAvailable);
}

void FrequencyCam::callbackEvents(EventArrayConstPtr msg)
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
    Event e;
    e.polarity = event_array_msgs::mono::decode_t_x_y_p(p, time_base, &e.t, &e.x, &e.y);
    lastEventTime = e.t;
    const size_t offset = e.y * width_ + e.x;
    State & s = state_[offset];
    Event e_f;  // filtered event, from the past
    if (filterNoise(&s, e, &e_f)) {
      updateState(&s, e_f);
    }
  }
  lastEventTime_ = lastEventTime;
  eventCount_ += msg->events.size() >> 3;
  msgCount_++;
  droppedSeq_ += static_cast<int64_t>(msg->seq) - lastSeq_ - 1;
  lastSeq_ = static_cast<int64_t>(msg->seq);

  const auto t_end = std::chrono::high_resolution_clock::now();
  totTime_ += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
}

void FrequencyCam::statistics()
{
  if (eventCount_ > 0 && totTime_ > 0) {
    const double usec = static_cast<double>(totTime_);
    RCLCPP_INFO(
      this->get_logger(), "%6.2f Mev/s, %8.2f msgs/s, %8.2f nsec/ev  %6.0f usec/msg, drop: %3ld",
      double(eventCount_) / usec, msgCount_ * 1.0e6 / usec, 1e3 * usec / (double)eventCount_,
      usec / msgCount_, droppedSeq_);
    eventCount_ = 0;
    totTime_ = 0;
    msgCount_ = 0;
    droppedSeq_ = 0;
  }
}

std::ostream & operator<<(std::ostream & os, const FrequencyCam::Event & e)
{
  os << std::fixed << std::setw(10) << std::setprecision(6) << e.t * 1e-9 << " " << (int)e.polarity;
  return (os);
}

}  // namespace event_fourier

RCLCPP_COMPONENTS_REGISTER_NODE(event_fourier::FrequencyCam)
