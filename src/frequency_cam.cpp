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
#include <event_camera_codecs/decoder.h>
#include <math.h>

#include <algorithm>  // std::sort, std::stable_sort, std::clamp
#include <filesystem>
#include <fstream>
#include <image_transport/image_transport.hpp>
#include <numeric>  // std::iota
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <sstream>
#include <thread>

#define DEBUG
//#define PRINT_IMAGE_HISTOGRAM

//#define NAIVE
#define AVERAGING

#ifdef DEBUG
std::ofstream debug("freq.txt");
std::ofstream debug_flip("flip.txt");
std::ofstream debug_readout("readout.txt");
#endif

namespace event_fourier
{
using EventPacket = event_camera_msgs::msg::EventPacket;

FrequencyCam::FrequencyCam(const rclcpp::NodeOptions & options) : Node("frequency_cam", options)
{
  if (!initialize()) {
    RCLCPP_ERROR(get_logger(), "frequency cam  startup failed!");
    throw std::runtime_error("startup of FrequencyCam node failed!");
  }
}

FrequencyCam::~FrequencyCam() { delete[] state_; }

static void compute_alpha_beta(const double T_cut, double * alpha, double * beta)
{
  const double omega_cut = 2 * M_PI / T_cut;
  const double phi = 2 - std::cos(omega_cut);
  *alpha = (1.0 - std::sin(omega_cut)) / std::cos(omega_cut);
  *beta = phi - std::sqrt(phi * phi - 1.0);  // see paper
}

bool FrequencyCam::initialize()
{
  rmw_qos_profile_t qosProf = rmw_qos_profile_default;
  imagePub_ = image_transport::create_publisher(this, "~/frequency_image", qosProf);
  const size_t EVENT_QUEUE_DEPTH(1000);
  auto qos = rclcpp::QoS(rclcpp::KeepLast(EVENT_QUEUE_DEPTH)).best_effort().durability_volatile();
  useSensorTime_ = this->declare_parameter<bool>("use_sensor_time", true);
  useTemporalNoiseFilter_ = this->declare_parameter<bool>("use_temporal_noise_filter", false);
  const std::string bag = this->declare_parameter<std::string>("bag_file", "");
  freq_[0] = this->declare_parameter<double>("min_frequency", 1.0);
  freq_[0] = std::max(freq_[0], 0.1);
  freq_[1] = this->declare_parameter<double>("max_frequency", -1.0);
  dtMax_ = 1.0 / freq_[0];
  dtMin_ = 1.0 / (freq_[1] >= freq_[0] ? freq_[1] : 1.0);
  RCLCPP_INFO_STREAM(get_logger(), "minimum frequency: " << freq_[0]);
  RCLCPP_INFO_STREAM(get_logger(), "maximum frequency: " << freq_[1]);
  useLogFrequency_ = this->declare_parameter<bool>("use_log_frequency", false);
  tfFreq_[0] = useLogFrequency_ ? LogTF::tf(std::max(freq_[0], 1e-8)) : freq_[0];
  tfFreq_[1] = useLogFrequency_ ? LogTF::tf(std::max(freq_[1], 1e-7)) : freq_[1];
  numClusters_ = this->declare_parameter<int>("num_frequency_clusters", 0);
  if (numClusters_ > 0) {
    RCLCPP_INFO_STREAM(get_logger(), "number of frequency clusters: " << numClusters_);
  } else {
    RCLCPP_INFO_STREAM(get_logger(), "NOT clustering by frequency!");
  }
  printClusterCenters_ = this->declare_parameter<bool>("print_cluster_centers", false);
  legendWidth_ = this->declare_parameter<int>("legend_width", 100);
  legendValues_ =
    this->declare_parameter<std::vector<double>>("legend_frequencies", std::vector<double>());
  if (legendValues_.empty()) {
    legendBins_ = this->declare_parameter<int>("legend_bins", 11);
  } else {
    if (numClusters_ > 0) {
      RCLCPP_WARN(get_logger(), "ignoring legend_frequencies when clustering!");
      legendValues_.clear();
    }
  }
  timeoutCycles_ = this->declare_parameter<int>("num_timeout_cycles", 2.0);
  overlayEvents_ = this->declare_parameter<bool>("overlay_events", false);
  dtMix_ = std::clamp(
    static_cast<float>(this->declare_parameter<double>("dt_averaging_alpha", 0.1)), 0.001f, 1.000f);
  dtDecay_ = 1.0 - dtMix_;
  const double T_prefilter =
    std::max(1.0, this->declare_parameter<double>("prefilter_event_cutoff", 40));
  double alpha_prefilter, beta_prefilter;
  compute_alpha_beta(T_prefilter, &alpha_prefilter, &beta_prefilter);

  c_[0] = alpha_prefilter + beta_prefilter;
  c_[1] = -alpha_prefilter * beta_prefilter;
  c_p_ = 0.5 * (1 + beta_prefilter) * alpha_prefilter;
  const double dt_pass = this->declare_parameter<double>("noise_dt_pass", 15.0e-6);
  const double dt_dead = this->declare_parameter<double>("noise_dt_dead", dt_pass);
  noiseFilterDtPass_ = static_cast<uint32_t>(std::abs(dt_pass) * 1e6);
  noiseFilterDtDead_ = static_cast<uint32_t>(std::abs(dt_dead) * 1e6);
  resetThreshold_ = this->declare_parameter<double>("reset_threshold", 0.2);
  stalePixelThreshold_ = this->declare_parameter<double>("stale_pixel_threshold", 10.0);
  numGoodCyclesRequired_ = static_cast<uint8_t>(this->declare_parameter<int>(
    "num_good_cycles_required", std::clamp((int)(1.0 / dtMix_), 0, 254)));
  RCLCPP_INFO_STREAM(
    get_logger(), static_cast<int>(numGoodCyclesRequired_) << " good cycles required");
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
      get_logger(),
      "using roi: (" << roi_[0] << ", " << roi_[1] << ") w: " << roi_[2] << " h: " << roi_[3]);
  }

  eventImageDt_ =
    1.0 / std::max(this->declare_parameter<double>("publishing_frequency", 20.0), 1.0);

  if (bag.empty()) {
    eventSub_ = this->create_subscription<EventPacket>(
      "~/events", qos, std::bind(&FrequencyCam::callbackEvents, this, std::placeholders::_1));
    pubTimer_ = rclcpp::create_timer(
      this, this->get_clock(), rclcpp::Duration::from_seconds(eventImageDt_),
      [=]() { this->publishImage(); });
    statsTimer_ = rclcpp::create_timer(
      this, this->get_clock(), rclcpp::Duration::from_seconds(2.0), [=]() { this->statistics(); });
  } else {
    // reading from bag is only for debugging
    playEventsFromBag(bag);
  }
  return (true);
}

void FrequencyCam::playEventsFromBag(const std::string & bagName)
{
  rclcpp::Time lastFrameTime(0);
  rosbag2_cpp::Reader reader;
  reader.open(bagName);
  rclcpp::Serialization<EventPacket> serialization;
  const auto delta_t = rclcpp::Duration::from_seconds(eventImageDt_);
  bool hasValidTime = false;
  uint32_t frameCount(0);
  const std::string path = this->declare_parameter<std::string>("path", "./frames");
  std::filesystem::create_directories(path);

  while (reader.has_next()) {
    auto bagmsg = reader.read_next();
    rclcpp::SerializedMessage serializedMsg(*bagmsg->serialized_data);
    EventPacket::SharedPtr msg(new EventPacket());
    serialization.deserialize_message(&serializedMsg, &(*msg));
    if (msg) {
      const rclcpp::Time t(msg->header.stamp);
      callbackEvents(msg);
      if (hasValidTime) {
        if (t - lastFrameTime > delta_t) {
          const cv::Mat img = makeImage((lastFrameTime + delta_t).nanoseconds());
          lastFrameTime = lastFrameTime + delta_t;
          char fname[256];
          snprintf(fname, sizeof(fname) - 1, "/frame_%05u.jpg", frameCount);
          cv::imwrite(path + fname, img);
          frameCount++;
        }
      } else {
        hasValidTime = true;
        lastFrameTime = t;
      }
    } else {
      RCLCPP_WARN(get_logger(), "skipped invalid message type in bag!");
    }
  }
  RCLCPP_INFO(get_logger(), "finished playing bag");
}

void FrequencyCam::initializeState(uint32_t width, uint32_t height, uint32_t t)
{
  RCLCPP_INFO_STREAM(
    get_logger(), "state image size is: " << (width * height * sizeof(State)) / (1 << 20)
                                          << "MB (better fit into CPU cache)");
  width_ = width;
  height_ = height;
  state_ = new State[width * height];
  for (size_t i = 0; i < width * height; i++) {
    State & s = state_[i];
    s.t_flip_up_down = t;
    s.t_flip_down_up = t;
    s.x[0] = 0;
    s.x[1] = 0;
#if defined(NAIVE) || !defined(AVERAGING)
    s.avg_cnt = 0;
#else
    s.avg_cnt = numGoodCyclesRequired_;
#endif
    s.dt_avg = -1;
#ifdef SIMPLE_EVENT_IMAGE
    s.last_update = 0;
#endif
    s.p_skip_idx = PackedVar();
    s.p_skip_idx.resetBadDataCount();
    // initialize lagged state
    for (size_t j = 0; j < 4; j++) {
      s.tp[j].set(t, false);
    }
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
  // raw change in polarity, will be 0 or +-2
  const float dp = 2 * (static_cast<int8_t>(e.polarity) - static_cast<int8_t>(s.p_skip_idx.p()));
  // run the filter (see paper)
  const auto x_k = c_[0] * s.x[0] + c_[1] * s.x[1] + c_p_ * dp;
#ifdef SIMPLE_EVENT_IMAGE
  s.last_update = e.t;
#endif
  if (x_k < 0 && s.x[0] > 0) {
    // measure period upon transition from upper to lower half, i.e.
    // when OFF events happen. This is more precise than on the other flank
    const float dt_ud = (e.t - s.t_flip_up_down) * 1e-6;
    if (dt_ud >= dtMin_ && dt_ud <= dtMax_) {
      s.dt_avg = dt_ud;
    } else {
      const float dt_du = (e.t - s.t_flip_down_up) * 1e-6;
      if (s.dt_avg > 0) {
        if (dt_ud > s.dt_avg * timeoutCycles_ && dt_du > 0.5 * s.dt_avg * timeoutCycles_) {
          s.dt_avg = 0;  // not heard from this pixel in a long time, erase average
        }
      } else {
        if (dt_du >= 0.5 * dtMin_ && dt_du <= 0.5 * dtMax_) {
          s.dt_avg = 2 * dt_du;  // don't have any average, make do with half-cycle
        }
      }
    }
    s.t_flip_up_down = e.t;
#ifdef DEBUG
    if (e.x == debugX_ && e.y == debugY_) {
      debug_flip << std::setprecision(10) << e.t << " " << dt_ud << " " << s.dt_avg << " "
                 << " " << x_k << " " << s.x[0] << std::endl;
    }
#endif
  } else if (x_k > 0 && s.x[0] < 0) {
    // use lower to upper transition if required. Less precise though.
    const float dt_du = (e.t - s.t_flip_down_up) * 1e-6;
    if (dt_du >= dtMin_ && dt_du <= dtMax_ && s.dt_avg <= 0) {
      s.dt_avg = dt_du;
    } else {
      const float dt_ud = (e.t - s.t_flip_up_down) * 1e-6;
      if (s.dt_avg > 0) {
        if (dt_du > s.dt_avg * timeoutCycles_ && dt_ud > 0.5 * s.dt_avg * timeoutCycles_) {
          s.dt_avg = 0;  // not heard from this pixel in a long time, erase average
        }
      } else {
        if (dt_ud >= 0.5 * dtMin_ && dt_ud <= 0.5 * dtMax_) {
          s.dt_avg = 2 * dt_ud;  // don't have any average, make do with half-cycle
        }
      }
    }
    s.t_flip_down_up = e.t;
#ifdef DEBUG
    if (e.x == debugX_ && e.y == debugY_) {
      debug_flip << std::setprecision(10) << e.t << " " << dt_du << " " << s.dt_avg << " "
                 << " " << x_k << " " << s.x[0] << std::endl;
    }
#endif
  }
#ifdef DEBUG
  if (e.x == debugX_ && e.y == debugY_) {
    const double dt = (e.t - s.t_flip_up_down) * 1e-6;
    debug << e.t << " " << dp << " " << x_k << " " << s.x[0] << " " << s.x[1] << " " << dt << " "
          << s.dt_avg << std::endl;
  }
#endif
  s.p_skip_idx.setPolarity(e.polarity);
  s.x[1] = s.x[0];
  s.x[0] = x_k;
}

static void compute_max(const cv::Mat & img, double * maxVal)
{
  {
    // no max frequency specified, calculate highest frequency
    cv::Point minLoc, maxLoc;
    double minVal;
    cv::minMaxLoc(img, &minVal, maxVal, &minLoc, &maxLoc);
  }
}

static void get_valid_pixels(
  const cv::Mat & img, float minVal, float maxVal, std::vector<float> * values,
  std::vector<cv::Point> * locations)
{
  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      const double v = img.at<float>(y, x);
      if (v > minVal && v <= maxVal) {
        values->push_back(v);
        locations->push_back(cv::Point(x, y));
      }
    }
  }
}

// sorting with index from stackoverflow:
// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T>
std::vector<size_t> sort_indexes(
  const std::vector<T> & v, std::vector<size_t> * inverse_idx, std::vector<T> * sorted_v)
{
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  for (size_t i = 0; i < idx.size(); i++) {
    (*inverse_idx)[idx[i]] = i;
    sorted_v->push_back(v[idx[i]]);
  }
  return (idx);
}

/*
 * Find frequency labels by clustering
 */
static cv::Mat label_image(
  const cv::Mat & freqImg, float minVal, float maxVal, int K, std::vector<float> * centers)
{
  std::vector<float> validPixels;
  std::vector<cv::Point> locations;
  get_valid_pixels(freqImg, minVal, maxVal, &validPixels, &locations);

  if ((int)validPixels.size() <= K) {
    // not enough valid pixels to cluster
    return (cv::Mat::zeros(freqImg.size(), CV_32FC1));
  }
  const int attempts = 3;
  std::vector<int> labels;
  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
  std::vector<float> unsorted_centers;
  (void)kmeans(validPixels, K, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, unsorted_centers);
  std::vector<size_t> reverse_idx(unsorted_centers.size());
  std::vector<size_t> idx = sort_indexes(unsorted_centers, &reverse_idx, centers);
  cv::Mat labeled = cv::Mat::zeros(freqImg.size(), CV_32FC1);
  for (size_t i = 0; i < validPixels.size(); i++) {
    labeled.at<float>(locations[i].y, locations[i].x) = reverse_idx[labels[i]] + 1;
  }
  return (labeled);
}

#ifdef PRINT_IMAGE_HISTOGRAM
/*
 * experimental code used during debugging of clustering
 */
static void print_image_histogram(cv::Mat & img, const int hbins, float minVal, float maxVal)
{
  int channels[1] = {0};  // only one channel
  cv::MatND hist;
  const int histSize[1] = {hbins};
  const float franges[2] = {minVal, maxVal};
  const float * ranges[1] = {franges};
  cv::calcHist(
    &img, 1 /* num images */, channels, cv::Mat(), hist, 1 /* dim of hist */,
    histSize /* num bins */, ranges /* frequency range */, true /* histogram is uniform */,
    false /* don't accumulate */);
  for (int i = 0; i < hbins; i++) {
    double lv = minVal + (maxVal - minVal) * (float)i / hbins;
    printf("%6.2f %8.0f\n", lv, hist.at<float>(i));
  }
}
#endif

/*
 * format frequency labels for opencv
 */
static std::string format_freq(double v)
{
  std::stringstream ss;
  //ss << " " << std::fixed << std::setw(7) << std::setprecision(1) << v;
  ss << " " << std::setw(6) << (int)v;
  return (ss.str());
}

static void draw_labeled_rectangle(
  cv::Mat * window, int x_off, int y_off, int height, const std::string & text,
  const cv::Vec3b & color)
{
  const cv::Point tl(x_off, y_off);                      // top left
  const cv::Point br(window->cols - 1, y_off + height);  // bottom right
  cv::rectangle(*window, tl, br, color, -1 /* filled */, cv::LINE_8);
  // place text
  const cv::Point tp(x_off - 2, y_off + height / 2 - 2);
  const cv::Scalar textColor = CV_RGB(0, 0, 0);
  cv::putText(
    *window, text, tp, cv::FONT_HERSHEY_PLAIN, 1.0, textColor, 2.0 /*thickness*/,
    cv::LINE_AA /* anti-alias */);
}

std::vector<float> FrequencyCam::findLegendValuesAndText(
  const double minVal, const double maxVal, const std::vector<float> & centers,
  std::vector<std::string> * text) const
{
  std::vector<float> values;
  if (legendValues_.empty()) {
    // no explicit legend values are provided
    if (numClusters_ == 0) {
      // if data is not labeled use equidistant bins between min and max val.
      // This could be either in linear or log space.
      // If the range of values is large, round to next integer
      const double range = maxVal - minVal;
      bool round_it = !useLogFrequency_ && (range / (legendBins_ - 1)) > 2.0;
      for (size_t i = 0; i < legendBins_; i++) {
        const double raw_v = minVal + (static_cast<float>(i) / (legendBins_ - 1)) * range;
        const double v = round_it ? std::round(raw_v) : raw_v;
        values.push_back(v);
        text->push_back(format_freq(useLogFrequency_ ? LogTF::inv(v) : v));
      }
    } else {
      // data is labeled (clustered). In this case the value is actually the
      // cluster label, i.e. an integer ranged 0 ... num_clusters - 1
      for (size_t i = 0; i < centers.size(); i++) {
        values.push_back(i);
        const double v = useLogFrequency_ ? LogTF::inv(centers[i]) : centers[i];
        text->push_back(format_freq(v));
      }
    }
  } else {
    // legend values are explicitly given
    for (const auto & lv : legendValues_) {
      values.push_back(useLogFrequency_ ? LogTF::tf(lv) : lv);
      text->push_back(format_freq(lv));
    }
  }
  return (values);
}

void FrequencyCam::addLegend(
  cv::Mat * window, const double minVal, const double maxVal,
  const std::vector<float> & centers) const
{
  const int x_off = window->cols - legendWidth_;  // left border of legend
  const double range = maxVal - minVal;
  std::vector<std::string> text;
  std::vector<float> values = findLegendValuesAndText(minVal, maxVal, centers, &text);
  if (!values.empty()) {
    cv::Mat valueMat(values.size(), 1, CV_32FC1);
    for (size_t i = 0; i < values.size(); i++) {
      valueMat.at<float>(i, 0) = values[i];
    }
    // rescale values matrix the same way as original image
    cv::Mat scaledValueMat;
    cv::convertScaleAbs(valueMat, scaledValueMat, 255.0 / range, -minVal * 255.0 / range);
    cv::Mat colorCode;
    cv::applyColorMap(scaledValueMat, colorCode, colorMap_);
    // draw filled rectangles and text labels
    const int height = window->rows / values.size();  // integer division
    for (size_t i = 0; i < values.size(); i++) {
      const int y_off = static_cast<float>(i) / values.size() * window->rows;
      draw_labeled_rectangle(window, x_off, y_off, height, text[i], colorCode.at<cv::Vec3b>(i, 0));
    }
  } else {
    // for some reason or the other (usually clustering failed), no legend could be drawn
    cv::Mat roiLegend = (*window)(cv::Rect(x_off, 0, legendWidth_, window->rows));
    roiLegend.setTo(CV_RGB(0, 0, 0));
  }
}

cv::Mat FrequencyCam::makeFrequencyAndEventImage(cv::Mat * eventImage) const
{
  if (overlayEvents_) {
    *eventImage = cv::Mat::zeros(height_, width_, CV_8UC1);
  }
  if (useLogFrequency_) {
    return (
      overlayEvents_ ? makeTransformedFrequencyImage<LogTF, EventFrameUpdater>(eventImage)
                     : makeTransformedFrequencyImage<LogTF, NoEventFrameUpdater>(eventImage));
  }
  return (
    overlayEvents_ ? makeTransformedFrequencyImage<NoTF, EventFrameUpdater>(eventImage)
                   : makeTransformedFrequencyImage<NoTF, NoEventFrameUpdater>(eventImage));
}

cv::Mat FrequencyCam::makeImage(uint64_t t) const
{
  cv::Mat eventImg;
  cv::Mat rawImg = makeFrequencyAndEventImage(&eventImg);
#ifdef DEBUG
  const double v = rawImg.at<float>(debugY_, debugX_);
  debug_readout << t * 1e-9 << " " << (useLogFrequency_ ? LogTF::inv(v) : NoTF::inv(v))
                << std::endl;
#endif
  cv::Mat scaled;
  double minVal = tfFreq_[0];
  double maxVal = tfFreq_[1];
  std::vector<float> centers;
  if (numClusters_ == 0) {
    if (freq_[1] < 0) {
      compute_max(rawImg, &maxVal);
    }
#ifdef PRINT_IMAGE_HISTOGRAM
    print_image_histogram(rawImg, 10, minVal, maxVal);
#endif
    const double range = maxVal - minVal;
    cv::convertScaleAbs(rawImg, scaled, 255.0 / range, -minVal * 255.0 / range);
  } else {
    cv::Mat labeled = label_image(rawImg, minVal, maxVal, numClusters_, &centers);
    minVal = 0;
    maxVal = numClusters_ - 1;
    const double range = maxVal - minVal;
    cv::convertScaleAbs(labeled, scaled, 255.0 / range, -minVal * 255.0 / range);
    if (!centers.empty() && printClusterCenters_) {
      // print out cluster centers to console
      std::stringstream ss;
      for (const auto & c : centers) {
        ss << " " << std::fixed << std::setw(10) << std::setprecision(6) << c;
      }
      RCLCPP_INFO(get_logger(), ss.str().c_str());
    }
  }
  cv::Mat window(scaled.rows, scaled.cols + legendWidth_, CV_8UC3);
  cv::Mat colorImg = window(cv::Rect(0, 0, scaled.cols, scaled.rows));
  cv::applyColorMap(scaled, colorImg, colorMap_);
  colorImg.setTo(CV_RGB(0, 0, 0), rawImg == 0);  // render invalid points black
  if (overlayEvents_) {
    const cv::Scalar eventColor = CV_RGB(127, 127, 127);
    // only show events where no frequency is detected
    colorImg.setTo(eventColor, (rawImg == 0) & eventImg);
  }
  if (legendWidth_ > 0) {
    addLegend(&window, minVal, maxVal, centers);
  }
  return (window);
}

void FrequencyCam::publishImage()
{
  if (imagePub_.getNumSubscribers() != 0 && height_ != 0) {
    const cv::Mat window = makeImage(this->get_clock()->now().nanoseconds());
    header_.stamp = lastTime_;
    imagePub_.publish(cv_bridge::CvImage(header_, "bgr8", window).toImageMsg());
  }
}

bool FrequencyCam::filterNoise(State * s, const Event & newEvent, Event * e_f)
{
  auto const * tp = s->tp;  // time and polarity
  auto * tp_nc = s->tp;     // non-const
  bool eventAvailable(false);
  const uint8_t idx = s->p_skip_idx.idx();
  const uint8_t lag_1 = idx;  // alias to make code more readable
  const uint8_t lag_2 = (idx + 3) & 0x03;
  const uint8_t lag_3 = (idx + 2) & 0x03;
  const uint8_t lag_4 = (idx + 1) & 0x03;

  if (s->p_skip_idx.skip() == 0) {
    eventAvailable = true;
    e_f->setTimeAndPolarity(tp[lag_4]);  // return the one that is 3 events old
  } else {
    s->p_skip_idx.decSkip();  // decrement skip variable
  }
  // if a DOWN event is followed quickly by an UP event, and
  // if before the DOWN event a significant amount of time has passed,
  // the DOWN/UP is almost certainly a noise event that needs to be
  // filtered out
  if (
    (!tp[lag_2].p() && tp[lag_1].p()) && (tp[lag_1].t() - tp[lag_2].t() < noiseFilterDtPass_) &&
    (tp[lag_2].t() - tp[lag_3].t() > noiseFilterDtDead_)) {
    s->p_skip_idx.startSkip();
  }
  // advance circular buffer pointer and store latest event
  s->p_skip_idx.setIdx(lag_4);
  tp_nc[lag_4].set(newEvent.t, newEvent.polarity);
  // signal whether a filtered event was produced, i.e if *e_f is valid
  return (eventAvailable);
}

void FrequencyCam::callbackEvents(EventPacketConstPtr msg)
{
  const auto t_start = std::chrono::high_resolution_clock::now();
  lastTime_ = rclcpp::Time(msg->header.stamp);
  auto decoder = decoderFactory_.getInstance(msg->encoding, msg->width, msg->height);
  if (state_ == 0 && !msg->events.empty()) {
    uint64_t t;
    if (!decoder->findFirstSensorTime(*msg, &t)) {
      return;
    }
    initializeState(msg->width, msg->height, shorten_time(t) - 1 /* - 1usec */);
    header_ = msg->header;  // copy frame id
    lastSeq_ = msg->seq - 1;
  }
  decoder->decode(*msg, this);
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
      get_logger(), "%6.2f Mev/s, %8.2f msgs/s, %8.2f nsec/ev  %6.0f usec/msg, drop: %3ld",
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
  os << std::fixed << std::setw(10) << std::setprecision(6) << e.t * 1e-6 << " " << (int)e.polarity
     << " " << e.x << " " << e.y;
  return (os);
}

}  // namespace event_fourier

RCLCPP_COMPONENTS_REGISTER_NODE(event_fourier::FrequencyCam)
