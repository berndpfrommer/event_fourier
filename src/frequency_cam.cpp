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
#include <math.h>

#include <algorithm>  // std::sort, std::stable_sort
#include <fstream>
#include <image_transport/image_transport.hpp>
#include <numeric>  // std::iota
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <sstream>
#include <thread>

//#define DEBUG
//#define PRINT_IMAGE_HISTOGRAM

#ifdef DEBUG
std::ofstream debug("freq.txt");
std::ofstream debug_flip("flip.txt");
#endif

namespace event_fourier
{
FrequencyCam::FrequencyCam(const rclcpp::NodeOptions & options) : Node("frequency_cam", options)
{
  if (!initialize()) {
    RCLCPP_ERROR(get_logger(), "frequency cam  startup failed!");
    throw std::runtime_error("startup of FrequencyCam node failed!");
  }
}

FrequencyCam::~FrequencyCam()
{
  // need to send signal to worker threads that they can exit
  keepRunning_ = false;
  for (auto & b : eventBuffer_) {
    b.writer_done();
  }
  // then harvest threads
  for (auto & th : threads_) {
    th.join();
  }
  delete[] state_;
}

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

static uint32_t shorten_time(uint64_t t)
{
  return (static_cast<uint32_t>((t / 1000) & 0xFFFFFFFF));
}

bool FrequencyCam::initialize()
{
  rmw_qos_profile_t qosProf = rmw_qos_profile_default;
  imagePub_ = image_transport::create_publisher(this, "~/frequency_image", qosProf);
  const size_t EVENT_QUEUE_DEPTH(1000);
  auto qos = rclcpp::QoS(rclcpp::KeepLast(EVENT_QUEUE_DEPTH)).best_effort().durability_volatile();
  useSensorTime_ = this->declare_parameter<bool>("use_sensor_time", true);
  const std::string bag = this->declare_parameter<std::string>("bag_file", "");
  freq_[0] = this->declare_parameter<double>("min_frequency", 1.0);
  freq_[0] = std::max(freq_[0], 0.1);
  freq_[1] = this->declare_parameter<double>("max_frequency", -1.0);
  dtMax_ = 1.0 / freq_[0];
  dtMin_ = 1.0 / (freq_[1] >= freq_[0] ? freq_[1] : 1.0);
  std::cout << "dtmin: " << dtMin_ << " dtmax: " << dtMax_ << std::endl;
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
  overlayEvents_ = this->declare_parameter<bool>("overlay_events", false);
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
  noiseFilterDtPass_ = static_cast<uint32_t>(std::abs(dt_pass) * 1e6);
  noiseFilterDtDead_ = static_cast<uint32_t>(std::abs(dt_dead) * 1e6);
  resetThreshold_ = this->declare_parameter<double>("reset_threshold", 0.2);
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
  startThreads();
  if (bag.empty()) {
    eventSub_ = this->create_subscription<EventArray>(
      "~/events", qos, std::bind(&FrequencyCam::callbackEvents, this, std::placeholders::_1));
    double T = 1.0 / std::max(this->declare_parameter<double>("publishing_frequency", 20.0), 1.0);
    eventImageDt_ = T;
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

void FrequencyCam::startThreads()
{
  const int numThreads = std::min(
    std::thread::hardware_concurrency(),
    static_cast<unsigned int>(this->declare_parameter<int>("worker_threads", 24)));

  if (numThreads >= 1) {
    RCLCPP_INFO_STREAM(get_logger(), "running with " << numThreads << " threads");
    eventBuffer_.resize(numThreads);  // create one buffer per thread
    threads_.resize(numThreads);
    for (auto id = decltype(numThreads){0}; id < numThreads; id++) {
      threads_[id] = std::thread(&FrequencyCam::worker, this, id);
    }
  } else {
    RCLCPP_INFO_STREAM(get_logger(), "running single threaded!");
  }
}

void FrequencyCam::worker(unsigned int id)
{
  auto & b = eventBuffer_[id];
  while (keepRunning_) {
    b.wait_until_write_complete();
    const std::vector<uint8_t> * events;
    uint64_t timeBase = b.get_events(&events);
    const uint8_t * p_base = &(*events)[0];
    for (const uint8_t * p = p_base; p < p_base + events->size();
         p += event_array_msgs::mono::bytes_per_event) {
      Event e;
      uint64_t t;
      e.polarity = event_array_msgs::mono::decode_t_x_y_p(p, timeBase, &t, &e.x, &e.y);
      e.t = shorten_time(t);
      if (e.y % threads_.size() == id) {
        const size_t offset = e.y * width_ + e.x;
        State & s = state_[offset];
        Event e_f;  // filtered event, from the past
        if (filterNoise(&s, e, &e_f)) {
          updateState(&s, e_f);
        }
      }
    }
    b.reader_done();
  }
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
    s.t = t;
    s.t_flip = t;
    s.x[0] = 0;
    s.x[1] = 0;
    s.dt_avg = -1;
    s.p_skip_idx = PackedVar();
    s.p_skip_idx.resetBadDataCount();
    // initialize lagged state
    for (size_t j = 0; j < 4; j++) {
      s.tp[j].set(t, false);
    }
#ifdef DEBUG
    if (i % width == debugX_ && i / width == debugY_) {
      std::cout << "initializing: " << s.t << " " << s.t_flip << std::endl;
    }
#endif
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
  // raw change in polarity, will be 0 or +-1
  const float dp = static_cast<int8_t>(e.polarity) - static_cast<int8_t>(s.p_skip_idx.p());
  // run the filter (see paper)
  const auto x_k = c_[0] * s.x[0] + c_[1] * s.x[1] + c_p_ * dp;

  if (x_k > 0 && s.x[0] <= 0) {
    // measure period upon transition from lower to upper half, i.e.
    // when ON events happen. This is more precise than on the other flank
    const float dt = (e.t - s.t_flip) * 1e-6;
    //
    if (s.dt_avg < 0) {  // initialization phase
      // restart the averaging process
      s.dt_avg = std::max(std::min(dt, dtMax_), dtMin_);
      s.p_skip_idx.resetBadDataCount();
#ifdef DEBUG
      if (e.x == debugX_ && e.y == debugY_) {
        std::cout << "starting avg dt: " << dt << " init avg: " << s.dt_avg << std::endl;
      }
#endif
    } else {
      // not restarting
      if (abs(dt - s.dt_avg) > s.dt_avg * resetThreshold_) {
        // too far away from avg, ignore it
        if (s.p_skip_idx.badDataCountLimitReached() || dt * resetThreshold_ > dtMin_) {
          // gotten too many bad dts or this pixel has not seen an update
          // in a very long time. Reset the average
          s.dt_avg = -1.0;  // signal that on next step dt can be computed
        } else {
          s.p_skip_idx.incBadDataCount();
        }
      } else {
        // all well, compound into average
        s.dt_avg = s.dt_avg * dtDecay_ + dtMix_ * dt;
        s.p_skip_idx.resetBadDataCount();
      }
    }
    s.t_flip = e.t;
#ifdef DEBUG
    if (e.x == debugX_ && e.y == debugY_) {
      debug_flip << std::setprecision(10) << e.t << " " << dt << " " << s.dt_avg << std::endl;
    }
#endif
    s.t_flip = e.t;
  }
  s.t = e.t;
  s.p_skip_idx.setPolarity(e.polarity);
  s.x[1] = s.x[0];
  s.x[0] = x_k;
#ifdef DEBUG
  if (e.x == debugX_ && e.y == debugY_) {
    const double dt = (e.t - s.t_flip) * 1e-6;
    const double f = s.dt_avg < 1e-6f ? 0 : (1.0 / s.dt_avg);
    debug << e.t << " " << x_k << " " << dt << " " << s.dt_avg << " " << f << " " << dp << " "
          << (int)s.p_skip_idx.getBadDataCount() << std::endl;
  }
#endif
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

cv::Mat FrequencyCam::makeFrequencyAndEventImage(cv::Mat * eventImage)
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

void FrequencyCam::publishImage()
{
  if (imagePub_.getNumSubscribers() != 0 && height_ != 0) {
    cv::Mat eventImg;
    cv::Mat rawImg = makeFrequencyAndEventImage(&eventImg);
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

uint32_t FrequencyCam::updateSingleThreaded(uint64_t timeBase, const std::vector<uint8_t> & events)
{
  uint32_t lastEventTime(0);
  const uint8_t * p_base = &events[0];

  for (const uint8_t * p = p_base; p < p_base + events.size();
       p += event_array_msgs::mono::bytes_per_event) {
    Event e;
    uint64_t t;
    e.polarity = event_array_msgs::mono::decode_t_x_y_p(p, timeBase, &t, &e.x, &e.y);
    e.t = shorten_time(t);
    lastEventTime = e.t;
    const size_t offset = e.y * width_ + e.x;
    State & s = state_[offset];
    Event e_f(e);  // filtered event, from the past
    if (filterNoise(&s, e, &e_f)) {
      updateState(&s, e_f);
    }
  }
  return (lastEventTime);
}

uint32_t FrequencyCam::updateMultiThreaded(uint64_t timeBase, const std::vector<uint8_t> & events)
{
  const uint8_t * p_base = &events[0];
  const uint8_t * p_last_event = p_base + events.size() - event_array_msgs::mono::bytes_per_event;
  uint32_t lastEventTime = shorten_time(event_array_msgs::mono::decode_t(p_last_event, timeBase));
  // tell consumers to start working
  for (auto & b : eventBuffer_) {
    b.set_events(&events, timeBase);
    b.writer_done();
  }
  // wait for consumers to finish
  for (auto & b : eventBuffer_) {
    b.wait_until_read_complete();
  }
  return (lastEventTime);
}

void FrequencyCam::callbackEvents(EventArrayConstPtr msg)
{
  const auto t_start = std::chrono::high_resolution_clock::now();
  const auto time_base =
    useSensorTime_ ? msg->time_base : rclcpp::Time(msg->header.stamp).nanoseconds();
  lastTime_ = rclcpp::Time(msg->header.stamp);

  if (state_ == 0 && !msg->events.empty()) {
    // first event ever, need to allocate state
    const uint8_t * p = &msg->events[0];
    uint64_t t;
    uint16_t x, y;
    (void)event_array_msgs::mono::decode_t_x_y_p(p, time_base, &t, &x, &y);
    initializeState(msg->width, msg->height, shorten_time(t) - 1 /* - 1usec */);
    header_ = msg->header;  // copy frame id
    lastSeq_ = msg->seq - 1;
  }
  lastEventTime_ = (!threads_.empty()) ? updateMultiThreaded(time_base, msg->events)
                                       : updateSingleThreaded(time_base, msg->events);
  eventCount_ += msg->events.size() / event_array_msgs::mono::bytes_per_event;
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
