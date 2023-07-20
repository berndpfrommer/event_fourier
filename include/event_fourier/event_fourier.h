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

#ifndef EVENT_FOURIER__EVENT_FOURIER_H_
#define EVENT_FOURIER__EVENT_FOURIER_H_

#include <event_camera_codecs/decoder_factory.h>
#include <event_camera_codecs/event_processor.h>
#include <stdlib.h>

#include <cstdlib>
#include <event_camera_msgs/msg/event_packet.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <opencv2/core/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace event_fourier
{
class EventFourier : public rclcpp::Node, event_camera_codecs::EventProcessor
{
public:
  typedef std::complex<double> complex_t;
  explicit EventFourier(const rclcpp::NodeOptions & options);
  ~EventFourier();
  // ------- inherited from EventProcessor
  void eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity);
  void eventExtTrigger(uint64_t, uint8_t, uint8_t) {}
  void finished() {}
  void rawData(const char *, size_t) {}
  // ------- end of inherited
  EventFourier(const EventFourier &) = delete;
  EventFourier & operator=(const EventFourier &) = delete;

private:
  using EventPacket = event_camera_msgs::msg::EventPacket;
  using EventPacketConstPtr = EventPacket::ConstSharedPtr;
  void readEventsFromBag(const std::string & bagName);
  bool initialize();
  void callbackEvents(EventPacketConstPtr msg);
  void resetState(uint32_t width, uint32_t height);
  void updateState(
    const uint8_t f_idx, const uint16_t x, const uint16_t y, uint64_t t, bool polarity);
  void publishImage();
  void statistics();
  inline double getRandomFreq() const
  {
    return (((double)std::rand() / RAND_MAX) * (freq_[1] - freq_[0]) + freq_[0]);
  }
  void copyState(uint32_t x, uint32_t y, uint8_t f_src, uint8_t f_dest);
  void copyState(
    uint32_t x_src, uint32_t y_src, uint8_t f_src, uint32_t x_dest, uint32_t y_dest,
    uint8_t f_dest);

  void initializeState(uint32_t x, uint32_t y, uint8_t f_idx, double f);
  void printFrequencies();

  // ------ variables ----
  rclcpp::Time lastTime_{0};
  uint64_t sliceTime_{0};
  bool useSensorTime_;
  image_transport::Publisher imagePub_;
  rclcpp::Subscription<EventPacket>::SharedPtr eventSub_;
  rclcpp::TimerBase::SharedPtr pubTimer_;
  rclcpp::TimerBase::SharedPtr statsTimer_;

  std::vector<double> freq_;
  std::vector<uint32_t> roi_;
  double windowSizeInCycles_{50.0};
  cv::Mat mask_;
  complex_t * state_{0};
  uint32_t width_;
  uint32_t height_;
  uint64_t eventCount_{0};
  uint64_t msgCount_{0};
  uint64_t lastCount_{0};
  uint64_t totTime_{0};
  uint64_t lastEventTime_;
  int64_t lastSeq_{0};
  int64_t droppedSeq_{0};
  std_msgs::msg::Header header_;
  event_camera_codecs::DecoderFactory<EventPacket, EventFourier> decoderFactory_;
};
}  // namespace event_fourier
#endif  // EVENT_FOURIER__EVENT_FOURIER_H_
