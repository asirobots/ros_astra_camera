// Copyright (c) 2008, Willow Garage, Inc.
// All rights reserved.
//
// Software License Agreement (BSD License 2.0)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of the Willow Garage nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// This file is originally from:
// https://github.com/ros-perception/image_pipeline/blob/da750d1/depth_image_proc/include/depth_image_proc/depth_conversions.h  // NOLINT

#ifndef DEPTH_TO_POINTCLOUD__DEPTH_CONVERSIONS_HPP_
#define DEPTH_TO_POINTCLOUD__DEPTH_CONVERSIONS_HPP_

#include <astra_camera/depth_traits.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <limits>

namespace depth_to_pointcloud
{

// Handles float or uint16 depths
template<typename T>
void convert(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  const image_geometry::PinholeCameraModel & model,
  double range_max = 0.0)
{
  // Use correct principal point from calibration
  float center_x = model.cx();
  float center_y = model.cy();

  uint32_t num_valid = 0;

  // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
  double unit_scaling = DepthTraits<T>::toMeters(T(1) );
  float constant_x = unit_scaling / model.fx();
  float constant_y = unit_scaling / model.fy();
  float bad_point = std::numeric_limits<float>::quiet_NaN();

  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
  const T * depth_row = reinterpret_cast<const T *>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(T);
  for (int v = 0; v < static_cast<int>(depth_msg->height); ++v, depth_row += row_step) {
    for (int u = 0; u < static_cast<int>(depth_msg->width); ++u) {
      T depth = depth_row[u];

      if (!DepthTraits<T>::valid(depth))
      {
        continue;
      }

      auto depth_meters = DepthTraits<T>::toMeters(depth);

      if (depth_meters < range_max)
      {
        // Fill in XYZ
        *iter_x = (u - center_x) * depth * constant_x;
        *iter_y = (v - center_y) * depth * constant_y;
        *iter_z = depth_meters;

        ++iter_x;
        ++iter_y;
        ++iter_z;

        num_valid++;
      }
    }
  }

  cloud_msg->height = num_valid;
}

}  // namespace depth_to_pointcloud

#endif  // DEPTH_TO_POINTCLOUD__DEPTH_CONVERSIONS_HPP_