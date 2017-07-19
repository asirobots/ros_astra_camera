#ifndef DELETE_THIS_FAKE_CV_BRIDGE_HPP_
#define DELETE_THIS_FAKE_CV_BRIDGE_HPP_

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <boost/endian/conversion.hpp>
#include <boost/regex.hpp>

#include <stdexcept>


inline static int depthStrToInt(const std::string depth) {
  if (depth == "8U") {
    return 0;
  } else if (depth == "8S") {
    return 1;
  } else if (depth == "16U") {
    return 2;
  } else if (depth == "16S") {
    return 3;
  } else if (depth == "32S") {
    return 4;
  } else if (depth == "32F") {
    return 5;
  }
  return 6;
}

inline int getCvType(const std::string& encoding)
{
  // Check for the most common encodings first
  if (encoding == sensor_msgs::image_encodings::BGR8)   return CV_8UC3;
  if (encoding == sensor_msgs::image_encodings::MONO8)  return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::RGB8)   return CV_8UC3;
  if (encoding == sensor_msgs::image_encodings::MONO16) return CV_16UC1;
  if (encoding == sensor_msgs::image_encodings::BGR16)  return CV_16UC3;
  if (encoding == sensor_msgs::image_encodings::RGB16)  return CV_16UC3;
  if (encoding == sensor_msgs::image_encodings::BGRA8)  return CV_8UC4;
  if (encoding == sensor_msgs::image_encodings::RGBA8)  return CV_8UC4;
  if (encoding == sensor_msgs::image_encodings::BGRA16) return CV_16UC4;
  if (encoding == sensor_msgs::image_encodings::RGBA16) return CV_16UC4;

  // For bayer, return one-channel
  if (encoding == sensor_msgs::image_encodings::BAYER_RGGB8) return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_BGGR8) return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_GBRG8) return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_GRBG8) return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_RGGB16) return CV_16UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_BGGR16) return CV_16UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_GBRG16) return CV_16UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_GRBG16) return CV_16UC1;

  // Miscellaneous
  if (encoding == sensor_msgs::image_encodings::YUV422) return CV_8UC2;

  // Check all the generic content encodings
  boost::cmatch m;

  if (boost::regex_match(encoding.c_str(), m,
        boost::regex("(8U|8S|16U|16S|32S|32F|64F)C([0-9]+)"))) {
    return CV_MAKETYPE(depthStrToInt(m[1].str()), atoi(m[2].str().c_str()));
  }

  if (boost::regex_match(encoding.c_str(), m,
        boost::regex("(8U|8S|16U|16S|32S|32F|64F)"))) {
    return CV_MAKETYPE(depthStrToInt(m[1].str()), 1);
  }

  throw std::runtime_error("Unrecognized image encoding [" + encoding + "]");
}


// Converts a ROS Image to a cv::Mat by sharing the data or chaning its endianness if needed
inline cv::Mat matFromImage(const sensor_msgs::msg::Image& source)
{
  int source_type = getCvType(source.encoding);
  int byte_depth = sensor_msgs::image_encodings::bitDepth(source.encoding) / 8;
  int num_channels = sensor_msgs::image_encodings::numChannels(source.encoding);

  if (source.step < source.width * byte_depth * num_channels)
  {
    std::stringstream ss;
    ss << "Image is wrongly formed: step < width * byte_depth * num_channels  or  " << source.step << " != " <<
        source.width << " * " << byte_depth << " * " << num_channels;
    throw std::runtime_error(ss.str());
  }

  if (source.height * source.step != source.data.size())
  {
    std::stringstream ss;
    ss << "Image is wrongly formed: height * step != size  or  " << source.height << " * " <<
              source.step << " != " << source.data.size();
    throw std::runtime_error(ss.str());
  }

  // If the endianness is the same as locally, share the data
  cv::Mat mat(source.height, source.width, source_type, const_cast<uchar*>(&source.data[0]), source.step);
  if ((boost::endian::order::native == boost::endian::order::big && source.is_bigendian) ||
      (boost::endian::order::native == boost::endian::order::little && !source.is_bigendian) ||
      byte_depth == 1)
    return mat;

  // Otherwise, reinterpret the data as bytes and switch the channels accordingly
  mat = cv::Mat(source.height, source.width, CV_MAKETYPE(CV_8U, num_channels*byte_depth),
                const_cast<uchar*>(&source.data[0]), source.step);
  cv::Mat mat_swap(source.height, source.width, mat.type());

  std::vector<int> fromTo;
  fromTo.reserve(num_channels*byte_depth);
  for(int i = 0; i < num_channels; ++i)
    for(int j = 0; j < byte_depth; ++j)
    {
      fromTo.push_back(byte_depth*i + j);
      fromTo.push_back(byte_depth*i + byte_depth - 1 - j);
    }
  cv::mixChannels(std::vector<cv::Mat>(1, mat), std::vector<cv::Mat>(1, mat_swap), fromTo);

  // Interpret mat_swap back as the proper type
  mat_swap = cv::Mat(source.height, source.width, source_type, mat_swap.data, mat_swap.step);

  return mat_swap;
}

inline sensor_msgs::msg::Image::SharedPtr toImageMsg(const cv::Mat& image, std_msgs::msg::Header header, const std::string& encoding)
{
  auto ros_image = std::make_shared<sensor_msgs::msg::Image>();
  ros_image->header = header;
  ros_image->height = image.rows;
  ros_image->width = image.cols;
  ros_image->encoding = encoding;
  ros_image->is_bigendian = (boost::endian::order::native == boost::endian::order::big);
  ros_image->step = image.cols * image.elemSize();
  size_t size = ros_image->step * image.rows;
  ros_image->data.resize(size);

  if (image.isContinuous())
  {
    memcpy((char*)(&ros_image->data[0]), image.data, size);
  }
  else
  {
    // Copy by row by row
    uchar* ros_data_ptr = (uchar*)(&ros_image->data[0]);
    uchar* cv_data_ptr = image.data;
    for (int i = 0; i < image.rows; ++i)
    {
      memcpy(ros_data_ptr, cv_data_ptr, ros_image->step);
      ros_data_ptr += ros_image->step;
      cv_data_ptr += image.step;
    }
  }

  return ros_image;
}

#endif // DELETE_THIS_FAKE_CV_BRIDGE_HPP_