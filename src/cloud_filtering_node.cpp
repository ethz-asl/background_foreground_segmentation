#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <iostream>
#include <iterator>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sstream>
#include <tf/tf.h>
#include <tf/transform_listener.h>

std::string cloud_frame;
std::string output_topic;
int density_threshold;

uint num_cameras;
std::vector<sensor_msgs::CameraInfoConstPtr> cam_infos;
std::vector<std::string> image_frames;

sensor_msgs::ImageConstPtr image;

tf::TransformListener *tf_listener;
ros::Publisher *publisher;


pcl::PointCloud<pcl::PointXYZI>
filter_pc(const pcl::PointCloud<pcl::PointXYZ> &cloud,
          const sensor_msgs::ImageConstPtr &image,
          const sensor_msgs::CameraInfoConstPtr &c_info,
          const std::string &image_frame) {
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  std::shared_ptr<tf::StampedTransform> t(new tf::StampedTransform);
  try {
  tf_listener->lookupTransform(image_frame, cloud_frame, ros::Time(0), *t);
  } catch (const std::exception &e) {
    ROS_WARN("TF Failed %s", e.what());
    return out_cloud;
  }

  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(c_info);

  cv_bridge::CvImageConstPtr img;
  try {
    img = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    ROS_WARN("CVBridge Failed %s", e.what());
    return out_cloud;
  }
  const cv::Mat &camera_image = img->image;

  int num_points_in_image = 0;
  bool rectify_warning = false;
  for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = cloud.begin();
       it != cloud.end(); ++it) {
    pcl::PointXYZI new_point;
    new_point.x = it->x;
    new_point.y = it->y;
    new_point.z = it->z;
    tf::Vector3 cloud_point(it->x, it->y, it->z);
    tf::Vector3 camera_local_tf = *t * cloud_point;
    if (camera_local_tf.z() <= 0)
      continue;
    cv::Point3d camera_local_cv(camera_local_tf.x(), camera_local_tf.y(),
                                camera_local_tf.z());
    cv::Point2d pixel = model.project3dToPixel(camera_local_cv);
    try {
      pixel = model.unrectifyPoint(pixel);
    } catch (const image_geometry::Exception &e) {
      if (!rectify_warning) {
        ROS_WARN("Could not unrectify image.");
        rectify_warning = true;
      }
    }
    if (pixel.x >= 0 && pixel.x < camera_image.cols && pixel.y >= 0 &&
        pixel.y < camera_image.rows) {
      uchar color = camera_image.at<uchar>(pixel.y, pixel.x);
      num_points_in_image += 1;
      if (color > density_threshold) {
        new_point.intensity = (float)std::max(color - density_threshold, 0);
        out_cloud.push_back(new_point);
      }
    }
  }
  ROS_INFO("Found %d points in image.", num_points_in_image);
  return out_cloud;
}

void _callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
               const std::vector<sensor_msgs::ImageConstPtr> &images) {
  pcl::PointCloud<pcl::PointXYZ> in_cloud;
  pcl::fromROSMsg(*cloud, in_cloud);
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  for (uint i = 0; i < num_cameras; i++) {
    // check if caminfo ready
    if (cam_infos[i] == nullptr) {
      ROS_INFO("Camera Calibration for Camera %d not ready yet", i);
    } else {
      if (image_frames.size() > 0) {
        out_cloud +=
            filter_pc(in_cloud, images[i], cam_infos[i], image_frames[i]);
      } else {
        std::string frame = images[i]->header.frame_id;
        out_cloud += filter_pc(in_cloud, images[i], cam_infos[i], frame);
      }
    }
  }

  sensor_msgs::PointCloud2Ptr out_cloud_msg =
      sensor_msgs::PointCloud2Ptr(new sensor_msgs::PointCloud2());
  pcl::toROSMsg(out_cloud, *out_cloud_msg);
  out_cloud_msg->header = cloud->header;
  publisher->publish(out_cloud_msg);
}

void callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
              const sensor_msgs::ImageConstPtr &image) {
  _callback(cloud, {image});
}

void callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
              const sensor_msgs::ImageConstPtr &image0,
              const sensor_msgs::ImageConstPtr &image1) {
  _callback(cloud, {image0, image1});
}

void callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
              const sensor_msgs::ImageConstPtr &image0,
              const sensor_msgs::ImageConstPtr &image1,
              const sensor_msgs::ImageConstPtr &image2) {
  _callback(cloud, {image0, image1, image2});
}

void callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
              const sensor_msgs::ImageConstPtr &image0,
              const sensor_msgs::ImageConstPtr &image1,
              const sensor_msgs::ImageConstPtr &image2,
              const sensor_msgs::ImageConstPtr &image3) {
  _callback(cloud, {image0, image1, image2, image3});
}

void registerCamInfo(const sensor_msgs::CameraInfoConstPtr &c_info,
                     const uint index) {
  if (cam_infos.size() != num_cameras) {
    // initialize
    cam_infos.erase(cam_infos.begin(), cam_infos.end());
    for (uint i = 0; i < num_cameras; i++)
      cam_infos.push_back(nullptr);
  }
  cam_infos[index] = c_info;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "cloud_filtering");
  ros::NodeHandle nh;
  std::string cloud_topic;
  std::vector<std::string> caminfo_topics;
  std::vector<std::string> image_topics;

  ros::NodeHandle("~").getParam("cloud_topic", cloud_topic);
  ros::NodeHandle("~").getParam("cloud_frame", cloud_frame);
  ros::NodeHandle("~").getParam("image_frames", image_frames);
  ros::NodeHandle("~").getParam("caminfo_topics", caminfo_topics);
  ros::NodeHandle("~").getParam("image_topics", image_topics);
  ros::NodeHandle("~").getParam("filtered_pc_topic", output_topic);
  ros::NodeHandle("~").getParam("density_threshold", density_threshold);

  std::ostringstream str;
  std::copy(image_topics.begin(), image_topics.end(),
            std::ostream_iterator<std::string>(str, ","));
  ROS_INFO("Image Topics %s", str.str().c_str());
  // time synchronization for image, cloud and camera info
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(
      nh, cloud_topic, 1);

  num_cameras = image_topics.size();
  std::vector<ros::Subscriber> subs;
  std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>>
      image_subs;
  for (uint i = 0; i < num_cameras; i++) {
    cam_infos.push_back(nullptr);
    subs.push_back(nh.subscribe<sensor_msgs::CameraInfo>(
        caminfo_topics[i], 10, boost::bind(&registerCamInfo, _1, i)));
    image_subs.push_back(
        std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(
            nh, image_topics[i], 1));
  }

  // Unfortunately, we cannot template or bind with dynamic number of members
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, sensor_msgs::Image>
      ApproxSyncPolicy1;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy1>> sync1;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Image>
      ApproxSyncPolicy2;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy2>> sync2;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Image,
      sensor_msgs::Image>
      ApproxSyncPolicy3;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy3>> sync3;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Image,
      sensor_msgs::Image, sensor_msgs::Image>
      ApproxSyncPolicy4;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy4>> sync4;
  if (num_cameras == 1) {
    sync1 = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy1>>(
        ApproxSyncPolicy1(10), cloud_sub, *image_subs[0]);
    sync1->registerCallback(boost::bind(&callback, _1, _2));
  } else if (num_cameras == 2) {
    sync2 = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy2>>(
        ApproxSyncPolicy2(10), cloud_sub, *image_subs[0], *image_subs[1]);
    sync2->registerCallback(boost::bind(&callback, _1, _2, _3));
  } else if (num_cameras == 3) {
    sync3 = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy3>>(
        ApproxSyncPolicy3(10), cloud_sub, *image_subs[0], *image_subs[1],
        *image_subs[2]);
    sync3->registerCallback(boost::bind(&callback, _1, _2, _3, _4));
  } else if (num_cameras == 4) {
    sync4 = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy4>>(
        ApproxSyncPolicy4(10), cloud_sub, *image_subs[0], *image_subs[1],
        *image_subs[2], *image_subs[3]);
    sync4->registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5));
  } else {
    ROS_WARN("Projection for more than 4 cameras not implemented yet!");
  }
  ROS_INFO("Registered callbacks for %d cameras.", num_cameras);
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 1);
  publisher = &pub;
  tf::TransformListener tl;
  tf_listener = &tl;
  ros::spin();
  return 0;
}
