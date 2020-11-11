#pragma once

#include <boost/foreach.hpp>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv/cv.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pointmatcher/PointMatcher.h>
#include <pointmatcher_ros/point_cloud.h>
#include <pointmatcher_ros/transform.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/ColorRGBA.h>
#include <std_srvs/Empty.h>
#include <std_srvs/SetBool.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <unordered_set>

namespace meshdist {

typedef PointMatcher<float> PM;
/*
 * Main class for the node to handle the ROS interfacing.
 */
class Labler {
public:
  /*
   * Constructor.
   * @param nodeHandle the ROS node handle.
   */
  Labler(ros::NodeHandle &nodeHandle);

  /*
   * Destructor.
   */
  virtual ~Labler();

  /*
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  std::string cloud_topic;
  std::string cloud_frame;
  std::string camera_frame;
  std::string output_topic;
  std::string camera_info_topic;
  std::string camera_image_topic;
  std::string output_folder;
  float max_distance;

  tf::TransformListener *tf_listener;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::CameraInfo>
      ApproxSyncPolicy;
  typedef message_filters::Synchronizer<ApproxSyncPolicy> Sync;

  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub;
  message_filters::Subscriber<sensor_msgs::Image> image_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> cam_info_sub;

  boost::shared_ptr<Sync> sync;
  image_transport::Publisher pub;

  image_transport::ImageTransport it_;

  void callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
                const sensor_msgs::ImageConstPtr &image,
                const sensor_msgs::CameraInfoConstPtr &c_info);

  // ROS node handle.
  ros::NodeHandle &nodeHandle_;

  // ROS topic subscriber.
  ros::Subscriber pc_with_dist_sub_;

  // Topic of point cloud (loaded from params)
  std::string pc_topic_;
};

} // namespace meshdist
