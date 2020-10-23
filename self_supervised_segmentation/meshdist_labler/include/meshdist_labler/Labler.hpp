#pragma once

#include <boost/foreach.hpp>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
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

private:
  /*
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  /**
   *
   * TODO
   *
   * */
  image_transport::CameraSubscriber sub_;
  tf::TransformListener tf_listener_;
  image_geometry::PinholeCameraModel cam_model_;
  std::vector<std::string> frame_ids_;
  image_transport::ImageTransport it_;
  image_transport::Publisher pub_;

  void imageCb(const sensor_msgs::ImageConstPtr &image_msg,
               const sensor_msgs::CameraInfoConstPtr &info_msg);
  /**
   * Point cloud callback
   */
  void gotCloud(const sensor_msgs::PointCloud2 &cloud_msg_in);

  // ROS node handle.
  ros::NodeHandle &nodeHandle_;

  // ROS topic subscriber.
  ros::Subscriber pc_with_dist_sub_;

  // Topic of point cloud (loaded from params)
  std::string pc_topic_;
};

} // namespace meshdist
