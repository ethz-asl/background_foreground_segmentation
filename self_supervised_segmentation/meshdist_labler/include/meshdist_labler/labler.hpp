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
#include <pointmatcher/PointMatcher.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/ColorRGBA.h>
#include <std_srvs/Empty.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <std_srvs/SetBool.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <unordered_set>

#include <pointmatcher_ros/point_cloud.h>
#include <pointmatcher_ros/transform.h>
namespace meshdist {

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;
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
    std::string camera_topic;
    std::string camera_frame;
    std::string output_topic;
    std::string camera_info_topic;
    std::string camera_image_topic;

    sensor_msgs::CameraInfoConstPtr camera_info;
    sensor_msgs::ImageConstPtr image;

    tf::TransformListener *tf_listener;

    typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::CameraInfo> ApproxSyncPolicy;
    typedef message_filters::Synchronizer<ApproxSyncPolicy> Sync;

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub;
    message_filters::Subscriber<sensor_msgs::Image> image_sub;
    message_filters::Subscriber<sensor_msgs::CameraInfo> cam_info_sub;

    boost::shared_ptr<Sync> sync;





    image_transport::Publisher pub;
    //image_transport::Publisher*publisher;

    image_transport::ImageTransport it_;

    void callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
                  const sensor_msgs::ImageConstPtr &image,
                  const sensor_msgs::CameraInfoConstPtr &c_info);
//
//  /**
//   *
//   * TODO
//   *
//   * */
//
//  void imageCbCam0(const sensor_msgs::ImageConstPtr &image_msg,
//                   const sensor_msgs::CameraInfoConstPtr &info_msg);
//
//  void imageCbCam1(const sensor_msgs::ImageConstPtr &image_msg,
//                   const sensor_msgs::CameraInfoConstPtr &info_msg);
//
//  void imageCbCam2(const sensor_msgs::ImageConstPtr &image_msg,
//                   const sensor_msgs::CameraInfoConstPtr &info_msg);
//
//  image_transport::CameraSubscriber sub1_;
//  image_transport::CameraSubscriber sub2_;
//  image_transport::CameraSubscriber sub3_;
//  tf::TransformListener tf_listener_;
//  image_geometry::PinholeCameraModel cam_model_;
//  std::vector<std::string> frame_ids_;
//  image_transport::ImageTransport it_;
//  image_transport::Publisher pub1_;
//  image_transport::Publisher pub2_;
//  image_transport::Publisher pub3_;
//
//  sensor_msgs::PointCloud2 lastCloud;
//  DP lastDP;
//
//  void imageCb(const sensor_msgs::ImageConstPtr &image_msg,
//               const sensor_msgs::CameraInfoConstPtr &info_msg,
//               std::string camera_frame, const image_transport::Publisher &pub);
//  /**
//   * Point cloud callback
//   */
//  void gotCloud(const sensor_msgs::PointCloud2 &cloud_msg_in);

  // ROS node handle.
  ros::NodeHandle &nodeHandle_;

  // ROS topic subscriber.
  ros::Subscriber pc_with_dist_sub_;

  // Topic of point cloud (loaded from params)
  std::string pc_topic_;
};

} // namespace meshdist
