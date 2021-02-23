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
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/ColorRGBA.h>
#include <std_srvs/Empty.h>
#include <std_srvs/SetBool.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <unordered_set>
#include <cgal_definitions/cgal_typedefs.h>
#include <cgal_conversions/mesh_conversions.h>
#include <cgal_definitions/mesh_model.h>
#include <cgal_msgs/TriangleMeshStamped.h>
#include "cpt_selective_icp/utils.h"

namespace dataset_creator {

/*
 * Main class for the node to handle the ROS interfacing.
 */
class Creator {
public:
  /*
   * Constructor.
   * @param nodeHandle the ROS node handle.
   */
  Creator(ros::NodeHandle &nh, ros::NodeHandle &nh_private);

  /*
   * Destructor.
   */
  virtual ~Creator();

private:
  /*
   * Creates a file containing information about the dataset, camera geometry
   * etc.
   */
  void createInfoFile(std::string timestamp,
                      image_geometry::PinholeCameraModel camera);

  /*
   * Initializes the dataset output folder specified by the parameters
   */
  bool initOutputFolder();

  // to load the mesh model
  void gotCAD(const cgal_msgs::TriangleMeshStamped &cad_mesh_in);

  /*
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  // Projects a given point cloud into the camera frame and stores labels
  void projectPointCloud(const std::string timestamp,
                         const cv::Mat &camera_image,
                         const sensor_msgs::PointCloud2ConstPtr &cloud,
                         const image_geometry::PinholeCameraModel &model);

  /*
   * Callback to extract information from dataset
   * @param cloud Point Cloud containing distance information
   * @param image Image of Camera from
   * @param c_info Camera information
   */
  void callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
                const sensor_msgs::ImageConstPtr &image,
                const sensor_msgs::CameraInfoConstPtr &c_info);

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::CameraInfo>
      ApproxSyncPolicy;

  typedef message_filters::Synchronizer<ApproxSyncPolicy> Sync;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::CameraInfo>
      ApproxSyncPolicyNoPC;

  typedef message_filters::Synchronizer<ApproxSyncPolicyNoPC> SyncNoPC;

  // Should dataset folder be overwritten
  bool overrideFolder;
  // Flag whether dataset folder has been initialized
  bool initialized_dataset;
  // flag if only images should be extracted or not
  bool ignore_pointcloud;

  std::string cloud_topic;
  std::string cloud_frame;
  std::string camera_frame;
  std::string output_topic;
  std::string camera_info_topic;
  std::string camera_image_topic;
  std::string output_folder;
  std::string file_type;
  std::string distance_topic;
  std::string labels_topic;

  bool create_preview;
  // if images should be stored on disk
  bool store_images;
  bool export_pose;

  // Point cloud distance from mesh will be stored as image -> [0,255]
  // Distances to mesh will be mapped as follows distance -> min(max_distance,
  // point_distance)*255 Points that have a bigger distance to a mesh than
  // max_distance will be reduced to max_distance
  float max_distance;
  // Same for lidar point distance
  float lidar_max_distance;

  tf::TransformListener *tf_listener;

  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub;
  message_filters::Subscriber<sensor_msgs::Image> image_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> cam_info_sub;

  // Sync function to call if point cloud measurements are available.
  boost::shared_ptr<Sync> sync;
  // Sync function to call if pc information are not available.
  boost::shared_ptr<SyncNoPC> no_pc_sync;

  image_transport::Publisher pub;

  image_transport::Publisher labels_publisher;
  image_transport::Publisher distance_publisher;
  // Used to publish images
  image_transport::ImageTransport it_;

  // ROS node handle.
  ros::NodeHandle &nodeHandle_;

  // ROS topic subscriber.
  ros::Subscriber pc_with_dist_sub_;
  ros::Subscriber cad_sub_;

  // Topic of point cloud (loaded from params)
  std::string pc_topic_;

  // added stuff
  std::string map_frame_;
  cad_percept::cgal::MeshModel::Ptr reference_mesh_;
};

} // namespace dataset_creator
