#include "dataset_creator/dataset_creator.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace dataset_creator {

Creator::Creator(ros::NodeHandle &nodeHandle)
    : nodeHandle_(nodeHandle), it_(nodeHandle), initialized_dataset(false),
      tf_listener(new tf::TransformListener) {
  if (!readParameters()) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
  }

  if (!initOutputFolder()) {
    ROS_ERROR("Could not init output folder");
    ros::requestShutdown();
  }

  image_sub.subscribe(nodeHandle_, camera_image_topic, 10);
  cam_info_sub.subscribe(nodeHandle_, camera_info_topic, 10);
  pub = it_.advertise(output_topic, 1);

  if (!ignore_pointcloud) {
    // time synchronization for image, cloud and camera info
    cloud_sub.subscribe(nodeHandle_, cloud_topic, 10);
    sync.reset(
        new Sync(ApproxSyncPolicy(40), cloud_sub, image_sub, cam_info_sub));
    sync->registerCallback(boost::bind(&Creator::callback, this, _1, _2, _3));

  } else {
    // Only extracts images, ignores pointcloud
    no_pc_sync.reset(
        new SyncNoPC(ApproxSyncPolicyNoPC(40), image_sub, cam_info_sub));
    no_pc_sync->registerCallback(
        boost::bind(&Creator::no_pc_callback, this, _1, _2));
  }

  ROS_INFO("Successfully launched node.");
}

Creator::~Creator() {}

bool Creator::readParameters() {
  if (!nodeHandle_.getParam("cloudTopic", cloud_topic))
    return false;
  if (!nodeHandle_.getParam("cloudFrame", cloud_frame))
    return false;
  if (!nodeHandle_.getParam("cameraFrame", camera_frame))
    return false;
  if (!nodeHandle_.getParam("cameraInfoTopic", camera_info_topic))
    return false;
  if (!nodeHandle_.getParam("cameraImageTopic", camera_image_topic))
    return false;
  if (!nodeHandle_.getParam("outputTopic", output_topic))
    return false;
  if (!nodeHandle_.getParam("outputFolder", output_folder))
    return false;
  if (!nodeHandle_.getParam("maxDistance", max_distance))
    return false;
  if (!nodeHandle_.getParam("overrideOutput", overrideFolder))
    return false;
  if (!nodeHandle_.getParam("lidarMaxDistance", lidar_max_distance))
    return false;
  if (!nodeHandle_.getParam("ignorePointcloud", ignore_pointcloud))
    return false;
  if (!nodeHandle_.getParam("fileType", file_type))
    return false;

  std::string camera_name;
  if (nodeHandle_.getParam("use_camera_stick", camera_name) &&
      !camera_name.empty()) {
    std::cout << "camera stick is set. Going to load default values for topics "
              << camera_name << std::endl;
    std::cout << "outputFolder  =  " << output_folder << std::endl;
    camera_image_topic = "/camera_stick/" + camera_name + "/image";
    camera_info_topic = "/camera_stick/" + camera_name + "/camera_info";
    camera_frame = "/" + camera_name;
    output_topic = "/" + camera_name + "_labels";
  }

  return true;
}

bool Creator::initOutputFolder() {
  if (!boost::filesystem::exists(output_folder)) {
    // Folder does not exist
    if (!boost::filesystem::create_directory(output_folder)) {
      if (!overrideFolder) {
        std::cerr << "could not create folder: " << output_folder << std::endl;
        return false;
      }
      std::cout << "Folder already exists. Results will be overwritten"
                << std::endl;
    }
  } else if (!overrideFolder) {
    std::cout << "Folder already exists and override flag is not set!"
              << std::endl;
    return false;
  }

  // change outputfolder from "out_folder/" -> "out_folder/cam0/"
  output_folder.erase(output_folder.size() - 1);
  output_folder += camera_frame + "/";
  if (!boost::filesystem::create_directory(output_folder)) {
    if (!overrideFolder) {
      std::cerr << "could not create folder: " << output_folder << std::endl;
      return false;
    }
    std::cout << "Camera folder already exists. Results will be overwritten "
              << output_folder << std::endl;
  }

  std::cout << "Initialized dataset folder";
  return true;
}

// Write Camera Information
void Creator::createInfoFile(std::string timestamp,
                             image_geometry::PinholeCameraModel camera) {
  if (!initialized_dataset) {
    initialized_dataset = true;
    std::ofstream myfile(output_folder + "camera_info.txt");
    myfile << camera.cameraInfo();
    myfile.close();
  }
}

// Only extracts image and pose and not pointcloud information
void Creator::no_pc_callback(const sensor_msgs::ImageConstPtr &image,
                             const sensor_msgs::CameraInfoConstPtr &c_info) {

  // Extract timestamp from header
  std_msgs::Header h = image->header;
  std::string timestamp = std::to_string(h.stamp.toSec());

  boost::filesystem::create_directory((output_folder + timestamp));

  // Wait for transform for map
  std::shared_ptr<tf::StampedTransform> map_transform(new tf::StampedTransform);
  tf_listener->waitForTransform("/map", camera_frame, ros::Time(0),
                                ros::Duration(3.0));
  tf_listener->lookupTransform("/map", camera_frame, ros::Time(0),
                               *map_transform);

  // Get image from camera
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(c_info);

  // Create information file containing pinhole camera parameters
  createInfoFile(timestamp, model);

  cv_bridge::CvImageConstPtr img =
      cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
  // Camera image
  const cv::Mat &camera_image = img->image;
  // Store original image
  cv::imwrite(output_folder + "/" + timestamp + "/original.png", camera_image);

  // Export pose of camera in map frame
  tf::Matrix3x3 rotation = (*map_transform).getBasis();
  tf::Vector3 origin = (*map_transform).getOrigin();

  // Convert pose to string to save
  std::string matrixAsString;
  for (int i = 0; i < 3; i++) {
    matrixAsString += std::to_string(rotation.getRow(i).getX()) + "," +
                      std::to_string(rotation.getRow(i).getY()) + "," +
                      std::to_string(rotation.getRow(i).getZ()) + "," +
                      std::to_string(origin[i]) + ";\n";
  }
  matrixAsString += "0,0,0,1;";
  // Save pose matrix
  std::ofstream myfile(output_folder + timestamp + "pose.txt");
  myfile << matrixAsString;
  myfile.close();

  // Publish to topic
  pub.publish(img->toImageMsg());
  std::cout << "published" << output_folder + timestamp + "_preview.png"
            << std::endl;
}

// Extracts images, pose and pointcloud information (Distance to mesh, Absolute
// distanc)
void Creator::callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
                       const sensor_msgs::ImageConstPtr &image,
                       const sensor_msgs::CameraInfoConstPtr &c_info) {

  // Extract timestamp from header
  std_msgs::Header h = image->header;
  std::string timestamp = std::to_string(h.stamp.toSec());

  boost::filesystem::create_directory((output_folder + timestamp));

  // Transform point cloud msg to pcl pointcloud
  pcl::PointCloud<pcl::PointXYZI> in_cloud;
  pcl::fromROSMsg(*cloud, in_cloud);

  // Wait for transform for pointcloud
  std::shared_ptr<tf::StampedTransform> t(new tf::StampedTransform);
  tf_listener->waitForTransform(camera_frame, cloud_frame, ros::Time(0),
                                ros::Duration(3.0));
  tf_listener->lookupTransform(camera_frame, cloud_frame, ros::Time(0), *t);

  // Wait for transform for map
  std::shared_ptr<tf::StampedTransform> map_transform(new tf::StampedTransform);
  tf_listener->waitForTransform("/map", camera_frame, ros::Time(0),
                                ros::Duration(3.0));
  tf_listener->lookupTransform("/map", camera_frame, ros::Time(0),
                               *map_transform);

  // Get image from camera
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(c_info);

  // Create information file containing pinhole camera parameters
  createInfoFile(timestamp, model);

  cv_bridge::CvImageConstPtr img =
      cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
  // Camera image
  const cv::Mat &camera_image = img->image;
  // Store original image
  cv::imwrite(output_folder + "/" + timestamp + "/original.png", camera_image);

  // Image that contains the projected pointcloud
  cv::Mat preview_img = camera_image.clone();

  // Image that contains only the pointcloud projection
  cv::Mat labels_img(camera_image.rows, camera_image.cols, CV_8UC1,
                     cv::Scalar(0, 0, 0));

  // Image that contains distance information for each point of the pointcloud
  cv::Mat distance_img(camera_image.rows, camera_image.cols, CV_8UC1,
                       cv::Scalar(0, 0, 0));

  pcl::PointCloud<pcl::PointXYZI> camera_frame_pc;
  for (pcl::PointCloud<pcl::PointXYZI>::iterator it = in_cloud.begin();
       it != in_cloud.end(); ++it) {

    tf::Vector3 cloud_point(it->x, it->y, it->z);
    tf::Vector3 camera_local_tf = *t * cloud_point;

    pcl::PointXYZI camera_frame_point;
    camera_frame_point.x = camera_local_tf.getX();
    camera_frame_point.y = camera_local_tf.getY();
    camera_frame_point.z = camera_local_tf.getZ();
    camera_frame_point.intensity = it->intensity;
    camera_frame_pc.push_back(camera_frame_point);

    if (camera_local_tf.z() <= 0) {
      // Point not visible for camera
      continue;
    }

    cv::Point3d camera_local_cv(camera_local_tf.x(), camera_local_tf.y(),
                                camera_local_tf.z());
    cv::Point2d pixel = model.project3dToPixel(camera_local_cv);
    if (pixel.x >= 0 && pixel.x < camera_image.cols && pixel.y >= 0 &&
        pixel.y < camera_image.rows) {

      float distance = camera_local_tf.z();
      // Cap lidar distance at max distance value. This is needed to fit value
      // into [0,255]
      if (distance > lidar_max_distance) {
        distance = lidar_max_distance;
      }
      int distance_value = int(distance / (lidar_max_distance)*254) + 1;
      // Convert distance value to RGB color. RGB value is between 1 and 255. 0
      // Is excluded as this should be interpreted as "no measurement"

      int distance_to_mesh = std::max(
          0, std::min(254, (int)(254 * (it->intensity / max_distance))));

      // Draw an camera image -> will be published to topic
      cv::circle(camera_image, pixel, 6,
                 CV_RGB(distance_to_mesh, 255 - distance_to_mesh, 0), CV_FILLED,
                 8, 0);
      // Draw on preview image -> will be stored for visual inspection
      cv::circle(preview_img, pixel, 1,
                 CV_RGB(distance_to_mesh, 255 - distance_to_mesh, 0), CV_FILLED,
                 8, 0);
      // Draw on labels groundtruth image
      cv::circle(labels_img, pixel, 1, cv::Scalar(distance_to_mesh));
      // Draw on distance groundtruth image.
      cv::circle(distance_img, pixel, 1, cv::Scalar(distance_value));
    }
  }

  // Export pose of camera in map frame
  tf::Matrix3x3 rotation = (*map_transform).getBasis();
  tf::Vector3 origin = (*map_transform).getOrigin();

  // Convert pose to string to save
  std::string matrixAsString;
  for (int i = 0; i < 3; i++) {
    matrixAsString += std::to_string(rotation.getRow(i).getX()) + "," +
                      std::to_string(rotation.getRow(i).getY()) + "," +
                      std::to_string(rotation.getRow(i).getZ()) + "," +
                      std::to_string(origin[i]) + ";\n";
  }
  matrixAsString += "0,0,0,1;";
  // Save pose matrix
  std::ofstream myfile(output_folder + timestamp + "pose.txt");
  myfile << matrixAsString;
  myfile.close();

  // Save images
  cv::imwrite(output_folder + "/" + timestamp + "/preview.png", preview_img);
  cv::imwrite(output_folder + timestamp + "/labels.png", labels_img);
  cv::imwrite(output_folder + timestamp + "/distance.png", distance_img);
  // Save pointcloud
  pcl::io::savePCDFile(output_folder + timestamp + "/pcl.pcd", camera_frame_pc);
  // Publish to topic
  pub.publish(img->toImageMsg());
  std::cout << "published" << output_folder + timestamp + "_preview."
            << file_type << std::endl;
}

} // namespace dataset_creator
