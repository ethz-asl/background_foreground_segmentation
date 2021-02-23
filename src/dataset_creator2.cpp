#include "dataset_creator/dataset_creator2.h"
#include "opencv2/highgui/highgui.hpp"
#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/ros.h>
#include <message_filters/synchronizer.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace dataset_creator {

Creator::Creator(ros::NodeHandle &nh, ros::NodeHandle &nh_private)
    : nodeHandle_(nh_private), it_(nh_private), initialized_dataset(false),
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
  cad_sub_ = nh.subscribe(
    nodeHandle_.param<std::string>("mesh_topic", "/mesh_publisher/mesh_out"),
    10,
    &Creator::gotCAD,
    this);
  pub = it_.advertise(output_topic, 1);
  labels_publisher = it_.advertise(labels_topic, 1);
  distance_publisher = it_.advertise(distance_topic, 1);

  // time synchronization for image, cloud and camera info
  cloud_sub.subscribe(nodeHandle_, cloud_topic, 10);
  sync.reset(
      new Sync(ApproxSyncPolicy(40), cloud_sub, image_sub, cam_info_sub));
  sync->registerCallback(boost::bind(&Creator::callback, this, _1, _2, _3));

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
  if (!nodeHandle_.getParam("storeImages", store_images))
    return false;
  if (!nodeHandle_.getParam("createPreview", create_preview))
    return false;
  if (!nodeHandle_.getParam("exportPose", export_pose))
    return false;
  if (!nodeHandle_.getParam("distanceTopic", distance_topic))
    return false;
  if (!nodeHandle_.getParam("labelsTopic", labels_topic))
    return false;

  return true;
}

void Creator::gotCAD(const cgal_msgs::TriangleMeshStamped &cad_mesh_in) {
  std::cout << "Dataset created received CAD mesh" << std::endl;
  map_frame_ = cad_mesh_in.header.frame_id;  // should be "marker2"
  cad_percept::cgal::msgToMeshModel(cad_mesh_in.mesh, &reference_mesh_);
}

// Extracts images, pose and pointcloud information (Distance to mesh,
// Absolute distanc)
void Creator::callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
                       const sensor_msgs::ImageConstPtr &image,
                       const sensor_msgs::CameraInfoConstPtr &c_info) {

  // Extract timestamp from header
  std_msgs::Header h = image->header;
  std::string timestamp = std::to_string(h.stamp.toSec());

  boost::filesystem::create_directory((output_folder + timestamp));

  // Wait for transform for map
  std::shared_ptr<tf::StampedTransform> map_transform(new tf::StampedTransform);
  tf_listener->waitForTransform("/map", camera_frame, h.stamp,
                                ros::Duration(0.5));
  try {
    tf_listener->lookupTransform("/map", camera_frame, h.stamp, *map_transform);
  } catch (const std::exception &e) {
    ROS_WARN("TF Failed %s", e.what());
    return;
  }

  // Get image from camera
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(c_info);

  if (store_images) {
    // Create information file containing pinhole camera parameters
    createInfoFile(timestamp, model);
  }

  cv_bridge::CvImageConstPtr img =
      cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
  // Camera image
  const cv::Mat &camera_image = img->image;

  if (store_images) {
    // Store original image
    cv::imwrite(output_folder + "/" + timestamp + "/original.png",
                camera_image);
  }

  if (cloud) {
    // Cloud is defined -> extract pc information
    projectPointCloud(timestamp, camera_image, cloud, model);
  }

  // Publish to topic
  pub.publish(img->toImageMsg());
  std::cout << "published" << output_folder + timestamp + "_preview."
            << file_type << std::endl;
}

void Creator::projectPointCloud(
    const std::string timestamp, const cv::Mat &camera_image,
    const sensor_msgs::PointCloud2ConstPtr &cloud,
    const image_geometry::PinholeCameraModel &model) {
  if (map_frame_ == "") {
    ROS_WARN("received cloud but not yet any mesh.");
    return;
  }
  // Transform point cloud msg to pcl pointcloud
  pcl::PointCloud<pcl::PointXYZ> in_cloud;
  pcl::fromROSMsg(*cloud, in_cloud);

  // Wait for transform for pointcloud
  std::shared_ptr<tf::StampedTransform> t(new tf::StampedTransform);
  std::shared_ptr<tf::StampedTransform> t_lidar_map(new tf::StampedTransform);
  tf_listener->waitForTransform(camera_frame, cloud_frame, cloud->header.stamp,
                                ros::Duration(0.4));
  tf_listener->lookupTransform(camera_frame, cloud_frame, cloud->header.stamp,
                               *t);
  tf_listener->lookupTransform(map_frame_, cloud_frame, cloud->header.stamp,
                               *t_lidar_map);
  cv::Mat preview_img;

  if (create_preview) {
    // Image that contains the projected pointcloud
    preview_img = camera_image.clone();
  }
  // Image that contains only the pointcloud projection
  cv::Mat labels_img(camera_image.rows, camera_image.cols, CV_8UC1,
                     cv::Scalar(0, 0, 0));

  // Image that contains distance information for each point of the pointcloud
  cv::Mat distance_img(camera_image.rows, camera_image.cols, CV_8UC1,
                       cv::Scalar(0, 0, 0));

  pcl::PointCloud<pcl::PointXYZI> camera_frame_pc;
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = in_cloud.begin();
       it != in_cloud.end(); ++it) {
    tf::Vector3 cloud_point(it->x, it->y, it->z);
    // project point into map frame and get distance
    tf::Vector3 point_in_map_frame = *t_lidar_map * cloud_point;
    float squared_distance = (float)sqrt(reference_mesh_->squaredDistance(
        cad_percept::cgal::Point(point_in_map_frame.x(),
                                 point_in_map_frame.y(),
                                 point_in_map_frame.z())));

    tf::Vector3 camera_local_tf = *t * cloud_point;

    pcl::PointXYZI camera_frame_point;
    camera_frame_point.x = camera_local_tf.getX();
    camera_frame_point.y = camera_local_tf.getY();
    camera_frame_point.z = camera_local_tf.getZ();
    camera_frame_point.intensity = squared_distance;
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
      // Convert distance value to RGB color. RGB value is between 1 and 255.
      // 0 Is excluded as this should be interpreted as "no measurement"

      int distance_to_mesh = std::max(
          0, std::min(254, (int)(254 * (squared_distance / max_distance))));

      if (create_preview) {
        // Draw on preview image -> will be stored for visual inspection
        cv::circle(preview_img, pixel, 1,
                   CV_RGB(distance_to_mesh, 255 - distance_to_mesh, 0),
                   CV_FILLED, 8, 0);
      }
      // Draw on labels groundtruth image
      cv::circle(labels_img, pixel, 1, cv::Scalar(distance_to_mesh));
      // Draw on distance groundtruth image.
      cv::circle(distance_img, pixel, 1, cv::Scalar(distance_value));
    }
  }

  if (store_images) {
    // Save images
    cv::imwrite(output_folder + "/" + timestamp + "/preview." + file_type,
                preview_img);
    cv::imwrite(output_folder + timestamp + "/labels.png", labels_img);
    cv::imwrite(output_folder + timestamp + "/distance.png", distance_img);
  }

  cv_bridge::CvImage out_msg;
  out_msg.header = cloud->header;
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
  out_msg.image = labels_img;
  labels_publisher.publish(out_msg.toImageMsg());

  out_msg.image = distance_img;
  distance_publisher.publish(out_msg.toImageMsg());
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
} // namespace dataset_creator
