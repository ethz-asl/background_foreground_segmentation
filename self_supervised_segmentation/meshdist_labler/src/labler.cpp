#include "meshdist_labler/labler.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/point_cloud2_iterator.h>
namespace meshdist {

Labler::Labler(ros::NodeHandle &nodeHandle)
    : nodeHandle_(nodeHandle), it_(nodeHandle),
      tf_listener(new tf::TransformListener) {
  if (!readParameters()) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
  }

  // time synchronization for image, cloud and camera info
  cloud_sub.subscribe(nodeHandle_, cloud_topic, 10);
  image_sub.subscribe(nodeHandle_, camera_image_topic, 10);
  cam_info_sub.subscribe(nodeHandle_, camera_info_topic, 10);

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::CameraInfo>
      ApproxSyncPolicy;

  sync.reset(
      new Sync(ApproxSyncPolicy(40), cloud_sub, image_sub, cam_info_sub));

  pub = it_.advertise(output_topic, 1);
  sync->registerCallback(boost::bind(&Labler::callback, this, _1, _2, _3));

  ROS_INFO("Successfully launched node.");
}

Labler::~Labler() {}

bool Labler::readParameters() {
  if (!nodeHandle_.getParam("pcTopic", pc_topic_))
    return false;
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
  return true;
}

void Labler::callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
                      const sensor_msgs::ImageConstPtr &image,
                      const sensor_msgs::CameraInfoConstPtr &c_info) {

  // Extract timestamp from header
  std_msgs::Header h = image->header;
  std::string timestamp = std::to_string(h.stamp.toSec());

  // Transform point cloud msg to pcl pointcloud
  pcl::PointCloud<pcl::PointXYZI> in_cloud;
  pcl::fromROSMsg(*cloud, in_cloud);

  // Wait for transform
  std::shared_ptr<tf::StampedTransform> t(new tf::StampedTransform);
  tf_listener->waitForTransform(camera_frame, cloud_frame, ros::Time(0),
                                ros::Duration(3.0));
  tf_listener->lookupTransform(camera_frame, cloud_frame, ros::Time(0), *t);

  // Get image from camera
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(c_info);

  cv_bridge::CvImageConstPtr img =
      cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
  const cv::Mat &camera_image = img->image;
  cv::imwrite(output_folder + timestamp + "_original.png", camera_image);
  // Image that contains the projected pointcloud
  cv::Mat preview_img = camera_image.clone();

  // Image that contains only the pointcloud projection
  cv::Mat labels_img(camera_image.rows, camera_image.cols, CV_8UC1,
                     cv::Scalar(0, 0, 0));

  for (pcl::PointCloud<pcl::PointXYZI>::iterator it = in_cloud.begin();
       it != in_cloud.end(); ++it) {

    tf::Vector3 cloud_point(it->x, it->y, it->z);
    tf::Vector3 camera_local_tf = *t * cloud_point;

    if (camera_local_tf.z() <= 0) {
      // Point not visible for camera
      continue;
    }

    cv::Point3d camera_local_cv(camera_local_tf.x(), camera_local_tf.y(),
                                camera_local_tf.z());
    cv::Point2d pixel = model.project3dToPixel(camera_local_cv);

    if (pixel.x >= 0 && pixel.x < camera_image.cols && pixel.y >= 0 &&
        pixel.y < camera_image.rows) {

      // Convert distance value to RGB color. RGB value is between 1 and 255
      // (max distance)
        int value = std::max(0, std::min(254, (int) (254 * (it->intensity/max_distance))));

        cv::circle(camera_image, pixel, 6, CV_RGB(value, 255 - value, 0),
                   CV_FILLED, 8, 0);
        cv::circle(preview_img, pixel, 1, CV_RGB(value, 255-value, 0),
                   CV_FILLED, 8, 0);
        cv::circle(labels_img, pixel, 1, cv::Scalar(value + 1));
    }
  }

  cv::imwrite(output_folder + timestamp + "_preview.png", preview_img);
  cv::imwrite(output_folder + timestamp + "_labels.png", labels_img);

  pub.publish(img->toImageMsg());
  std::cout << "published" << output_folder + timestamp + "_preview.png" << std::endl;
}

} // namespace meshdist
