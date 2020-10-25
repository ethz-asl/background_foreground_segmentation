#include "meshdist_labler/labler.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace meshdist {

Labler::Labler(ros::NodeHandle &nodeHandle)
    : nodeHandle_(nodeHandle), it_(nodeHandle) {
  if (!readParameters()) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
  }

  ROS_INFO("Successfully launched node.");

  pc_with_dist_sub_ =
      nodeHandle_.subscribe(pc_topic_, 1, &Labler::gotCloud, this);

  pub1_ = it_.advertise("cam0_image_out", 1);
  pub2_ = it_.advertise("cam1_image_out", 1);
  pub3_ = it_.advertise("cam2_image_out", 1);

  sub1_ = it_.subscribeCamera("/camera_stick/cam0/image", 1,
                              &Labler::imageCbCam0, this);
  sub2_ = it_.subscribeCamera("/camera_stick/cam1/image", 1,
                              &Labler::imageCbCam1, this);
  sub3_ = it_.subscribeCamera("/camera_stick/cam2/image", 1,
                              &Labler::imageCbCam2, this);
}

Labler::~Labler() {}

bool Labler::readParameters() {
  if (!nodeHandle_.getParam("pc_topic", pc_topic_))
    return false;
  return true;
}

void Labler::gotCloud(const sensor_msgs::PointCloud2 &cloud_msg_in) {
  std::cout << "got cloud" << std::endl;
  lastCloud = cloud_msg_in;
}

void Labler::imageCbCam0(const sensor_msgs::ImageConstPtr &image_msg,
                         const sensor_msgs::CameraInfoConstPtr &info_msg) {
  Labler::imageCb(image_msg, info_msg, "cam0", Labler::pub1_);
}

void Labler::imageCbCam1(const sensor_msgs::ImageConstPtr &image_msg,
                         const sensor_msgs::CameraInfoConstPtr &info_msg) {
  Labler::imageCb(image_msg, info_msg, "cam1", Labler::pub2_);
}

void Labler::imageCbCam2(const sensor_msgs::ImageConstPtr &image_msg,
                         const sensor_msgs::CameraInfoConstPtr &info_msg) {
  Labler::imageCb(image_msg, info_msg, "cam2", Labler::pub3_);
}

void Labler::imageCb(const sensor_msgs::ImageConstPtr &image_msg,
                     const sensor_msgs::CameraInfoConstPtr &info_msg,
                     std::string camera_frame,
                     const image_transport::Publisher &pub) {
  std::cout << "got image!" << std::endl;

  cv::Mat source_img;
  cv::Mat preview_img;
  cv_bridge::CvImagePtr input_bridge;

  try {
    input_bridge =
        cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    source_img = input_bridge->image;
    preview_img = source_img.clone();

  } catch (cv_bridge::Exception &ex) {
    ROS_ERROR("[meshdist_labler] Failed to convert image");
    return;
  }

  // Empty greyscale iamge
  cv::Mat labels_img(source_img.rows, source_img.cols, CV_8UC1,
                     cv::Scalar(0, 0, 0));
  std::cout << "got camera image to draw on" << std::endl;
  cam_model_.fromCameraInfo(info_msg);

  // Frame of the pointcloud
  std::string frame_id = "map";
  sensor_msgs::PointCloud2 out;
  // Transform pointcloud to camera frame
  pcl_ros::transformPointCloud(camera_frame, lastCloud, out, tf_listener_);

  // Convert sensor_msgs pointcloud to pcl pointcloud
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(out, pcl_pc2);
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);

  // Project points into camera frame
  for (auto point : *temp_cloud) {
    cv::Point3d pt_cv(point.x, point.y, point.z);
    cv::Point2d uv;
    uv = cam_model_.project3dToPixel(pt_cv);
    int val = std::min(255, (int)(500.0 * point.intensity));
    cv::circle(source_img, uv, 3, CV_RGB(254 - val, val, 0), -1);
    cv::circle(preview_img, uv, 1, cv::Scalar(val));
    cv::circle(labels_img, uv, 1, cv::Scalar(val));
  }
  std::cout << "Done drawing on image" << std::endl;
  // Store images. Preview image are points on top of camera image,
  // labels_img are only projeted points without real camera image
  cv::imwrite("out1.png", preview_img);
  cv::imwrite("out2.png", labels_img);

  // RVIZ
  pub.publish(input_bridge->toImageMsg());
};

} // namespace meshdist
