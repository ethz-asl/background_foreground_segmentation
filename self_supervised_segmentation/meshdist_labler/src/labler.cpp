#include "meshdist_labler/labler.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.h>
namespace meshdist {

Labler::Labler(ros::NodeHandle &nodeHandle) : nodeHandle_(nodeHandle), it_(nodeHandle), tf_listener(new tf::TransformListener) {
  if (!readParameters()) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
  }

  ROS_INFO("Successfully launched node.");


    // time synchronization for image, cloud and camera info

    cloud_sub.subscribe(nodeHandle_, cloud_topic, 10);
    image_sub.subscribe(nodeHandle_, camera_image_topic, 10);
    cam_info_sub.subscribe( nodeHandle_, camera_info_topic, 10);

    typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::CameraInfo> ApproxSyncPolicy;



    std::cout << cloud_topic << " - " << camera_image_topic << " - " << camera_info_topic << std::endl;
    sync.reset(new Sync(ApproxSyncPolicy(40), cloud_sub, image_sub, cam_info_sub));

    std::cout << "registered cb" << std::endl;

    //publisher = it_.advertise(output_topic, 1);

    pub = it_.advertise(output_topic, 1);
    sync->registerCallback(boost::bind(&Labler::callback, this,  _1, _2, _3));
  //  pub2_ = it_.advertise("cam1_image_out", 1);
  //  pub3_ = it_.advertise("cam2_image_out", 1);
  //
  //  sub1_ = it_.subscribeCamera("/camera_stick/cam0/image", 1,
  //                              &Labler::imageCbCam0, this);
  //  sub2_ = it_.subscribeCamera("/camera_stick/cam1/image", 1,
  //                              &Labler::imageCbCam1, this);
  //  sub3_ = it_.subscribeCamera("/camera_stick/cam2/image", 1,
  //                              &Labler::imageCbCam2, this);
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
  return true;
}
//
// void Labler::gotCloud(const sensor_msgs::PointCloud2 &cloud_msg_in) {
//  std::cout << "got cloud" << std::endl;
//  lastCloud = cloud_msg_in;
//}
//
// void Labler::imageCbCam0(const sensor_msgs::ImageConstPtr &image_msg,
//                         const sensor_msgs::CameraInfoConstPtr &info_msg) {
//  Labler::imageCb(image_msg, info_msg, "cam0", Labler::pub1_);
//}
//
// void Labler::imageCbCam1(const sensor_msgs::ImageConstPtr &image_msg,
//                         const sensor_msgs::CameraInfoConstPtr &info_msg) {
//  Labler::imageCb(image_msg, info_msg, "cam1", Labler::pub2_);
//}
//
// void Labler::imageCbCam2(const sensor_msgs::ImageConstPtr &image_msg,
//                         const sensor_msgs::CameraInfoConstPtr &info_msg) {
//  Labler::imageCb(image_msg, info_msg, "cam2", Labler::pub3_);
//}
//
// void Labler::imageCb(const sensor_msgs::ImageConstPtr &image_msg,
//                     const sensor_msgs::CameraInfoConstPtr &info_msg,
//                     std::string camera_frame,
//                     const image_transport::Publisher &pub) {
//  std::cout << "got image!" << std::endl;
//
//  cv::Mat source_img;
//  cv::Mat preview_img;
//  cv_bridge::CvImagePtr input_bridge;
//
//  try {
//    input_bridge =
//        cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
//    source_img = input_bridge->image;
//    preview_img = source_img.clone();
//
//  } catch (cv_bridge::Exception &ex) {
//    ROS_ERROR("[meshdist_labler] Failed to convert image");
//    return;
//  }
//
//  // Empty greyscale iamge
//  cv::Mat labels_img(source_img.rows, source_img.cols, CV_8UC1,
//                     cv::Scalar(0, 0, 0));
//  std::cout << "got camera image to draw on" << std::endl;
//  cam_model_.fromCameraInfo(info_msg);
//
//  // Frame of the pointcloud
//  std::string frame_id = "map";
//  sensor_msgs::PointCloud2 out;
//  // Transform pointcloud to camera frame
//  pcl_ros::transformPointCloud(camera_frame, lastCloud, out, tf_listener_);
//
//  // Convert sensor_msgs pointcloud to pcl pointcloud
//  pcl::PCLPointCloud2 pcl_pc2;
//  pcl_conversions::toPCL(out, pcl_pc2);
//  pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
//      new pcl::PointCloud<pcl::PointXYZI>);
//  pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);
//
//  // Project points into camera frame
//  for (auto point : *temp_cloud) {
//    cv::Point3d pt_cv(point.x, point.y, point.z);
//    cv::Point2d uv;
//    uv = cam_model_.project3dToPixel(pt_cv);
//    int val = std::min(255, (int)(500.0 * point.intensity));
//    cv::circle(source_img, uv, 3, CV_RGB(254 - val, val, 0), -1);
//    cv::circle(preview_img, uv, 1, cv::Scalar(val));
//    cv::circle(labels_img, uv, 1, cv::Scalar(val));
//  }
//  std::cout << "Done drawing on image" << std::endl;
//  // Store images. Preview image are points on top of camera image,
//  // labels_img are only projeted points without real camera image
//  cv::imwrite("out1.png", preview_img);
//  cv::imwrite("out2.png", labels_img);
//
//  // RVIZ
//  pub.publish(input_bridge->toImageMsg());
//};

void Labler::callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
                      const sensor_msgs::ImageConstPtr &image,
                      const sensor_msgs::CameraInfoConstPtr &c_info) {
    std::cout << "in cb" << std::endl;
  pcl::PointCloud<pcl::PointXYZI> in_cloud;
    std::cout << "in cb" << std::endl;
  pcl::fromROSMsg(*cloud, in_cloud);
    std::cout << "in cb" << std::endl;

  std::shared_ptr<tf::StampedTransform> t(new tf::StampedTransform);
    std::cout << "in cb" << std::endl;
    tf_listener->waitForTransform(camera_frame, cloud_frame, ros::Time(0), ros::Duration(3.0));
    tf_listener->lookupTransform(camera_frame, cloud_frame, ros::Time(0), *t);

    //tf_listener->lookupTransform(camera_frame, cloud_frame, ros::Time(0), *t);
    std::cout << "in cb" << std::endl;
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(c_info);
    std::cout << "in cb" << std::endl;

  cv_bridge::CvImageConstPtr img =
      cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);

    std::cout << "in cb12" << std::endl;
  const cv::Mat &camera_image = img->image;

  int num_points_in_image = 0;
  bool rectify_warning = false;

  std_msgs::Header h = image->header;
  std::string timestamp =  std::to_string(h.stamp.toSec());

  cv::imwrite(timestamp+"_original.png", camera_image);
  cv::Mat preview_img = camera_image.clone();
  cv::Mat labels_img(camera_image.rows, camera_image.cols, CV_8UC1,
                     cv::Scalar(0, 0, 0));

    std::cout << "in 123123cb" << std::endl;
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
    /**
    try {
      pixel = model.unrectifyPoint(pixel);
    } catch (const image_geometry::Exception &e) {
        ROS_WARN("Could not unrectify image.");
    }*/

    if (pixel.x >= 0 && pixel.x < camera_image.cols && pixel.y >= 0 &&
        pixel.y < camera_image.rows) {
       // camera_image
       double val =  (254.0 - (1000.0 * it->intensity));
       int value = std::max((int)val, 0);
       std::cout << val << " - " << value << "int" << it->intensity << std::endl;

       cv::circle(camera_image,pixel, 6, CV_RGB(255 -value, value, 0), CV_FILLED, 8, 0);
        cv::circle(preview_img,pixel, 1, CV_RGB(255 -value, value, 0), CV_FILLED, 8, 0);
        cv::circle(labels_img, pixel, 1, cv::Scalar(value + 1));
     // cv::Vec3b color = camera_image.at<cv::Vec3b>(pixel.y, pixel.x);
      num_points_in_image += 1;
    }
  }
    std::cout << "done cb" << std::endl;

  std::cout << "points in image:" << num_points_in_image << std::endl;

  cv::imwrite(timestamp + "_preview.png", preview_img);
  cv::imwrite(timestamp + "labels.png", labels_img);

  pub.publish(img->toImageMsg());

}

} // namespace meshdist
