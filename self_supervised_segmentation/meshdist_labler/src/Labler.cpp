#include "meshdist_labler/Labler.hpp"

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

  pub_ = it_.advertise("image_out", 1);
  sub_ = it_.subscribeCamera("/camera_stick/cam0/image", 1, &Labler::imageCb,
                             this);
}

Labler::~Labler() {}

bool Labler::readParameters() {
  if (!nodeHandle_.getParam("pc_topic", pc_topic_))
    return false;
  return true;
}

void Labler::gotCloud(const sensor_msgs::PointCloud2 &cloud_msg_in) {
  std::cout << "got cloud" << std::endl;
}

void Labler::imageCb(const sensor_msgs::ImageConstPtr &image_msg,
                     const sensor_msgs::CameraInfoConstPtr &info_msg) {
  std::cout << "got image!" << std::endl;
  cv::Mat image;
  cv_bridge::CvImagePtr input_bridge;
  try {
    input_bridge =
        cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    image = input_bridge->image;
  } catch (cv_bridge::Exception &ex) {
    ROS_ERROR("[draw_frames] Failed to convert image");
    return;
  }

  std::cout << "after failde" << std::endl;
  cam_model_.fromCameraInfo(info_msg);
  std::string frame_id = "base";

  tf::StampedTransform transform;
  try {
    ros::Time acquisition_time = info_msg->header.stamp;
    ros::Duration timeout(1.0 / 30);
    tf_listener_.waitForTransform(cam_model_.tfFrame(), frame_id,
                                  acquisition_time, timeout);
    tf_listener_.lookupTransform(cam_model_.tfFrame(), frame_id,
                                 acquisition_time, transform);
  } catch (tf::TransformException &ex) {
    ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
    return;
  }

  tf::Point pt = transform.getOrigin();
  std::cout << "got opomt" << std::endl;
  /*
  cv::Point3d pt_cv(pt.x(), pt.y(), pt.z());
  cv::Point2d uv;
  uv = cam_model_.project3dToPixel(pt_cv);

  static const int RADIUS = 3;
  cv::circle(image, uv, RADIUS, CV_RGB(255, 0, 0), -1);
  CvSize text_size;
  int baseline;
  cvGetTextSize(frame_id.c_str(), &font_, &text_size, &baseline);
  CvPoint origin =
      cvPoint(uv.x - text_size.width / 2, uv.y - RADIUS - baseline - 3);
cv:
  putText(image, frame_id.c_str(), origin, cv::FONT_HERSHEY_SIMPLEX, 12,
          CV_RGB(255, 0, 0));

  /**

  BOOST_FOREACH (const std::string &frame_id, frame_ids_) {
    tf::StampedTransform transform;
    try {
      ros::Time acquisition_time = info_msg->header.stamp;
      ros::Duration timeout(1.0 / 30);
      tf_listener_.waitForTransform(cam_model_.tfFrame(), frame_id,
                                    acquisition_time, timeout);
      tf_listener_.lookupTransform(cam_model_.tfFrame(), frame_id,
                                   acquisition_time, transform);
    } catch (tf::TransformException &ex) {
      ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
      return;
    }

    tf::Point pt = transform.getOrigin();
    cv::Point3d pt_cv(pt.x(), pt.y(), pt.z());
    cv::Point2d uv;
    uv = cam_model_.project3dToPixel(pt_cv);

    static const int RADIUS = 3;
    cv::circle(image, uv, RADIUS, CV_RGB(255, 0, 0), -1);
    CvSize text_size;
    int baseline;
    cvGetTextSize(frame_id.c_str(), &font_, &text_size, &baseline);
    CvPoint origin =
        cvPoint(uv.x - text_size.width / 2, uv.y - RADIUS - baseline - 3);
  cv:
    putText(image, frame_id.c_str(), origin, cv::FONT_HERSHEY_SIMPLEX, 12,
            CV_RGB(255, 0, 0));
  }
 */
  pub_.publish(input_bridge->toImageMsg());
};

} // namespace meshdist
