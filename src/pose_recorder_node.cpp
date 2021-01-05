#include <fstream>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

std::string image_topic;
std::string camera_frame;
std::string out_file;

tf::TransformListener *tf_listener;

void callback(const sensor_msgs::ImageConstPtr &msg) {

  std_msgs::Header h = msg->header;
  std::string timestamp = std::to_string(h.stamp.toSec());

  std::ofstream outfile;
  outfile.open(out_file, std::ios_base::app);
  outfile << timestamp;

  try {
    std::shared_ptr<tf::StampedTransform> map_transform(
        new tf::StampedTransform);
    tf_listener->waitForTransform("/map", camera_frame, msg->header.stamp,
                                  ros::Duration(0.5));
    tf_listener->lookupTransform("/map", camera_frame, h.stamp, *map_transform);

    tf::Vector3 origin = map_transform->getOrigin();
    tf::Quaternion rotation = map_transform->getRotation();

    outfile << "," << origin[0] << "," << origin[1] << "," << origin[2] << ","
            << rotation.x() << "," << rotation.y() << "," << rotation.z() << ","
            << rotation.w() << "," << std::endl;

  } catch (const std::exception &ex) {
    std::cerr << "Error occured: " << ex.what() << std::endl;
    outfile << ",,,,,,,," << std::endl;
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "pose_recorder");
  ros::NodeHandle nh;
  ros::NodeHandle("~").getParam("imageTopic", image_topic);
  ros::NodeHandle("~").getParam("cameraFrame", camera_frame);
  ros::NodeHandle("~").getParam("outputFile", out_file);

  tf::TransformListener tl;
  tf_listener = &tl;

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe(image_topic, 10, callback);

  std::ofstream outfile;
  outfile.open(out_file);
  outfile << "Timestamp,x_pos,y_pos,z_pos,x,y,z,w" << std::endl;
  outfile.close();

  ros::spin();
  return 0;
}
