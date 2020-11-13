#include "dataset_creator/dataset_creator.hpp"
#include <ros/ros.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "dataset_creator");
  ros::NodeHandle nodeHandle("~");

  dataset_creator::Creator Creator(nodeHandle);

  ros::spin();
  return 0;
}
