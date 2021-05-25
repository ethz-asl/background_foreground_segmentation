#include "dataset_creator/dataset_creator_dense.hpp"
#include <ros/ros.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "background_foreground_segmentation_dense");
  ros::NodeHandle nodeHandle("~");

  dataset_creator_dense::Creator Creator(nodeHandle);

  ros::spin();
  return 0;
}
