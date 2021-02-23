#include "dataset_creator/dataset_creator2.h"
#include <ros/ros.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "background_foreground_segmentation");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  dataset_creator::Creator Creator(nh, nh_private);

  ros::spin();
  return 0;
}
