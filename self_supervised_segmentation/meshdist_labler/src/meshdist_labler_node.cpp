#include "meshdist_labler/labler.hpp"
#include <ros/ros.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "meshdist_labler");
  ros::NodeHandle nodeHandle("~");

  meshdist::Labler Labler(nodeHandle);

  ros::spin();
  return 0;
}
