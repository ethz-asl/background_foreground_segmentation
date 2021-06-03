#!/usr/bin/env python
"""
  ROS node that logs the ICP pose to file
"""
import sys
import rospy
import yaml
import time
import csv
import os
from geometry_msgs.msg import PointStamped


def callback(msgdata, filename):
  receivetime = rospy.get_time()
  print(msgdata, filename)
  with open(filename, 'a') as f:
    writer = csv.writer(f)
    writer.writerow([
        msgdata.header.stamp,
        receivetime,
        msgdata.point.x,
        msgdata.point.y,
        msgdata.point.z,
    ])


if __name__ == '__main__':
  rospy.init_node('pose_logger')

  # get a filename that is unused
  basis_filename = rospy.get_param('~filename')
  # strip csv ending
  if basis_filename.endswith('.csv'):
    basis_filename = basis_filename[:-4]
  if rospy.get_param('~overwrite', False):
    filename = '{}.csv'.format(basis_filename)
  else:
    # check if filename already exists
    i = 1
    filename = '{}_{}.csv'.format(basis_filename, i)
    while os.path.exists(filename):
      i += 1
      filename = '{}_{}.csv'.format(basis_filename, i)

  # create file header
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerow([
        'headerstamp',
        'receivestamp',
        'trans_x',
        'trans_y',
        'trans_z',
    ])

  rospy.Subscriber(rospy.get_param('~topic', '/leica/position'),
                   PointStamped,
                   callback,
                   callback_args=filename)
  rospy.spin()
