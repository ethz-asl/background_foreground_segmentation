cmake_minimum_required(VERSION 2.8.3)
project(background_foreground_segmentation)

find_package(catkin_simple REQUIRED)

find_package(
        Boost
)


catkin_python_setup()
catkin_simple(ALL_DEPS_REQUIRED)


add_definitions(-std=c++14 -Wall -Wextra)


#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME}
        src/dataset_creator.cpp)

cs_add_library(${PROJECT_NAME}2
        src/dataset_creator2.cpp)

############
# BINARIES #
############
cs_add_executable(${PROJECT_NAME}_node
        src/background_foreground_segmentation_node.cpp
        )
cs_add_executable(${PROJECT_NAME}_node2
        src/background_foreground_segmentation_node2.cpp
        )

cs_add_executable(cloud_filtering_node
        src/cloud_filtering_node.cpp
        )

target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_node2 ${PROJECT_NAME}2)

############
#  TESTS   #
############
#catkin_add_gtest(${PROJECT_NAME}_tests
#       )

#target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()


catkin_install_python(PROGRAMS
    src/pose_logger.py
    src/point_logger.py
    src/label_aggregator.py
    src/inference_node.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
