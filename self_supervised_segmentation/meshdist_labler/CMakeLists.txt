cmake_minimum_required(VERSION 2.8.3)
project(meshdist_labler)

find_package(catkin_simple REQUIRED)
catkin_python_setup()
catkin_simple(ALL_DEPS_REQUIRED)


add_definitions(-std=c++14 -Wall -Wextra)


#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME}
        src/Labler.cpp)

############
# BINARIES #
############
cs_add_executable(${PROJECT_NAME}_node
        src/meshdist_labler_node.cpp
        )

target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

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
