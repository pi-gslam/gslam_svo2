cmake_minimum_required(VERSION 2.8)

include(cmake/PICMake.cmake)

list(APPEND CMAKE_CXX_FLAGS  -std=c++11  )

pi_add_target(gslam_svo2 SHARED src REQUIRED System OpenCV3 SVO2 Eigen3)
pi_report_target()

