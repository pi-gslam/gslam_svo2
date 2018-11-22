include(/opt/opencv-3.3.0/share/OpenCV/OpenCVConfig.cmake)

set(OPENCV3_FOUND TRUE)
set(OPENCV3_LIBS ${OpenCV_LIBS})
set(OPENCV3_INCLUDES ${OpenCV_INCLUDE_DIRS})
set(OPENCV3_VERSION ${OpenCV_VERSION})
