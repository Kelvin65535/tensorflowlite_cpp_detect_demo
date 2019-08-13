cmake_minimum_required(VERSION 3.7)
project(tflite_cpp)

set(CMAKE_CXX_STANDARD 14)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(tflite_cpp main.cpp)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

target_link_libraries(tflite_cpp ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(tflite_cpp ${CMAKE_SOURCE_DIR}/lib/libflatbuffers.a)
target_link_libraries(tflite_cpp ${CMAKE_SOURCE_DIR}/lib/libtensorflow-lite.a)
#target_link_libraries(tflite_cpp /usr/lib/arm-linux-gnueabihf/libopencv_core.so.2.4.9)
#target_link_libraries(tflite_cpp ${CMAKE_SYSROOT}/usr/lib/arm-linux-gnueabihf/libopencv_highgui.so.2.4.9)
#target_link_libraries(tflite_cpp ${CMAKE_SYSROOT}/usr/lib/arm-linux-gnueabihf/libopencv_imgproc.so.2.4.9)
