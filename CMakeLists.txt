CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(Projet)

ADD_EXECUTABLE(correcter main.cpp image.cpp image.h decomposition.cpp decomposition.h initialization.cpp initialization.h motion.cpp motion.h interpolation.cpp interpolation.h)
TARGET_LINK_LIBRARIES(correcter ${OpenCV_LIBS})

