﻿# CMakeList.txt : CMake project for neuralnetwork, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("neuralnetwork")

find_package (JNI REQUIRED)

# Add source to this project's executable.
add_library (neuralnetwork SHARED "StreamBuffer.h" "TrainingSamples.h" "NeuralNetwork.cpp")
target_include_directories(neuralnetwork PUBLIC ${JNI_INCLUDE_DIRS} "jni")

install (TARGETS neuralnetwork RUNTIME DESTINATION .)
