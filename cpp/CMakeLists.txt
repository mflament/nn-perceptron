# CMakeList.txt : CMake project for neuralnetwork, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("neuralnetwork")

find_package (JNI REQUIRED)

file(GLOB JNI_HEADERS ${PROJECT_DIR}/jni/*.h)

# Add source to this project's executable.
add_library (neuralnetwork SHARED ${JNI_HEADERS} "StreamBuffer.h" "StreamBuffer.cpp" "Matrix.h" "Matrix.cpp" "TrainingSamples.h" "TrainingSamples.cpp" "NeuralNetwork.cpp")
target_include_directories(neuralnetwork PUBLIC ${JNI_INCLUDE_DIRS})

install (TARGETS neuralnetwork RUNTIME DESTINATION .)
