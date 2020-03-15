#include "jni/org_yah_tests_perceptron_jni_NativeTrainingSamples.h"

#include "TrainingSamples.h"
#include <iostream>

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    create
 * Signature: (ILjava/nio/Buffer;Ljava/nio/Buffer;Ljava/nio/Buffer;)J
 */
JNIEXPORT jlong JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_create(JNIEnv*, jclass, jint, jobject, jobject, jobject) {
    return 0;
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_delete(JNIEnv*, jclass, jlong) {

}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    size
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_size(JNIEnv*, jclass, jlong) {
    return 0;
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    batchSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_batchSize(JNIEnv*, jclass, jlong) {
    return 0;
}
