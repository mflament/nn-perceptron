/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_yah_tests_perceptron_jni_NativeTrainingSamples */

#ifndef _Included_org_yah_tests_perceptron_jni_NativeTrainingSamples
#define _Included_org_yah_tests_perceptron_jni_NativeTrainingSamples
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    create
 * Signature: (ILjava/nio/Buffer;Ljava/nio/Buffer;Ljava/nio/Buffer;)J
 */
JNIEXPORT jlong JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_create
  (JNIEnv *, jclass, jint, jobject, jobject, jobject);

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_delete
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    size
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_size
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    batchSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_batchSize
  (JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif
#endif