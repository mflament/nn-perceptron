#include "jni/org_yah_tests_perceptron_jni_NativeTrainingSamples.h"

#include "StreamBuffer.h"
#include "TrainingSamples.h"
#include <iostream>

int TrainingSamples::slide(int offset, int columns) {
	int res = inputs.slide(offset, columns);
	if (expectedOutputs.capacity) {
		expectedOutputs.slide(offset, columns);
		expectedIndices.slide(offset, columns);
	}
	return res;
}

void TrainingSamples::rewind() {
	inputs.slide(0, inputs.capacity);
	if (expectedOutputs.capacity) {
		expectedOutputs.slide(0, expectedOutputs.capacity);
		expectedIndices.slide(0, expectedIndices.capacity);
	}
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    create
 * Signature: (ILjava/nio/Buffer;Ljava/nio/Buffer;Ljava/nio/Buffer;)J
 */
JNIEXPORT jlong JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_create(JNIEnv* env, jclass, jint batchSize, jobject javaInputsMatrix, jobject javaOutputsMatrix, jobject javaOutputsIndices) {
	TrainingSamples samples;
	StreamBuffer sb(env, javaInputsMatrix);
	if (!sb.nextMatrix(samples.inputs)) return 0;
	if (javaOutputsMatrix) {
		sb = StreamBuffer(env, javaOutputsMatrix);
		if (!sb.nextMatrix(samples.expectedOutputs)) return 0;
		if (!javaOutputsIndices) return 0;
		sb = StreamBuffer(env, javaOutputsIndices);
		if (!sb.nextMatrix(samples.expectedIndices)) return 0;
	}
	samples.batchSize = batchSize == 0 ? samples.inputs.columns : batchSize;

	TrainingSamples* res = new TrainingSamples();
	memcpy(res, &samples, sizeof(TrainingSamples));
	return (jlong)res;
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_delete(JNIEnv*, jobject, jlong address) {
	delete ((TrainingSamples*)address);
}						

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    size
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_size(JNIEnv*, jclass, jlong address) {
	return ((TrainingSamples*)address)->size();
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeTrainingSamples
 * Method:    batchSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeTrainingSamples_batchSize(JNIEnv*, jclass, jlong address) {
	return ((TrainingSamples*)address)->batchSize;
}
