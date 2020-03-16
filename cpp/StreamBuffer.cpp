#include "StreamBuffer.h"

StreamBuffer::StreamBuffer(JNIEnv* env, jobject buffer) : pos(0) {
	if (buffer) {
		ptr = env->GetDirectBufferAddress(buffer);
		size = env->GetDirectBufferCapacity(buffer);
	}
	else {
		size = 0;
		ptr = 0;
	}
}

bool StreamBuffer::nextInt(int& res) {
	if (remaining() < sizeof(int)) return false;
	res = *((int*)address());
	pos += sizeof(int);
	return true;
}

bool StreamBuffer::nextMatrix(Matrix& matrix) {
	if (!nextInt(matrix.rows)) return false;
	if (!nextInt(matrix.capacity)) return false;
	size_t size = (size_t)matrix.rows * matrix.capacity * sizeof(double);
	if (remaining() < size) return false;
	matrix.columns = matrix.capacity;
	matrix.data = (double*)address();
	matrix.managed = false;
	pos += size;
	return true;
}

