#pragma once

#include <jni.h>
#include "Matrix.h"

struct StreamBuffer {
	size_t size;
	size_t pos;
	void* ptr;

	StreamBuffer(JNIEnv* env, jobject buffer);

	inline size_t remaining() { return size - pos; }

	inline void* address() { return (char*)ptr + pos; }

	bool nextInt(int& res);

	bool nextMatrix(Matrix& matrix);
};

