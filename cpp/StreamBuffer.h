#pragma once

#include <jni.h>
#include <algorithm>

class StreamBuffer {
	size_t size;
	size_t pos;
	void* ptr;

public:
	StreamBuffer(JNIEnv* env, jobject buffer) : pos(0) {
		if (buffer) {
			ptr = env->GetDirectBufferAddress(buffer);
			size = env->GetDirectBufferCapacity(buffer);
		}
		else {
			size = 0;
			ptr = 0;
		}
	};
	inline size_t remaining() { return std::max((size_t)0, size - pos); }
	inline void* address() { return (char*)ptr + pos; }

	template<typename T> bool next(T& res) {
		if (remaining() < sizeof(T)) return false;
		res = *((T*)address());
		pos += sizeof(T);
		return true;
	}

	template<typename T> bool array(T*& res, const unsigned int length) {
		int bytes = sizeof(T) * length;
		if (remaining() < bytes) return false;
		res = (T*)address();
		pos += bytes;
		return true;
	}
};

