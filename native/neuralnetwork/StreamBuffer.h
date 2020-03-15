
struct StreamBuffer {
	int size;
	int pos;
	void* ptr;

	StreamBuffer(int _size, void* _ptr) : size(_size), prt(_ptr), pos(0) {}

	int remaining() {
		return size - pos;
	}

	int nextInt() {
		if (remaining() < sizeof(int)) return -1;
		int res = *((int*)(ptr + pos));
		pos += sizeof(int);
		return res;
	}

	double nextDouble() {
		if (remaining() < sizeof(double)) return -1;
		double res = *((double*)(ptr + pos));
		pos += sizeof(double);
		return res;
	}
};

