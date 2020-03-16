#pragma once

#include "Matrix.h"

struct TrainingSamples {
	int batchSize = 0;
	Matrix inputs;
	Matrix expectedOutputs;
	Matrix expectedIndices;
	
	inline int size() {
		return inputs.columns;
	}

	int slide(int offset, int columns);
	void rewind();
};

