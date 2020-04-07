#include "StreamBuffer.h"
#include "TrainingSamples.h"
#include <iostream>

/*
int TrainingSamples::slide(int offset, int columns) {
	int res = inputs.slide(offset, columns);
	if (expectedOutputs.maxColumns) {
		expectedOutputs.slide(offset, columns);
		expectedIndices.slide(offset, columns);
	}
	return res;
}

void TrainingSamples::rewind() {
	inputs.slide(0, inputs.maxColumns);
	if (expectedOutputs.maxColumns) {
		expectedOutputs.slide(0, expectedOutputs.maxColumns);
		expectedIndices.slide(0, expectedIndices.maxColumns);
	}
}
*/
