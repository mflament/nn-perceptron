#include "Matrix.h"

struct TrainingSamples {
	int size;
	int batchSize;
	Matrix inputs;
	Matrix expectedOutputs;
	Matrix expectedIndices;
};
