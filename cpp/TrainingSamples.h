#pragma once

#include <algorithm>

struct TrainingSamples {
	int size = 0;
	int batchSize = 0;
	int features = 0;
	
	double* inputs = 0;
	int* expectedIndices = 0;
};

struct TrainingBatch {
	const TrainingSamples samples;
	int size = 0;
	int offset = 0;

	TrainingBatch(const TrainingSamples& _samples) : samples(_samples) {};

	inline bool hasNext() {
		return offset < samples.size;
	}

	inline void next() {
		size = std::min(samples.batchSize, samples.size - offset);
		offset += size;
	}

	inline const double* inputs() const {
		return samples.inputs + (size_t) offset * samples.features;
	}

	inline int expectedIndex(int index) const {
		return samples.expectedIndices ? samples.expectedIndices[offset + index] : 0;
	}

};
