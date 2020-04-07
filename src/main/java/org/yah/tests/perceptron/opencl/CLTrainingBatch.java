package org.yah.tests.perceptron.opencl;

import org.yah.tests.perceptron.base.TrainingBatch;
import org.yah.tools.opencl.mem.CLMemObject;

class CLTrainingBatch implements TrainingBatch  {
    private final CLTrainingSamples samples;
    int offset;
    int batchSize;

    public CLTrainingBatch(CLTrainingSamples samples) {
        this.samples = samples;
    }

    public CLMemObject getInputs() { return samples.inputsBuffer; }

    public CLMemObject getExpectedIndices() { return samples.expectedIndicesBuffer; }

    @Override
    public int size() {
        return batchSize;
    }
}
