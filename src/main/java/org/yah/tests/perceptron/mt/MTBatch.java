package org.yah.tests.perceptron.mt;

import org.yah.tests.perceptron.base.TrainingBatch;

class MTBatch implements TrainingBatch  {
    final MTMatrix inputs;
    int[] expectedIndices;

    private int offset;
    private int batchSize;

    public MTBatch(MTMatrix inputs, int[] expectedIndices) {
        this.inputs = new MTMatrix(inputs);
        this.expectedIndices = expectedIndices;
    }

    int slide(int offset, int size) {
        this.offset = offset;
        batchSize = Math.min(size, inputs.maxColumn() - offset);
        inputs.offset(offset * inputs.rows());
        inputs.columns(batchSize);
        return batchSize;
    }

    public int expectedIndex(int sample) {
        return expectedIndices == null ? -1 : expectedIndices[offset + sample];
    }

    @Override
    public int size() {
        return batchSize;
    }

    public int offset() {
        return offset;
    }

}
