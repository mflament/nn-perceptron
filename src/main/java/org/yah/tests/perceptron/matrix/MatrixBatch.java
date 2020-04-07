package org.yah.tests.perceptron.matrix;

import org.yah.tests.perceptron.base.TrainingBatch;

class MatrixBatch<M extends Matrix<M>> implements TrainingBatch {
    final M inputs;
    final int[] expectedIndices;
    int expectedOffset;

    public MatrixBatch(MatrixSamples<M> samples) {
        inputs = samples.inputs.createView();
        expectedIndices = samples.expectedIndices;
    }

    @Override
    public int size() {
        return inputs.columns();
    }

    public int slide(int offset, int columns) {
        int newSize = inputs.slide(offset, columns);
        expectedOffset = offset;
        return newSize;
    }

    public int expectedIndex(int sample) {
        return expectedIndices[expectedOffset + sample];
    }
}
