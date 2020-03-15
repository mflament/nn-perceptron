/**
 * 
 */
package org.yah.tests.perceptron.matrix;

import org.yah.tests.perceptron.Batch;

/**
 * @author Yah
 *
 */
public class MatrixBatch<M extends Matrix<M>> implements Batch {
    private final M inputs;
    private final M expectedOutputs;
    private final M expectedIndices;
    private int index;

    public MatrixBatch(M inputs) {
        this(inputs, null, null);
    }

    public MatrixBatch(M inputs, M expectedOutputs, M expectedIndices) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
        this.expectedIndices = expectedIndices;
    }

    public int slide(int offset, int columns, int index) {
        this.index = index;
        int newSize = inputs.slide(offset, columns);
        int s = expectedOutputs.slide(offset, columns);
        if (expectedOutputs != null) {
            assert s == newSize;
            s = expectedIndices.slide(offset, columns);
            assert s == newSize;
        }
        return newSize;
    }

    @Override
    public int size() {
        return inputs.columns();
    }

    @Override
    public int index() {
        return index;
    }

    public M inputs() {
        return inputs;
    }

    public M expectedOutputs() {
        return expectedOutputs;
    }

    public M expectedIndices() {
        return expectedIndices;
    }

    public double accuracy(M outputs, int[] outputIndices) {
        int matched = 0;
        int samples = outputs.columns();
        M expectedIndices = expectedIndices();
        assert samples == expectedIndices.columns();
        for (int sample = 0; sample < samples; sample++) {
            int outputIndex = outputs.maxRowIndex(sample);
            if (expectedIndices.get(0, sample) == outputIndex)
                matched++;
            if (outputIndices != null)
                outputIndices[sample] = outputIndex;
        }
        return matched / (double) samples;
    }

}
