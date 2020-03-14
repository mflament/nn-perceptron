/**
 * 
 */
package org.yah.tests.perceptron;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * @author Yah
 *
 */
public class MatrixBatchSource<M extends Matrix<M>> implements BatchSource<M> {

    private final MatrixNeuralNetwork<M> network;

    public MatrixBatchSource(MatrixNeuralNetwork<M> network) {
        this.network = network;
    }

    @Override
    public TrainingSet<M> createBatches(double[][] inputs, int[] expecteds, int batchSize,
            boolean transposeInputs) {
        M inputsMatrix = createInputs(inputs, transposeInputs);
        checkExpecteds(expecteds, inputsMatrix.columns());
        M expectedOutputs = createOutputs(expecteds);
        M expectedIndices = createExpectedIndices(expecteds);
        return new MatrixTrainingSet<>(batchSize, inputsMatrix, expectedOutputs, expectedIndices);
    }

    @Override
    public Batch<M> createBatch(double[][] inputs, int[] expecteds, boolean transposeInputs) {
        M inputsMatrix = createInputs(inputs, transposeInputs);
        checkExpecteds(expecteds, inputsMatrix.columns());
        M expectedOutputs = createOutputs(expecteds);
        M expectedIndices = createExpectedIndices(expecteds);
        return new MatrixBatch<>(inputsMatrix, expectedOutputs, expectedIndices);
    }

    private M createInputs(double[][] inputs, boolean transpose) {
        int samples = transpose ? inputs.length : inputs[0].length;
        int inputsFeatures = transpose ? inputs[0].length : inputs.length;
        if (inputsFeatures != network.features()) {
            throw new IllegalArgumentException(
                    "Invalid inputs features: " + inputsFeatures + ", expected "
                            + network.features());
        }
        M res = network.newMatrix(inputsFeatures, samples);
        res.apply(transpose ? (r, c, v) -> inputs[c][r] : (r, c, v) -> inputs[r][c]);
        return res;
    }

    private void checkExpecteds(int[] expecteds, int samples) {
        if (expecteds.length != samples) {
            throw new IllegalArgumentException("Mismatched inputs/expected size: "
                    + samples + "/ " + expecteds.length);
        }
        for (int i = 0; i < expecteds.length; i++) {
            if (expecteds[i] < 0 || expecteds[i] >= network.outputs())
                throw new IllegalArgumentException("Invalid expected index " + expecteds[i]);
        }
    }

    private M createOutputs(int[] expecteds) {
        M res = network.newMatrix(network.outputs(), expecteds.length);
        res.apply((r, c, v) -> expecteds[c] == r ? 1 : 0);
        return res;
    }

    private M createExpectedIndices(int[] expecteds) {
        M res = network.newMatrix(1, expecteds.length);
        res.apply((r, c, v) -> expecteds[c]);
        return res;
    }

    private static final class MatrixTrainingSet<M extends Matrix<M>> implements TrainingSet<M> {
        private final M inputs;
        private final M expectedOutputs;
        private final M expectedIndices;
        private final int batchSize;

        public MatrixTrainingSet(int batchSize, M inputs, M expectedOutputs, M expectedIndices) {
            this.inputs = inputs;
            this.expectedOutputs = expectedOutputs;
            this.expectedIndices = expectedIndices;
            this.batchSize= batchSize;
        }

        @Override
        public Iterator<Batch<M>> iterator() {
            return new MatrixBatchIterator<>(inputs, expectedOutputs, expectedIndices, batchSize);
        }

        @Override
        public int samples() {
            return inputs.columns();
        }

        @Override
        public int batchSize() {
            return batchSize;
        }
        
    }
    
    private static final class MatrixBatchIterator<M extends Matrix<M>>
            implements Iterator<Batch<M>> {

        private final M inputs;
        private final M expectedOutputs;
        private final M expectedIndices;

        private final ThreadLocal<MatrixBatch<M>> batches = new ThreadLocal<MatrixBatch<M>>() {
            @Override
            protected MatrixBatch<M> initialValue() {
                return newBatch();
            }
        };

        private final int batchSize;
        private int offset;
        private int batchIndex;

        public MatrixBatchIterator(M inputs, M expectedOutputs, M expectedIndices, int batchSize) {
            this.inputs = inputs;
            this.expectedOutputs = expectedOutputs;
            this.expectedIndices = expectedIndices;
            this.batchSize = batchSize;
        }

        private MatrixBatch<M> newBatch() {
            return new MatrixBatch<>(inputs.createView(), expectedOutputs.createView(),
                    expectedIndices.createView());
        }

        @Override
        public synchronized boolean hasNext() {
            return offset < inputs.columns();
        }

        @Override
        public synchronized Batch<M> next() {
            if (!hasNext())
                throw new NoSuchElementException();
            MatrixBatch<M> batch = batches.get();
            offset += batch.slide(offset, batchSize, batchIndex++);
            return batch;
        }

    }

    private static final class MatrixBatch<M extends Matrix<M>> implements Batch<M> {
        private final M inputs;
        private final M expectedOutputs;
        private final M expectedIndices;
        private int index;

        public MatrixBatch(M inputs, M expectedOutputs, M expectedIndices) {
            this.inputs = inputs;
            this.expectedOutputs = expectedOutputs;
            this.expectedIndices = expectedIndices;
        }

        public int slide(int offset, int columns, int index) {
            int actualColumns = inputs.slide(offset, columns);
            int s = expectedOutputs.slide(offset, columns);
            assert s == actualColumns;
            s = expectedIndices.slide(offset, columns);
            assert s == actualColumns;
            this.index = index;
            return actualColumns;
        }

        @Override
        public int index() {
            return index;
        }

        @Override
        public M inputs() {
            return inputs;
        }

        @Override
        public M expectedOutputs() {
            return expectedOutputs;
        }

        @Override
        public M expectedIndices() {
            return expectedIndices;
        }
    }
}
