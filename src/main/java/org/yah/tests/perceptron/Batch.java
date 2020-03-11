package org.yah.tests.perceptron;

import static org.yah.tests.perceptron.Matrix.matrix;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class Batch {
    public final double[][] inputs;
    public final int[] expectedIndices;
    public final double[][] expectedMatrix;

    public Batch(double[][] inputs, int[] expectedIndices, int outputs) {
        this.inputs = inputs;
        this.expectedIndices = expectedIndices;
        this.expectedMatrix = Labels.toExpectedMatrix(expectedIndices, 0, outputs);
    }

    public Batch(double[][] inputs, double[][] expectedMatrix) {
        this.inputs = inputs;
        this.expectedMatrix = expectedMatrix;
        this.expectedIndices = new int[size()];
        Labels.toExpectedIndex(expectedMatrix, expectedIndices);
    }

    public NeuralNetwork network() {
        return this.network();
    }

    public int size() {
        return inputs[0].length;
    }

    public double accuracy(double[][] outputs) {
        return Labels.accuracy(outputs, expectedIndices);
    }

    public double accuracy(int[] outputs) {
        return Labels.accuracy(outputs, expectedIndices);
    }
    
    public interface BatchSource {
        int size();

        int batchSize();

        void load(int index, int size, Batch batch);
    }
    
    public static Iterator<Batch> iterator(NeuralNetwork network, BatchSource source) {
        return new BatchIterator(network, source);
    }

    private static class BatchIterator implements Iterator<Batch> {

        private final BatchSource batchSource;
        private final NeuralNetwork network;
        private int remaining;
        private Batch batch;

        public BatchIterator(NeuralNetwork network, BatchSource batchSource) {
            this.batchSource = batchSource;
            this.network = network;
            remaining = batchSource.size();
        }

        @Override
        public boolean hasNext() {
            return remaining > 0;
        }

        @Override
        public Batch next() {
            if (!hasNext())
                throw new NoSuchElementException();
            int size = Math.min(batchSource.batchSize(), remaining);
            if (batch == null || size != batch.size()) {
                batch = new Batch(matrix(network.features(), remaining), matrix(network.outputs(), remaining));
            }
            int startIndex = batchSource.size() - remaining;
            batchSource.load(startIndex, size, batch);
            remaining -= size;
            return batch;
        }
    }

    public static class ArrayBatchSource implements BatchSource {
        private final double[][] inputs;
        private final int[] expected;
        private final int batchSize;

        public ArrayBatchSource(double[][] inputs, int[] expected) {
            this(inputs, expected, inputs.length);
        }

        public ArrayBatchSource(double[][] inputs, int[] expected, int batchSize) {
            if (inputs.length != expected.length)
                throw new IllegalArgumentException("Size mismatch");
            this.inputs = inputs;
            this.expected = expected;
            this.batchSize = batchSize;
        }

        @Override
        public int size() {
            return inputs.length;
        }

        @Override
        public int batchSize() {
            return batchSize;
        }

        @Override
        public void load(int index, int size, Batch batch) {
            Matrix.zero(batch.expectedMatrix);
            for (int i = 0; i < size; i++) {
                int row = i + index;
                for (int col = 0; col < inputs[0].length; col++) {
                    batch.inputs[col][i] = inputs[row][col];
                }
                batch.expectedIndices[i] = expected[row];
                batch.expectedMatrix[expected[row]][i] = 1;
            }
        }
    }
}