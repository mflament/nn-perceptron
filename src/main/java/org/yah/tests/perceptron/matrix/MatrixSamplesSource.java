/**
 * 
 */
package org.yah.tests.perceptron.matrix;

import java.util.Iterator;
import java.util.NoSuchElementException;

import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;

/**
 * Note: all inputs are expected to be column major. They can be transposed
 * using the corresponding parameter if necessary.
 * 
 * @author Yah
 *
 */
public class MatrixSamplesSource<M extends Matrix<M>> implements SamplesSource {

    private final MatrixNeuralNetwork<M> network;

    public MatrixSamplesSource(MatrixNeuralNetwork<M> network) {
        this.network = network;
    }

    @Override
    public MatrixSamples<M> createInputs(SamplesProvider provider, int batchSize) {
        M inputsMatrix = createInputs(provider);
        return new MatrixSamples<>(batchSize, inputsMatrix);
    }

    @Override
    public MatrixSamples<M> createTraining(TrainingSamplesProvider provider, int batchSize) {
        M inputsMatrix = createInputs(provider);
        checkExpecteds(provider, inputsMatrix.columns());
        M expectedOutputs = createOutputs(provider);
        M expectedIndices = createExpectedIndices(provider);
        return new MatrixSamples<>(batchSize, inputsMatrix, expectedOutputs, expectedIndices);
    }

    private M createInputs(SamplesProvider provider) {
        M res = network.newMatrix(network.features(), provider.samples());
        res.apply((r, c, v) -> provider.input(c, r));
        return res;
    }

    private void checkExpecteds(TrainingSamplesProvider provider, int samples) {
        for (int i = 0; i < samples; i++) {
            int index = provider.outputIndex(i);
            if (index < 0 || index >= network.outputs())
                throw new IllegalArgumentException("Invalid expected index " + index);
        }
    }

    private M createOutputs(TrainingSamplesProvider provider) {
        M res = network.newMatrix(network.outputs(), provider.samples());
        res.apply((r, c, v) -> provider.outputIndex(c) == r ? 1 : 0);
        return res;
    }

    private M createExpectedIndices(TrainingSamplesProvider provider) {
        M res = network.newMatrix(1, provider.samples());
        res.apply((r, c, v) -> provider.outputIndex(c));
        return res;
    }

    static final class MatrixSamples<M extends Matrix<M>>
            implements TrainingSamples, Iterable<MatrixBatch<M>> {
        private final M inputs;
        private final M expectedOutputs;
        private final M expectedIndices;
        private final int batchSize;

        public MatrixSamples(int batchSize, M inputs) {
            this(batchSize, inputs, null, null);
        }

        public MatrixSamples(int batchSize, M inputs, M expectedOutputs, M expectedIndices) {
            this.batchSize = batchSize == 0 ? inputs.columns() : batchSize;
            this.inputs = inputs;
            this.expectedOutputs = expectedOutputs;
            this.expectedIndices = expectedIndices;
        }

        @Override
        public Iterator<MatrixBatch<M>> iterator() {
            return new MatrixBatchIterator<>(this);
        }

        @Override
        public int size() {
            return inputs.columns();
        }

        @Override
        public int batchSize() {
            return batchSize;
        }

        M inputs() {
            return inputs;
        }

        M expectedIndices() {
            return expectedIndices;
        }

        M expectedOutputs() {
            return expectedOutputs;
        }

    }

    private static final class MatrixBatchIterator<M extends Matrix<M>>
            implements Iterator<MatrixBatch<M>> {

        private final MatrixSamples<M> samples;

        private final ThreadLocal<MatrixBatch<M>> batches = new ThreadLocal<MatrixBatch<M>>() {
            @Override
            protected MatrixBatch<M> initialValue() {
                return newBatch();
            }
        };

        private int offset;

        public MatrixBatchIterator(MatrixSamples<M> samples) {
            this.samples = samples;
        }

        private MatrixBatch<M> newBatch() {
            return new MatrixBatch<>(samples);
        }

        @Override
        public synchronized boolean hasNext() {
            return offset < samples.size();
        }

        @Override
        public synchronized MatrixBatch<M> next() {
            if (!hasNext())
                throw new NoSuchElementException();
            MatrixBatch<M> batch = batches.get();
            offset += batch.slide(offset, samples.batchSize);
            return batch;
        }
    }

    static class MatrixBatch<M extends Matrix<M>> {
        private final M inputs;
        private final M expectedOutputs;
        private final M expectedIndices;

        public MatrixBatch(MatrixSamples<M> samples) {
            inputs = samples.inputs.createView();
            if (samples.expectedIndices != null) {
                expectedIndices = samples.expectedIndices.createView();
                expectedOutputs = samples.expectedOutputs.createView();
            } else {
                expectedIndices = expectedOutputs = null;
            }
        }

        public int slide(int offset, int columns) {
            int newSize = inputs.slide(offset, columns);
            int s = expectedOutputs.slide(offset, columns);
            if (expectedOutputs != null) {
                assert s == newSize;
                s = expectedIndices.slide(offset, columns);
                assert s == newSize;
            }
            return newSize;
        }

        public int batchSize() {
            return inputs.columns();
        }

        public int countMatchedOutputs(int[] actualOutputs) {
            int matched = 0;
            int samples = batchSize();
            for (int sample = 0; sample < samples; sample++) {
                if (expectedIndices.get(0, sample) == actualOutputs[sample])
                    matched++;
            }
            return matched;
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

    }

}
