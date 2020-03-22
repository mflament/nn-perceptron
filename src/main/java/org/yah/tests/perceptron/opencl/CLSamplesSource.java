/**
 * 
 */
package org.yah.tests.perceptron.opencl;

import java.util.Iterator;
import java.util.NoSuchElementException;

import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tools.opencl.mem.BufferProperties;
import org.yah.tools.opencl.mem.CLBuffer;
import org.yah.tools.opencl.mem.CLMemObject;

/**
 * @author Yah
 *
 */
class CLSamplesSource implements SamplesSource {

    private final CLNeuralNetwork network;

    public CLSamplesSource(CLNeuralNetwork network) {
        this.network = network;
    }

    @Override
    public InputSamples createInputs(SamplesProvider provider, int batchSize) {
        CLBuffer inputsBuffer = network.createMatrixBuffer(network.features(), provider.samples(),
                (r, c) -> (float) provider.input(c, r), BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_ONLY, BufferProperties.MEM_HOST_WRITE_ONLY);
        return new CLTrainingSamples(provider.samples(), batchSize, inputsBuffer);
    }

    @Override
    public TrainingSamples createTraining(TrainingSamplesProvider provider, int batchSize) {
        CLBuffer inputsBuffer = network.createMatrixBuffer(network.features(), provider.samples(),
                (r, c) -> (float) provider.input(c, r), BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_ONLY, BufferProperties.MEM_HOST_WRITE_ONLY);

        CLBuffer expectedOutputsBuffer = network.createMatrixBuffer(network.outputs(),
                provider.samples(),
                (r, c) -> provider.outputIndex(c) == r ? 1f : 0f,
                BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_ONLY, BufferProperties.MEM_HOST_WRITE_ONLY);

        int[] expectedIndices = new int[provider.samples()];
        for (int i = 0; i < expectedIndices.length; i++) {
            expectedIndices[i] = provider.outputIndex(i);
        }

        return new CLTrainingSamples(provider.samples(), batchSize, inputsBuffer,
                expectedOutputsBuffer, expectedIndices);
    }

    public static class CLTrainingBatch {
        private final CLTrainingSamples samples;
        public int offset;
        public int batchSize;

        public CLTrainingBatch(CLTrainingSamples samples) {
            this.samples = samples;
        }

        public CLMemObject getInputs() { return samples.inputsBuffer; }

        public CLMemObject getExpectedOutputs() { return samples.expectedOutputsBuffer; }

        public int getExpectedIndex(int sample) {
            return samples.expectedIndices[offset + sample];
        }
    }

    static private class CLTrainingBatchIterator implements Iterator<CLTrainingBatch> {

        private final CLTrainingSamples samples;
        private final CLTrainingBatch batch;
        private int offset;

        public CLTrainingBatchIterator(CLTrainingSamples samples) {
            this.samples = samples;
            batch = new CLTrainingBatch(samples);
        }

        @Override
        public boolean hasNext() {
            return offset < samples.size;
        }

        @Override
        public CLTrainingBatch next() {
            if (!hasNext())
                throw new NoSuchElementException();

            batch.offset = offset;
            batch.batchSize = Math.min(samples.batchSize, samples.size - offset);
            offset += batch.batchSize;
            return batch;
        }
    }

    static class CLTrainingSamples implements TrainingSamples, AutoCloseable, Iterable<CLTrainingBatch> {
        private final int size;
        private final int batchSize;

        final CLBuffer inputsBuffer;
        final CLBuffer expectedOutputsBuffer;
        final int[] expectedIndices;

        public CLTrainingSamples(int size, int batchSize, CLBuffer inputsBuffer) {
            this(size, batchSize, inputsBuffer, null, null);
        }

        public CLTrainingSamples(int size, int batchSize,
                CLBuffer inputsBuffer,
                CLBuffer expectedOutputsBuffer,
                int[] expectedIndices) {
            this.size = size;
            this.batchSize = batchSize == 0 ? size : batchSize;
            this.inputsBuffer = inputsBuffer;
            this.expectedOutputsBuffer = expectedOutputsBuffer;
            this.expectedIndices = expectedIndices;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int batchSize() {
            return batchSize;
        }

        @Override
        public Iterator<CLTrainingBatch> iterator() {
            return new CLTrainingBatchIterator(this);
        }

        @Override
        public void close() throws Exception {
            inputsBuffer.close();
            if (expectedOutputsBuffer != null) { expectedOutputsBuffer.close(); }
        }

        public int expectedIndex(int sample) {
            return expectedIndices[sample];
        }
    }
}
