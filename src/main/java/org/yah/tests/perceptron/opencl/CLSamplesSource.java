package org.yah.tests.perceptron.opencl;

import java.nio.ByteBuffer;
import java.util.Iterator;
import java.util.NoSuchElementException;

import org.lwjgl.BufferUtils;
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
        CLBuffer inputsBuffer = createInputsBuffer(provider);
        return new CLTrainingSamples(provider.samples(), batchSize, inputsBuffer);
    }

    @Override
    public TrainingSamples createTraining(TrainingSamplesProvider provider, int batchSize) {
        CLBuffer inputsBuffer = createInputsBuffer(provider);

        int samples = provider.samples();
        ByteBuffer buffer = BufferUtils.createByteBuffer(samples * Integer.BYTES);
        for (int i = 0; i < samples; i++) {
            buffer.putInt(provider.outputIndex(i));
        }
        buffer.flip();
        CLBuffer expectedIndicesBuffer = network.environment.mem(buffer,
                BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_ONLY,
                BufferProperties.MEM_HOST_WRITE_ONLY);
        return new CLTrainingSamples(samples, batchSize, inputsBuffer, expectedIndicesBuffer);
    }

    private CLBuffer createInputsBuffer(SamplesProvider provider) {
        return network.createMatrixBuffer(network.features(), provider.samples(),
                (r, c) -> provider.input(c, r), BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_ONLY, BufferProperties.MEM_HOST_NO_ACCESS);
    }

    static class CLTrainingBatch {
        private final CLTrainingSamples samples;
        int offset;
        int batchSize;

        public CLTrainingBatch(CLTrainingSamples samples) {
            this.samples = samples;
        }

        public CLMemObject getInputs() { return samples.inputsBuffer; }

        public CLMemObject getExpectedIndices() { return samples.expectedIndicesBuffer; }
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
        final CLBuffer expectedIndicesBuffer;

        public CLTrainingSamples(int size, int batchSize, CLBuffer inputsBuffer) {
            this(size, batchSize, inputsBuffer, null);
        }

        public CLTrainingSamples(int size, int batchSize,
                CLBuffer inputsBuffer,
                CLBuffer expectedIndicesBuffer) {
            this.size = size;
            this.batchSize = batchSize == 0 ? size : batchSize;
            this.inputsBuffer = inputsBuffer;
            this.expectedIndicesBuffer = expectedIndicesBuffer;
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
        public void close() {
            inputsBuffer.close();
            if (expectedIndicesBuffer != null) {
                expectedIndicesBuffer.close();
            }
        }

    }
}
