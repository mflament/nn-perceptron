package org.yah.tests.perceptron.opencl;

import org.yah.tests.perceptron.base.BatchedSamples;
import org.yah.tools.opencl.mem.CLBuffer;

import java.util.Iterator;
import java.util.NoSuchElementException;

class CLTrainingSamples implements BatchedSamples<CLTrainingBatch>, AutoCloseable {
    private final int size;
    private final int batchSize;

    final CLBuffer inputsBuffer;
    final CLBuffer expectedIndicesBuffer;

    CLTrainingSamples(int size, int batchSize, CLBuffer inputsBuffer) {
        this(size, batchSize, inputsBuffer, null);
    }

    CLTrainingSamples(int size, int batchSize,
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

    private static class CLTrainingBatchIterator implements Iterator<CLTrainingBatch> {
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
}
