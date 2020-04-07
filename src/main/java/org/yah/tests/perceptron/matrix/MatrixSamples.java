package org.yah.tests.perceptron.matrix;

import org.yah.tests.perceptron.base.BatchedSamples;

import java.util.Iterator;
import java.util.NoSuchElementException;

final class MatrixSamples<M extends Matrix<M>>
        implements BatchedSamples<MatrixBatch<M>>, Iterable<MatrixBatch<M>> {
    final M inputs;
    final int[] expectedIndices;
    final int batchSize;

    public MatrixSamples(int batchSize, M inputs) {
        this(batchSize, inputs, null);
    }

    public MatrixSamples(int batchSize, M inputs, int[] expectedIndices) {
        this.batchSize = batchSize == 0 ? inputs.columns() : batchSize;
        this.inputs = inputs;
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

    private static final class MatrixBatchIterator<M extends Matrix<M>>
            implements Iterator<MatrixBatch<M>> {
        private final MatrixSamples<M> samples;
        private final MatrixBatch<M> batch;
        private int offset;

        public MatrixBatchIterator(MatrixSamples<M> samples) {
            this.samples = samples;
            this.batch = new MatrixBatch<>(samples);
        }

        @Override
        public synchronized boolean hasNext() {
            return offset < samples.size();
        }

        @Override
        public synchronized MatrixBatch<M> next() {
            if (!hasNext())
                throw new NoSuchElementException();
            offset += batch.slide(offset, samples.batchSize);
            return batch;
        }
    }

}
