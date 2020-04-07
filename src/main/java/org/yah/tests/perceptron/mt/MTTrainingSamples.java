package org.yah.tests.perceptron.mt;

import org.yah.tests.perceptron.base.BatchedSamples;

import java.util.Iterator;
import java.util.NoSuchElementException;

class MTTrainingSamples implements BatchedSamples<MTBatch> {
    private final int batchSize;
    final MTMatrix inputs;
    final int[] expectedIndices;

    public MTTrainingSamples(MTMatrix inputs, int batchSize) {
        this(inputs, null, batchSize);
    }

    public MTTrainingSamples(MTMatrix inputs, int[] expectedIndices, int batchSize) {
        this.inputs = inputs;
        this.expectedIndices = expectedIndices;
        this.batchSize = batchSize == 0 ? inputs.columns() : batchSize;
    }

    @Override
    public int size() {
        return inputs.columns();
    }

    @Override
    public int batchSize() {
        return batchSize;
    }

    @Override
    public Iterator<MTBatch> iterator() {
        return new BatchIterator();
    }

    private class BatchIterator implements Iterator<MTBatch> {
        private final MTBatch batch = new MTBatch(inputs, expectedIndices);

        private int offset;

        @Override
        public boolean hasNext() {
            return offset < size();
        }

        @Override
        public MTBatch next() {
            if (!hasNext())
                throw new NoSuchElementException();
            int size = batch.slide(offset, batchSize);
            offset += size;
            return batch;
        }
    }
}
