/**
 * 
 */
package org.yah.tests.perceptron.mt;

import java.util.Iterator;
import java.util.NoSuchElementException;

import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;

/**
 * @author Yah
 *
 */
public class MTSamplesSource implements SamplesSource {

    private final MTNeuralNetwork network;

    public MTSamplesSource(MTNeuralNetwork network) {
        this.network = network;
    }

    @Override
    public InputSamples createInputs(SamplesProvider provider, int batchSize) {
        MTMatrix inputs = createInputs(provider);
        return new MTTrainingSamples(inputs, batchSize);
    }

    @Override
    public TrainingSamples createTraining(TrainingSamplesProvider provider, int batchSize) {
        MTMatrix inputs = createInputs(provider);
        int samples = provider.samples();
        int[] expectedIndices = new int[samples];
        MTMatrix expectedOutputs = new MTMatrix(network.outputs(), samples);
        for (int sample = 0; sample < samples; sample++) {
            int expectedIndex = provider.outputIndex(sample);
            expectedIndices[sample] = expectedIndex;
            expectedOutputs.set(expectedIndex, sample, 1);
        }
        return new MTTrainingSamples(inputs, expectedOutputs, expectedIndices, batchSize);
    }

    private MTMatrix createInputs(SamplesProvider provider) {
        int samples = provider.samples();
        int features = network.features();
        MTMatrix inputs = new MTMatrix(features, samples);
        for (int sample = 0; sample < samples; sample++) {
            for (int feature = 0; feature < features; feature++) {
                inputs.set(feature, sample, provider.input(sample, feature));
            }
        }
        return inputs;
    }

    public static class MTTrainingSamples implements TrainingSamples, Iterable<MTBatch> {
        private final int batchSize;

        final MTMatrix inputs;
        final MTMatrix expectedOutputs;
        final int[] expectedIndices;

        public MTTrainingSamples(MTMatrix inputs, int batchSize) {
            this(inputs, null, null, batchSize);
        }

        public MTTrainingSamples(MTMatrix inputs, MTMatrix expectedOutputs,
                int[] expectedIndices, int batchSize) {
            this.inputs = inputs;
            this.expectedOutputs = expectedOutputs;
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
            private final MTBatch batch = new MTBatch(inputs, expectedOutputs, expectedIndices);

            private int offset;

            @Override
            public boolean hasNext() {
                return offset < size();
            }

            @Override
            public MTBatch next() {
                if (!hasNext())
                    throw new NoSuchElementException();
                int size = Math.min(batchSize, inputs.columns() - offset);
                batch.slide(offset, size);
                offset += size;
                return batch;
            }
        }
    }

    public static class MTBatch {
        private final MTMatrix inputs;
        private final MTMatrix expectedOutputs;
        private final int[] expectedIndices;

        private int offset;
        private int batchSize;

        public MTBatch(MTMatrix inputs, MTMatrix expectedOutputs, int[] expectedIndices) {
            this.inputs = new MTMatrix(inputs);
            this.expectedOutputs = expectedOutputs != null ? new MTMatrix(expectedOutputs) : null;
            this.expectedIndices = expectedIndices;
        }

        private void slide(int offset, int size) {
            this.offset = offset;
            this.batchSize = size;
            inputs.offset(offset * inputs.rows());
            inputs.columns(batchSize);
            if (expectedOutputs != null) {
                expectedOutputs.offset(offset * expectedOutputs.rows());
                expectedOutputs.columns(batchSize);
            }
        }

        public int batchSize() {
            return batchSize;
        }

        public MTMatrix inputs() {
            return inputs;
        }

        public MTMatrix expectedOutputs() {
            return expectedOutputs;
        }

        public int[] expectedIndices() {
            return expectedIndices;
        }

        public int expectedIndex(int sample) {
            return expectedIndices[offset + sample];
        }

        public int offset() {
            return offset;
        }

        public boolean hasExpecteds() {
            return expectedIndices != null;
        }
    }

}
