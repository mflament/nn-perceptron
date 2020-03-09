package org.yah.tests.perceptron;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class NeuralNetwork {

    private final int[] layerSizes;

    private final Matrix[] weights;
    private final Matrix[] biases;

    public NeuralNetwork(int... layerSizes) {
        if (layerSizes.length < 2)
            throw new IllegalArgumentException("Invalid layers counts " + layerSizes.length);
        this.layerSizes = layerSizes;
        int layers = layers();
        weights = new Matrix[layers];
        biases = new Matrix[layers];
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            weights[layer] = new ArrayMatrix(neurons, features);
            biases[layer] = new ArrayMatrix(neurons, 1);
            // He-et-al Initialization
            // https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
            weights[layer].random().mul((float) Math.sqrt(2.0 / features));
            biases[layer].random().mul((float) Math.sqrt(2.0 / features));
        }
    }

    public Matrix weights(int layer) {
        return weights[layer];
    }

    public Matrix biases(int layer) {
        return biases[layer];
    }

    public int layers() {
        return layerSizes.length - 1;
    }

    public int features() {
        return layerSizes[0];
    }

    public int outputs() {
        return layerSizes[layerSizes.length - 1];
    }

    public int features(int layer) {
        return layerSizes[layer];
    }

    public int neurons(int layer) {
        return layerSizes[layer + 1];
    }

    public Iterable<Batch> batchIterable(BatchSource source) {
        return () -> batchIterator(source);
    }

    public Iterator<Batch> batchIterator(BatchSource source) {
        return new BatchIterator(source);
    }

    static float sigmoid(float v) {
        return (float) (1.0 / (1.0 + exp(-v)));
    }

    static float sigmoid_prime(float v) {
        float sv = sigmoid(v);
        return sv * (1.0f - sv);
    }

    static double exp(double val) {
        final long tmp = (long) (1512775 * val + (1072693248 - 60801));
        return Double.longBitsToDouble(tmp << 32);
    }

    public float evaluate(Iterator<Batch> batchIter) {
        Matrix outputs = null;
        float total = 0;
        int count = 0;
        while (batchIter.hasNext()) {
            NeuralNetwork.Batch batch = batchIter.next();
            if (outputs == null || outputs.columns() != batch.inputs.columns())
                outputs = new ArrayMatrix(outputs(), batch.inputs.columns());
            propagate(batch.inputs, outputs);
            total += batch.accuracy(outputs);
            count++;
        }
        return total / count;
    }

    public float evaluate(Batch batch) {
        Matrix outputs = propagate(batch.inputs, null);
        return batch.accuracy(outputs);
    }

    public Matrix propagate(Matrix inputs, Matrix outputs) {
        if (outputs == null)
            outputs = new ArrayMatrix(outputs(), inputs.columns());
        assert inputs.columns() == outputs.columns();
        int lastLayer = layers() - 1;
        for (int layer = 0; layer < lastLayer; layer++) {
            inputs = propagate(inputs, weights[layer], biases[layer], null);
            inputs.apply(NeuralNetwork::sigmoid);
        }
        propagate(inputs, weights[lastLayer], biases[lastLayer], outputs);
        outputs.apply(NeuralNetwork::sigmoid);
        return outputs;
    }

    public void train(int epochs, Iterable<Batch> batchesFactory, float learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            train(batchesFactory.iterator(), learningRate);
        }
    }

    public void train(Iterator<Batch> batchIter, float learningRate) {
        BatchContext context = new BatchContext();
        while (batchIter.hasNext()) {
            NeuralNetwork.Batch batch = batchIter.next();
            context.train(batch.inputs, batch.expected, learningRate);
        }
    }

    public void train(Batch batch, float learningRate) {
        new BatchContext().train(batch.inputs, batch.expected, learningRate);
    }

    public class Batch {
        public final Matrix inputs;
        public final Matrix expected;

        public Batch(int size) {
            inputs = new ArrayMatrix(features(), size);
            expected = new ArrayMatrix(outputs(), size);
        }

        public Batch(Matrix inputs, Matrix expected) {
            this.inputs = inputs;
            this.expected = expected;
        }

        public NeuralNetwork network() {
            return this.network();
        }

        public int size() {
            return inputs.columns();
        }

        public float accuracy(Matrix outputs) {
            int macthed = 0;
            for (int sample = 0; sample < expected.columns(); sample++) {
                if (expected.maxRowIndex(sample) == outputs.maxRowIndex(sample))
                    macthed++;
            }
            return macthed / (float) expected.columns();
        }
    }

    static Matrix propagate(Matrix inputs, Matrix weights, Matrix biases, Matrix outputs) {
        assert weights.columns() == inputs.rows();
        assert weights.rows() == biases.rows();
        assert outputs == null || outputs.rows() == weights.rows();
        assert outputs == null || outputs.columns() == inputs.columns();
        if (outputs == null)
            outputs = new ArrayMatrix(weights.rows(), inputs.columns());
        for (int row = 0; row < weights.rows(); row++) {
            for (int col = 0; col < inputs.columns(); col++) {
                float v = biases.get(row, 0);
                for (int ir = 0; ir < inputs.rows(); ir++) {
                    v += inputs.get(ir, col) * weights.get(row, ir);
                }
                outputs.set(row, col, v);
            }
        }
        return outputs;
    }

    class BatchContext {

        class LayerContext {
            private final int layer;
            private final Matrix z; // results of weight + bias [neurons[layer] X batchSize]
            private final Matrix activation; // sigmoid(z) [neurons[layer] X batchSize]

            private final Matrix wgrad; // weight gradients [neurons[layer] X features]
            private final Matrix bgrad; // bias gradients [neurons[layer] X 1]

            public LayerContext(int layer, int batchSize) {
                this.layer = layer;
                int neurons = neurons(layer);
                z = new ArrayMatrix(neurons, batchSize);
                activation = new ArrayMatrix(neurons, batchSize);
                wgrad = new ArrayMatrix(neurons, features(layer));
                bgrad = new ArrayMatrix(neurons, 1);
            }

            public Matrix forward(Matrix inputs) {
                propagate(inputs, weights[layer], biases[layer], z);
                Matrix.apply(z, NeuralNetwork::sigmoid, activation);
                return activation;
            }

            public Matrix backward(Matrix inputs) {
                z.apply(NeuralNetwork::sigmoid_prime);
                activation.mul(z);
                for (int r = 0; r < activation.rows(); r++) {
                    float sum = 0;
                    for (int c = 0; c < activation.columns(); c++) {
                        sum += activation.get(r, c);
                    }
                    bgrad.set(r, 0, sum);
                }
                Matrix.dot(activation, inputs.transpose(), wgrad);
                return activation;
            }

            public void updateNetwork(float learningRate) {
                int neurons = neurons(layer);
                int features = features(layer);
                int batchSize = z.columns();
                for (int neuron = 0; neuron < neurons; neuron++) {
                    float b = biases[layer].get(neuron, 0);
                    b -= (learningRate / batchSize) * bgrad.get(neuron, 0);
                    biases[layer].set(neuron, 0, b);
                    for (int feature = 0; feature < features; feature++) {
                        float w = weights[layer].get(neuron, feature);
                        w -= (learningRate / batchSize) * wgrad.get(neuron, feature);
                        weights[layer].set(neuron, feature, w);
                    }
                }
            }

        }

        private LayerContext[] layerContexts;

        private int batchSize;

        private final int layers;

        public BatchContext() {
            this.layers = layers();
        }

        private void prepare(int batchSize) {
            if (layerContexts == null || this.batchSize != batchSize) {
                layerContexts = new LayerContext[layers];
                for (int layer = 0; layer < layers; layer++) {
                    layerContexts[layer] = new LayerContext(layer, batchSize);
                }
                this.batchSize = batchSize;
            }
        }

        public void train(Matrix inputs, Matrix expected, float learningRate) {
            prepare(inputs.columns());
            Matrix activation = inputs;
            // forward propagation
            for (int layer = 0; layer < layers; layer++) {
                activation = layerContexts[layer].forward(activation);
            }

            // backward propagation
            // compute gradients
            // cost derivative = outputs - y
            Matrix.sub(activation, expected, activation);
            LayerContext layerContext = layerContexts[layers - 1];
            Matrix delta = layerContext.backward(layerContexts[layers - 2].activation);
            for (int layer = weights.length - 2; layer >= 0; layer--) {
                layerContext = layerContexts[layer];
                Matrix layerInputs = layer > 0 ? layerContexts[layer - 1].activation : inputs;
                Matrix.dot(weights[layer + 1].transpose(), delta, layerContext.activation);
                delta = layerContext.backward(layerInputs);
            }

            // apply gradients
            for (int layer = 0; layer < layers; layer++) {
                layerContexts[layer].updateNetwork(learningRate);
            }
        }
    }

    public interface BatchSource {
        int size();

        int batchSize();

        void load(int index, int size, Batch batch);
    }

    private class BatchIterator implements Iterator<Batch> {

        private final BatchSource batchSource;
        private int remaining;
        private Batch batch;

        public BatchIterator(BatchSource batchSource) {
            this.batchSource = batchSource;
            remaining = batchSource.size();
            batch = new Batch(batchSource.batchSize());
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
            if (size != batch.size())
                batch = new Batch(size);
            int startIndex = batchSource.size() - remaining;
            batchSource.load(startIndex, size, batch);
            remaining -= size;
            return batch;
        }
    }

    public static class ArrayBatchSource implements BatchSource {
        private final Matrix inputs;
        private final int[] expected;
        private final int batchSize;

        public ArrayBatchSource(float[][] inputs, int[] expected) {
            this(inputs, expected, inputs.length);
        }

        public ArrayBatchSource(float[][] inputs, int[] expected, int batchSize) {
            if (inputs.length != expected.length)
                throw new IllegalArgumentException("Size mismatch");
            this.inputs = new ArrayMatrix(inputs);
            this.expected = expected;
            this.batchSize = batchSize;
        }

        @Override
        public int size() {
            return inputs.rows();
        }

        @Override
        public int batchSize() {
            return batchSize;
        }

        @Override
        public void load(int index, int size, Batch batch) {
            for (int i = 0; i < size; i++) {
                int row = i + index;
                for (int col = 0; col < inputs.columns(); col++) {
                    batch.inputs.set(col, i, inputs.get(row, col));
                }
            }
            Labels.toExpectedMatrix(expected, index, batch.expected);
        }
    }

    public static class Labels {
        public static void toExpectedMatrix(int[] indices, int indexOffset, Matrix m) {
            assert m.columns() == indices.length;
            for (int i = indexOffset, col = 0; i < indices.length
                    && col < m.columns(); i++, col++) {
                int index = indices[i];
                for (int row = 0; row < m.rows(); row++) {
                    m.set(row, col, index == row ? 1 : 0);
                }
            }
        }

        public static void toExpectedIndex(Matrix m, int[] indices) {
            assert m.columns() == indices.length;
            for (int i = 0; i < m.columns(); i++) {
                indices[i] = m.maxRowIndex(i);
            }
        }

        public static int countMatched(int[] expected, int[] actuals) {
            assert expected.length == actuals.length;
            int matched = 0;
            for (int i = 0; i < expected.length; i++) {
                if (actuals[i] == expected[i])
                    matched++;
            }
            return matched;
        }
    }
}
