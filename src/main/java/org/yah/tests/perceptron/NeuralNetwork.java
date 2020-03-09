package org.yah.tests.perceptron;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;

public class NeuralNetwork {

    public static final Random RANDOM = createRandom();

    private static Random createRandom() {
        long seed = seed();
        return seed < 0 ? new Random() : new Random(seed);
    }

    public static long seed() {
        long seed = -1;
        String prop = System.getProperty("seed");
        if (prop != null) {
            try {
                seed = Long.parseLong(prop);
            } catch (NumberFormatException e) {}
        }
        return seed;
    }
    
    @FunctionalInterface
    private interface MatrixFunction {
        float apply(float f);
    }
    
    private final int[] layerSizes;

    private final float[][][] weights;
    private final float[][] biases;

    public NeuralNetwork(int... layerSizes) {
        if (layerSizes.length < 2)
            throw new IllegalArgumentException("Invalid layers counts " + layerSizes.length);
        this.layerSizes = layerSizes;
        int layers = layers();
        weights = new float[layers][][];
        biases = new float[layers][];
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            weights[layer] = new float[neurons][features];
            biases[layer] = new float[neurons];
            // He-et-al Initialization
            // https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
            float q = (float) Math.sqrt(2.0 / features);
            set(weights[layer], v -> (float) RANDOM.nextGaussian() * q);
            set(biases[layer], v -> (float) RANDOM.nextGaussian() * q);
        }
    }

    private void set(float[][] m, MatrixFunction f) {
        for (int row = 0; row < m.length; row++) {
            set(m[row], f);
        }
    }

    private void set(float[] v, MatrixFunction f) {
        for (int i = 0; i < v.length; i++) {
            v[i] = f.apply(v[i]);
        }
    }

    public float[][] weights(int layer) {
        return weights[layer];
    }

    public float[] biases(int layer) {
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

    public float evaluate(Iterator<Batch> batchIter) {
        float[][] outputs = null;
        float total = 0;
        int count = 0;
        while (batchIter.hasNext()) {
            NeuralNetwork.Batch batch = batchIter.next();
            if (outputs != null && outputs[0].length != batch.size())
                outputs = null;
            outputs = propagate(batch.inputs, outputs);
            total += batch.accuracy(outputs);
            count++;
        }
        return total / count;
    }

    public float evaluate(Batch batch) {
        float[][] outputs = propagate(batch.inputs, null);
        return batch.accuracy(outputs);
    }

    public float[][] propagate(float[][] inputs, float[][] outputs) {
        assert outputs == null || inputs[0].length == outputs[0].length;
        int lastLayer = layers() - 1;
        for (int layer = 0; layer < lastLayer; layer++) {
            inputs = propagate(inputs, weights[layer], biases[layer], null);
            sigmoid(inputs);
        }
        outputs = propagate(inputs, weights[lastLayer], biases[lastLayer], outputs);
        sigmoid(outputs);
        return outputs;
    }

    static float[][] propagate(float[][] inputs, float[][] weights, float[] biases,
            float[][] outputs) {
        assert weights[0].length == inputs.length;
        assert weights.length == biases.length;
        assert outputs == null || outputs.length == weights.length;
        assert outputs == null || outputs[0].length == inputs[0].length;
        if (outputs == null)
            outputs = new float[weights.length][inputs[0].length];
        for (int row = 0; row < weights.length; row++) {
            for (int col = 0; col < inputs[0].length; col++) {
                float v = biases[row];
                for (int ir = 0; ir < inputs.length; ir++) {
                    v += inputs[ir][col] * weights[row][ir];
                }
                outputs[row][col] = v;
            }
        }
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
        public final float[][] inputs;
        public final float[][] expected;

        public Batch(int size) {
            inputs = new float[features()][size];
            expected = new float[outputs()][size];
        }

        public Batch(float[][] inputs, float[][] expected) {
            this.inputs = inputs;
            this.expected = expected;
        }

        public NeuralNetwork network() {
            return this.network();
        }

        public int size() {
            return inputs[0].length;
        }

        public float accuracy(float[][] outputs) {
            int size = size();
            int macthed = 0;
            for (int sample = 0; sample < size; sample++) {
                if (maxRowIndex(expected, sample) == maxRowIndex(outputs, sample))
                    macthed++;
            }
            return macthed / (float) size;
        }
    }

    class BatchContext {

        class LayerContext {
            private final int layer;
            private final float[][] z; // results of weight + bias [neurons[layer] X batchSize]
            private final float[][] activation; // sigmoid(z) [neurons[layer] X batchSize]

            private final float[][] wgrad; // weight gradients [neurons[layer] X features]
            private final float[] bgrad; // bias gradients [neurons[layer] X 1]

            public LayerContext(int layer, int batchSize) {
                this.layer = layer;
                int neurons = neurons(layer);
                z = new float[neurons][batchSize];
                activation = new float[neurons][batchSize];
                wgrad = new float[neurons][features(layer)];
                bgrad = new float[neurons];
            }

            public float[][] forward(float[][] inputs) {
                propagate(inputs, weights[layer], biases[layer], z);
                sigmoid(z, activation);
                return activation;
            }

            public float[][] backward(float[][] inputs) {
                for (int r = 0; r < activation.length; r++) {
                    bgrad[r] = 0;
                    for (int c = 0; c < activation[r].length; c++) {
                        activation[r][c] *= sigmoid_prime(z[r][c]);
                        bgrad[r] += activation[r][c];
                    }
                }

                zero(wgrad);
                // delta . T(inputs)
                for (int r = 0; r < activation.length; r++) {
                    for (int c = 0; c < activation[r].length; c++) {
                        for (int ir = 0; ir < inputs.length; ir++) {
                            wgrad[r][ir] += activation[r][c] * inputs[ir][c];
                        }
                    }
                }
                return activation;
            }

            public void updateNetwork(float learningRate) {
                int neurons = neurons(layer);
                int features = features(layer);
                int batchSize = z[0].length;
                float lr = learningRate / batchSize;
                for (int neuron = 0; neuron < neurons; neuron++) {
                    biases[layer][neuron] -= lr * bgrad[neuron];
                    for (int feature = 0; feature < features; feature++) {
                        weights[layer][neuron][feature] -= lr * wgrad[neuron][feature];
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

        public void train(float[][] inputs, float[][] expected, float learningRate) {
            prepare(inputs[0].length);
            float[][] activation = inputs;
            // forward propagation
            for (int layer = 0; layer < layers; layer++) {
                activation = layerContexts[layer].forward(activation);
            }

            // backward propagation
            // compute gradients
            // cost derivative = outputs - y
            costDerivative(activation, expected);

            activation = layerContexts[layers - 2].activation;
            float[][] delta = layerContexts[layers - 1].backward(activation);
            for (int layer = weights.length - 2; layer >= 0; layer--) {
                // T(W[layer+1]) . delta
                zero(activation);
                float[][] m = weights[layer + 1];
                for (int c = 0; c < m[0].length; c++) {
                    for (int r = 0; r < m.length; r++) {
                        for (int dc = 0; dc < delta[r].length; dc++) {
                            activation[c][dc] += m[r][c] * delta[r][dc];
                        }
                    }
                }
                activation = layer > 0 ? layerContexts[layer - 1].activation : inputs;
                delta = layerContexts[layer].backward(activation);
            }

            // apply gradients
            for (int layer = 0; layer < layers; layer++) {
                layerContexts[layer].updateNetwork(learningRate);
            }
        }

        private void costDerivative(float[][] actual, float[][] expected) {
            for (int r = 0; r < actual.length; r++) {
                for (int c = 0; c < actual[r].length; c++) {
                    actual[r][c] -= expected[r][c];
                }
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
        private final float[][] inputs;
        private final int[] expected;
        private final int batchSize;

        public ArrayBatchSource(float[][] inputs, int[] expected) {
            this(inputs, expected, inputs.length);
        }

        public ArrayBatchSource(float[][] inputs, int[] expected, int batchSize) {
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
            for (int i = 0; i < size; i++) {
                int row = i + index;
                for (int col = 0; col < inputs[0].length; col++) {
                    batch.inputs[col][i] = inputs[row][col];
                }
            }
            Labels.toExpectedMatrix(expected, index, batch.expected);
        }
    }

    public static class Labels {
        public static void toExpectedMatrix(int[] indices, int indexOffset, float[][] m) {
            for (int i = indexOffset, col = 0; i < indices.length
                    && col < m[0].length; i++, col++) {
                int index = indices[i];
                for (int row = 0; row < m.length; row++) {
                    m[row][col] = index == row ? 1 : 0;
                }
            }
        }

        public static void toExpectedIndex(float[][] m, int[] indices) {
            for (int i = 0; i < m[0].length; i++) {
                indices[i] = maxRowIndex(m, i);
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

    public static int maxRowIndex(float[][] m, int col) {
        int res = -1;
        float max = Float.MIN_VALUE;
        for (int row = 0; row < m.length; row++) {
            float v = m[row][col];
            if (v > max) {
                max = v;
                res = row;
            }
        }
        return res;
    }

    static void sigmoid(float[][] in) {
        sigmoid(in, in);
    }

    static void sigmoid(float[][] in, float[][] out) {
        for (int r = 0; r < in.length; r++) {
            for (int c = 0; c < in[r].length; c++) {
                out[r][c] = sigmoid(in[r][c]);
            }
        }
    }

    static void zero(float[][] m) {
        for (int r = 0; r < m.length; r++) {
            for (int c = 0; c < m[r].length; c++) {
                m[r][c] = 0f;
            }
        }
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
}
