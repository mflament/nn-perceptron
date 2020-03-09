package org.yah.tests.perceptron;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;

import static org.yah.tests.perceptron.Activation.*;
import static org.yah.tests.perceptron.Matrix.*;

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
        double apply(double f);
    }

    private final int[] layerSizes;

    private final double[][][] weights;
    private final double[][] biases;

    private final ThreadLocal<BatchContext> contexts = new ThreadLocal<BatchContext>() {
        protected BatchContext initialValue() {
            return new BatchContext();
        }
    };

    public NeuralNetwork(int... layerSizes) {
        if (layerSizes.length < 2)
            throw new IllegalArgumentException("Invalid layers counts " + layerSizes.length);
        this.layerSizes = layerSizes;
        int layers = layers();
        weights = new double[layers][][];
        biases = new double[layers][];
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            weights[layer] = matrix(neurons, features);
            biases[layer] = new double[neurons];
            // He-et-al Initialization
            // https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
            double q = Math.sqrt(2.0 / features);
            set(weights[layer], v -> (double) RANDOM.nextGaussian() * q);
            set(biases[layer], v -> (double) RANDOM.nextGaussian() * q);
        }
    }

    private void set(double[][] m, MatrixFunction f) {
        for (int row = 0; row < m.length; row++) {
            set(m[row], f);
        }
    }

    private void set(double[] v, MatrixFunction f) {
        for (int i = 0; i < v.length; i++) {
            v[i] = f.apply(v[i]);
        }
    }

    public double[][] weights(int layer) {
        return weights[layer];
    }

    public double[] biases(int layer) {
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

    public double evaluate(Iterator<Batch> batchIter) {
        double[][] outputs = null;
        double total = 0;
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

    public double evaluate(Batch batch) {
        double[][] outputs = propagate(batch.inputs, null);
        return batch.accuracy(outputs);
    }

    public double[][] propagate(double[][] inputs, double[][] outputs) {
        assert outputs == null || inputs[0].length == outputs[0].length;
        return contexts.get().propagate(inputs, outputs);
    }

    public void train(int epochs, Iterable<Batch> batchesFactory, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            train(batchesFactory.iterator(), learningRate);
        }
    }

    public void train(int epochs, Batch batch, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            train(batch, learningRate);
        }
    }

    public void train(Iterator<Batch> batchIter, double learningRate) {
        BatchContext context = contexts.get();
        while (batchIter.hasNext()) {
            NeuralNetwork.Batch batch = batchIter.next();
            context.train(batch.inputs, batch.expectedMatrix, learningRate);
        }
    }

    public void train(Batch batch, double learningRate) {
        contexts.get().train(batch.inputs, batch.expectedMatrix, learningRate);
    }

    public Batch newBatch(int size) {
        return new Batch(matrix(features(), size), matrix(outputs(), size));
    }

    public static class Batch {
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
            return Matrix.accuracy(outputs, expectedIndices);
        }
    }

    class BatchContext {

        class LayerContext {
            private final int layer;
            private final double[][] z; // results of weight + bias [neurons[layer] X batchSize]
            private final double[][] activation; // sigmoid(z) [neurons[layer] X batchSize]

            private final double[][] wgrad; // weight gradients [neurons[layer] X features]
            private final double[] bgrad; // bias gradients [neurons[layer] X 1]

            public LayerContext(int layer, int batchSize) {
                this.layer = layer;
                int neurons = neurons(layer);
                z = matrix(neurons, batchSize);
                activation = matrix(neurons, batchSize);
                wgrad = matrix(neurons, features(layer));
                bgrad = new double[neurons];
            }

            public double[][] forward(double[][] inputs) {
                return forward(inputs, activation);
            }

            public double[][] forward(double[][] inputs, double[][] outputs) {
                propagate(inputs);
                if (outputs == null)
                    outputs = matrix(outputs(), inputs[0].length);
                sigmoid(z, outputs);
                return outputs;
            }
            
            private void propagate(double[][] inputs) {
                int neurons = neurons(layer);
                int samples = inputs[0].length;
                for (int row = 0; row < neurons; row++) {
                    for (int col = 0; col < samples; col++) {
                        z[row][col] = 0;
                        for (int ir = 0; ir < inputs.length; ir++) {
                            z[row][col] += weights[layer][row][ir] * inputs[ir][col];
                        }
                        z[row][col] += biases[layer][row];
                    }
                }
            }

            public double[][] backward(double[][] inputs) {
                for (int r = 0; r < activation.length; r++) {
                    bgrad[r] = 0;
                    for (int c = 0; c < activation[r].length; c++) {
                        activation[r][c] *= sigmoid_prime(z[r][c]);
                        bgrad[r] += activation[r][c];
                    }
                }

                Matrix.zero(wgrad);
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

            public void updateNetwork(double learningRate) {
                int neurons = neurons(layer);
                int features = features(layer);
                int batchSize = z[0].length;
                double lr = learningRate / batchSize;
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

        public double[][] propagate(double[][] inputs, double[][] outputs) {
            prepare(inputs[0].length);
            int lastLayer = layers() - 1;
            for (int layer = 0; layer < lastLayer; layer++) {
                inputs = layerContexts[layer].forward(inputs);
            }
            return layerContexts[lastLayer].forward(inputs, outputs);
        }

        public void train(double[][] inputs, double[][] expected, double learningRate) {
            prepare(inputs[0].length);
            double[][] activation = inputs;
            // forward propagation
            for (int layer = 0; layer < layers; layer++) {
                activation = layerContexts[layer].forward(activation);
            }

            // backward propagation
            // compute gradients
            // cost derivative = outputs - y
            costDerivative(activation, expected);

            activation = layerContexts[layers - 2].activation;
            double[][] delta = layerContexts[layers - 1].backward(activation);
            for (int layer = weights.length - 2; layer >= 0; layer--) {
                // T(W[layer+1]) . delta
                Matrix.zero(activation);
                double[][] m = weights[layer + 1];
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

        private void costDerivative(double[][] actual, double[][] expected) {
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
            batch = newBatch(batchSource.batchSize());
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
                batch = newBatch(size);
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
