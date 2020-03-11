package org.yah.tests.perceptron;

import static org.yah.tests.perceptron.Activation.sigmoid;
import static org.yah.tests.perceptron.Activation.sigmoid_prime;
import static org.yah.tests.perceptron.Matrix.matrix;

import java.util.Iterator;
import java.util.Random;

public class JavaNeuralNetwork implements NeuralNetwork {

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
    private double accuracy = Double.NaN;

    private final ThreadLocal<BatchContext> contexts = new ThreadLocal<BatchContext>() {
        @Override
        protected BatchContext initialValue() {
            return new BatchContext();
        }
    };

    public JavaNeuralNetwork(int... layerSizes) {
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
            set(weights[layer], v -> RANDOM.nextGaussian() * q);
            set(biases[layer], v -> RANDOM.nextGaussian() * q);
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

    @Override
    public double[][] weights(int layer) {
        return weights[layer];
    }

    @Override
    public double[] biases(int layer) {
        return biases[layer];
    }

    @Override
    public int layers() {
        return layerSizes.length - 1;
    }

    @Override
    public int features() {
        return layerSizes[0];
    }

    @Override
    public int outputs() {
        return layerSizes[layerSizes.length - 1];
    }

    @Override
    public int features(int layer) {
        return layerSizes[layer];
    }

    @Override
    public int neurons(int layer) {
        return layerSizes[layer + 1];
    }

    @Override
    public void propagate(double[][] inputs, int[] outputs) {
        assert outputs == null || inputs[0].length == outputs.length;
        contexts.get().propagate(inputs, outputs);
    }

    @Override
    public double train(Iterator<Batch> batchIter, double learningRate) {
        BatchContext batchContext = contexts.get();
        while(batchIter.hasNext()) {
            accuracy = batchContext.train(batchIter.next(), learningRate);
        }
        return accuracy;
    }

    @Override
    public double train(Batch batch, double learningRate) {
        return accuracy = contexts.get().train(batch, learningRate);
    }

    @Override
    public double accuracy() {
        return accuracy;
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

        public void propagate(double[][] inputs, int[] outputs) {
            prepare(inputs[0].length);
            for (int layer = 0; layer < layers; layer++) {
                inputs = layerContexts[layer].forward(inputs);
            }
            Labels.toExpectedIndex(inputs, outputs);
        }

        public double train(Batch batch, double learningRate) {
            prepare(batch.inputs[0].length);
            double[][] activation = batch.inputs;
            // forward propagation
            for (int layer = 0; layer < layers; layer++) {
                activation = layerContexts[layer].forward(activation);
            }
            double res = batch.accuracy(activation);
            // backward propagation
            // compute gradients
            // cost derivative = outputs - y
            costDerivative(activation, batch.expectedMatrix);

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
                activation = layer > 0 ? layerContexts[layer - 1].activation : batch.inputs;
                delta = layerContexts[layer].backward(activation);
            }

            // apply gradients
            for (int layer = 0; layer < layers; layer++) {
                layerContexts[layer].updateNetwork(learningRate);
            }
            return res;
        }

        private void costDerivative(double[][] actual, double[][] expected) {
            for (int r = 0; r < actual.length; r++) {
                for (int c = 0; c < actual[r].length; c++) {
                    actual[r][c] -= expected[r][c];
                }
            }
        }
    }

}
