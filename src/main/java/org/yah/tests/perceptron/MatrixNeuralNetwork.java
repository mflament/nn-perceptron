/**
 * 
 */
package org.yah.tests.perceptron;

import static org.yah.tests.perceptron.Activation.sigmoid;

import java.util.Iterator;
import java.util.Objects;
import java.util.Random;

/**
 * @author Yah
 *
 */
public class MatrixNeuralNetwork<M extends Matrix<M>> implements NeuralNetwork<M> {

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

    private final ThreadLocal<BatchContext> contexts = new ThreadLocal<BatchContext>() {
        @Override
        protected BatchContext initialValue() {
            return new BatchContext();
        }
    };

    private final int[] layerSizes;
    private final MatrixFactory<M> matrixFactory;
    protected final int layers;
    private final M[] weights;
    private final M[] biases;
    protected double accuracy = Double.NaN;

    @FunctionalInterface
    public interface MatrixFactory<M> {
        M newMatrix(int rows, int columns);
    }

    public MatrixNeuralNetwork(MatrixFactory<M> matrixFactory, int... layerSizes) {
        if (layerSizes.length < 2)
            throw new IllegalArgumentException("Invalid layers counts " + layerSizes.length);
        this.matrixFactory = Objects.requireNonNull(matrixFactory);
        this.layerSizes = layerSizes;
        layers = layerSizes.length - 1;
        weights = newMatrixArray(layers);
        biases = newMatrixArray(layers);
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            weights[layer] = newMatrix(neurons, features);
            biases[layer] = newMatrix(neurons, 1);
            // He-et-al Initialization
            // https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
            double q = Math.sqrt(2.0 / features);
            weights[layer].apply((r, c, f) -> RANDOM.nextGaussian() * q);
            biases[layer].apply((r, c, f) -> RANDOM.nextGaussian() * q);
        }
    }

    public BatchSource<M> createBatchSource() {
        return new MatrixBatchSource<>(this);
    }

    public M newMatrix(int rows, int columns) {
        return matrixFactory.newMatrix(rows, columns);
    }

    @SuppressWarnings("unchecked")
    public M[] newMatrixArray(int length) {
        return (M[]) new Matrix[length];
    }

    @Override
    public int layers() {
        return layers;
    }

    @Override
    public int features() {
        return layerSizes[0];
    }

    @Override
    public int outputs() {
        return layerSizes[layers];
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
    public double accuracy() {
        return accuracy;
    }

    @Override
    public void propagate(M inputs, int[] outputs) {
        contexts.get().propagate(inputs, outputs);
    }

    @Override
    public double evaluate(Batch<M> batch, int[] outputs) {
        return accuracy = contexts.get().evaluate(batch, outputs);
    }

    @Override
    public double train(Batch<M> batch, double learningRate) {
        accuracy = contexts.get().train(batch, learningRate);
        return accuracy;
    }

    @Override
    public double train(Iterator<Batch<M>> batchIter, double learningRate) {
        BatchContext context = contexts.get();
        while (batchIter.hasNext()) {
            Batch<M> batch = batchIter.next();
            accuracy = context.train(batch, learningRate);
        }
        return accuracy;
    }

    protected class BatchContext {
        private int batchSize;
        private int capacity;

        private M[] zs; // results of weight + bias [neurons[layer] X batchSize]
        private M[] activations; // sigmoid(z) [neurons[layer] X batchSize]

        private M[] wgrads; // weight gradients [neurons[layer] X features]
        private M[] bgrads; // bias gradients [neurons[layer] X 1]

        public BatchContext() {
            zs = newMatrixArray(layers);
            activations = newMatrixArray(layers);
            bgrads = newMatrixArray(layers);
            wgrads = newMatrixArray(layers);
            for (int layer = 0; layer < layers; layer++) {
                wgrads[layer] = newMatrix(neurons(layer), features(layer));
                bgrads[layer] = newMatrix(neurons(layer), 1);
            }
        }

        private void setBatchSize(int batchSize) {
            if (batchSize == this.batchSize)
                return;
            if (batchSize > capacity) {
                for (int layer = 0; layer < layers; layer++) {
                    zs[layer] = newMatrix(neurons(layer), batchSize);
                    activations[layer] = newMatrix(neurons(layer), batchSize);
                }
                capacity = batchSize;
            } else {
                for (int layer = 0; layer < layers; layer++) {
                    zs[layer].slide(0, batchSize);
                    activations[layer].slide(0, batchSize);
                }
            }
            this.batchSize = batchSize;
        }

        public void propagate(M inputs, int[] outputs) {
            int size = inputs.columns();
            setBatchSize(size);
            for (int layer = 0; layer < layers; layer++) {
                inputs = forward(layer, inputs);
            }

            for (int input = 0; input < size; input++) {
                outputs[input] = inputs.maxRowIndex(input);
            }
        }

        public double evaluate(Batch<M> batch, int[] outputs) {
            setBatchSize(batch.size());
            M inputs = batch.inputs();
            for (int layer = 0; layer < layers; layer++) {
                inputs = forward(layer, inputs);
            }
            double res = batch.accuracy(inputs);
            if (outputs != null) {
                for (int i = 0; i < outputs.length; i++) {
                    outputs[i] = inputs.maxRowIndex(i);
                }
            }
            return res;
        }

        public double train(Batch<M> batch, double learningRate) {
            setBatchSize(batch.size());
            // forward propagation
            double res = evaluate(batch, null);

            // backward propagation
            // compute gradients
            // cost derivative = outputs - y
            costDerivative(batch.expectedOutputs());

            backward(layers - 1, activations[layers - 2]);
            for (int layer = layers - 2; layer > 0; layer--) {
                backward(layer, activations[layer - 1]);
            }
            backward(0, batch.inputs());

            // apply gradients
            for (int layer = 0; layer < layers; layer++) {
                updateNetwork(layer, learningRate);
            }
            return res;
        }

        private void costDerivative(M expected) {
            M actual = activations[layers - 1];
            actual.sub(expected);
        }

        public M forward(int layer, M inputs) {
            // weight . inputs + bias
            weights[layer].dot(inputs, zs[layer]);
            zs[layer].addColumnVector(biases[layer]);
            zs[layer].sigmoid(activations[layer]);
            return activations[layer];
        }

        public void backward(int layer, M inputs) {
            M activation = activations[layer];
            M z = zs[layer];

            // activation = activation * sigmoid_prime(z)
            z.sigmoid_prime();
            activation.mul(z);

            activation.sumRows(bgrads[layer]);

            // wgrad = delta . T(inputs)
            activation.dot_transpose(inputs, wgrads[layer]);

            if (layer > 0) {
                // delta = T(W[layer]) . delta
                weights[layer].transpose_dot(activation, activations[layer - 1]);
            }
        }

        public void updateNetwork(int layer, double learningRate) {
            double lr = learningRate / batchSize;
            // w = w - (learningRate/batchSize) * wgrad
            weights[layer].sub(wgrads[layer].mul(lr));
            // b = b - (learningRate/batchSize) * bgrad
            biases[layer].sub(bgrads[layer].mul(lr));
        }
    }

}
