/**
 * 
 */
package org.yah.tests.perceptron.matrix;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;

import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.RandomUtils;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tests.perceptron.matrix.MatrixSamplesSource.MatrixBatch;
import org.yah.tests.perceptron.matrix.MatrixSamplesSource.MatrixSamples;

/**
 * @author Yah
 *
 */
public class MatrixNeuralNetwork<M extends Matrix<M>> implements NeuralNetwork {

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
            weights[layer].apply((r, c, f) -> RandomUtils.nextGaussian() * q);
            //biases[layer].apply((r, c, f) -> RandomUtils.nextGaussian() * q);
        }
    }

    @Override
    public MatrixSamplesSource<M> createSampleSource() {
        return new MatrixSamplesSource<>(this);
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

    @SuppressWarnings("unchecked")
    @Override
    public void propagate(InputSamples samples, int[] outputs) {
        contexts.get().propagate((MatrixSamples<M>) samples, outputs);
    }

    @SuppressWarnings("unchecked")
    @Override
    public double evaluate(TrainingSamples samples, int[] outputs) {
        return contexts.get().evaluate((MatrixSamples<M>) samples, outputs);
    }

    @SuppressWarnings("unchecked")
    @Override
    public void train(TrainingSamples samples, double learningRate) {
        contexts.get().train((MatrixSamples<M>) samples, learningRate);
    }

    @Override
    public String toString() {
        return Arrays.toString(layerSizes);
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

        public void propagate(MatrixSamples<M> samples, int[] outputs) {
            assert outputs != null && outputs.length == samples.size();

            Iterator<MatrixBatch<M>> iterator = samples.iterator();
            int outputsIndex = 0;
            while (iterator.hasNext()) {
                MatrixBatch<M> batch = iterator.next();
                propagate(batch, outputs, outputsIndex);
                outputsIndex += batch.size();
            }
        }

        public double evaluate(MatrixSamples<M> samples, int[] outputs) {
            assert outputs == null || outputs.length == samples.size();

            Iterator<MatrixBatch<M>> iterator = samples.iterator();
            int outputsIndex = 0;
            double total = 0;
            while (iterator.hasNext()) {
                MatrixBatch<M> batch = iterator.next();
                propagate(batch, outputs, outputsIndex);
                outputsIndex += batch.size();

                M outputMatrix = forward(batch);
                total += batch.accuracy(outputMatrix, outputs);
            }
            return total / samples.batchCount();
        }

        private void propagate(MatrixBatch<M> batch, int[] outputs, int outputsIndex) {
            setBatchSize(batch.size());
            M inputs = batch.inputs();
            for (int layer = 0; layer < layers; layer++) {
                inputs = forward(layer, inputs);
            }
            if (outputs != null) {
                for (int col = 0; col < batchSize; col++) {
                    outputs[outputsIndex + col] = inputs.maxRowIndex(col);
                }
            }
        }

        public void train(MatrixSamples<M> samples, double learningRate) {
            Iterator<MatrixBatch<M>> iterator = samples.iterator();
            while (iterator.hasNext()) {
                MatrixBatch<M> batch = iterator.next();
                train(batch, learningRate);
            }
        }

        private void train(MatrixBatch<M> batch, double learningRate) {
            // forward propagation
            forward(batch);

            // backward propagation
            // compute gradients
            // cost derivative = outputs - y
            costDerivative(batch.expectedOutputs());

            for (int layer = layers - 1; layer > 0; layer--) {
                backward(layer, activations[layer - 1]);
            }
            backward(0, batch.inputs());

            // apply gradients
            for (int layer = 0; layer < layers; layer++) {
                updateNetwork(layer, learningRate);
            }
        }

        private M forward(MatrixBatch<M> batch) {
            setBatchSize(batch.size());
            M inputs = batch.inputs();
            for (int layer = 0; layer < layers; layer++) {
                inputs = forward(layer, inputs);
            }
            return inputs;
        }

        public M forward(int layer, M inputs) {
            // weight . inputs + bias
            weights[layer].dot(inputs, zs[layer]);
            zs[layer].addColumnVector(biases[layer]);
            return zs[layer].sigmoid(activations[layer]);
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

        private void costDerivative(M expected) {
            M actual = activations[layers - 1];
            actual.sub(expected);
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
