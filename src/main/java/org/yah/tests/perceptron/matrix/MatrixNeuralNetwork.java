package org.yah.tests.perceptron.matrix;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
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

    private final int[] layerSizes;
    private final MatrixFactory<M> matrixFactory;
    protected final int layers;
    private final M[] weights;
    private final M[] biases;

    private int batchSize;
    private int capacity;

    private M[] zs; // results of weight + bias [neurons[layer] X batchSize]
    private M[] activations; // sigmoid(z) [neurons[layer] X batchSize]

    private M[] wgrads; // weight gradients [neurons[layer] X features]
    private M[] bgrads; // bias gradients [neurons[layer] X 1]

    private int[] batchOutputs;

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
        zs = newMatrixArray(layers);
        activations = newMatrixArray(layers);
        bgrads = newMatrixArray(layers);
        wgrads = newMatrixArray(layers);
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            weights[layer] = newMatrix(neurons, features);
            biases[layer] = newMatrix(neurons, 1);
            // He-et-al Initialization
            // https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
            double q = Math.sqrt(2.0 / features);
            weights[layer].apply((r, c, f) -> RandomUtils.nextGaussian() * q);
            // biases[layer].apply((r, c, f) -> RandomUtils.nextGaussian() * q);
            wgrads[layer] = newMatrix(neurons(layer), features(layer));
            bgrads[layer] = newMatrix(neurons(layer), 1);
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

    @Override
    public void snapshot(int layer, DoubleBuffer buffer) {
        int neurons = neurons(layer);
        int features = features(layer);
        for (int n = 0; n < neurons; n++) {
            for (int f = 0; f < features; f++) {
                buffer.put(weights[layer].get(n, f));
            }
        }
        for (int n = 0; n < neurons; n++) {
            buffer.put(biases[layer].get(n, 0));
        }
    }

    @Override
    public void propagate(InputSamples samples, int[] outputs) {
        propagate(samples, IntBuffer.wrap(outputs));
    }

    @Override
    public void propagate(InputSamples samples, IntBuffer outputs) {
        assert outputs != null && outputs.remaining() == samples.size();
        Iterator<MatrixBatch<M>> iterator = batchIterator(samples);
        while (iterator.hasNext()) {
            MatrixBatch<M> batch = iterator.next();
            propagate(batch);
            outputs.put(batchOutputs, 0, batch.batchSize());
        }
        outputs.flip();
    }

    @Override
    public double evaluate(TrainingSamples samples, int[] outputs) {
        return evaluate(samples, outputs != null ? IntBuffer.wrap(outputs) : null);
    }

    @Override
    public double evaluate(TrainingSamples samples, IntBuffer outputs) {
        assert outputs == null || outputs.remaining() == samples.size();
        Iterator<MatrixBatch<M>> iterator = batchIterator(samples);
        int total = 0;
        while (iterator.hasNext()) {
            MatrixBatch<M> batch = iterator.next();
            propagate(batch);
            total += batch.countMatchedOutputs(batchOutputs);
            if (outputs != null)
                outputs.put(batchOutputs, 0, batch.batchSize());
        }
        if (outputs != null)
            outputs.flip();
        return total / (double) samples.size();
    }

    @Override
    public void train(TrainingSamples samples, double learningRate) {
        Iterator<MatrixBatch<M>> iterator = batchIterator(samples);
        while (iterator.hasNext()) {
            MatrixBatch<M> batch = iterator.next();
            train(batch, learningRate);
        }
    }

    @SuppressWarnings("unchecked")
    private Iterator<MatrixBatch<M>> batchIterator(InputSamples samples) {
        return ((MatrixSamples<M>) samples).iterator();
    }

    @Override
    public String toString() {
        return Arrays.toString(layerSizes);
    }

    private void setBatchSize(int batchSize) {
        if (batchSize == this.batchSize)
            return;
        if (batchSize > capacity) {
            for (int layer = 0; layer < layers; layer++) {
                zs[layer] = newMatrix(neurons(layer), batchSize);
                activations[layer] = newMatrix(neurons(layer), batchSize);
            }
            batchOutputs = new int[batchSize];
            capacity = batchSize;
        } else {
            for (int layer = 0; layer < layers; layer++) {
                zs[layer].slide(0, batchSize);
                activations[layer].slide(0, batchSize);
            }
        }
        this.batchSize = batchSize;
    }

    private void propagate(MatrixBatch<M> batch) {
        setBatchSize(batch.batchSize());
        M inputs = batch.inputs();
        for (int layer = 0; layer < layers; layer++) {
            inputs = forward(layer, inputs);
        }
        for (int col = 0; col < batchSize; col++) {
            batchOutputs[col] = inputs.maxRowIndex(col);
        }
    }

    private void train(MatrixBatch<M> batch, double learningRate) {
        // forward propagation
        setBatchSize(batch.batchSize());
        M inputs = batch.inputs();
        for (int layer = 0; layer < layers; layer++) {
            inputs = forward(layer, inputs);
        }

        // backward propagation
        // compute gradients
        // cost derivative = actual - expected
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
            // activation[layer-1] (next inputs) = T(weight[layer]) . delta
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
