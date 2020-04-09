package org.yah.tests.perceptron.matrix;

import org.yah.tests.perceptron.NeuralNetworkState;
import org.yah.tests.perceptron.base.AbstractBatchedNeuralNetwork;
import org.yah.tests.perceptron.base.ArrayNetworkOutputs;
import org.yah.tests.perceptron.base.SamplesSource;

import java.util.Locale;


/**
 * @author Yah
 */
public final class MatrixNeuralNetwork<M extends Matrix<M>>
        extends AbstractBatchedNeuralNetwork<MatrixBatch<M>, ArrayNetworkOutputs> {

    private final MatrixFactory<M> matrixFactory;

    private int batchSize;
    private int capacity;

    private M[] weightMatrices;
    private M[] biasesMatrices;

    private M[] zs; // results of weight + bias [neurons[layer] X batchSize]
    private M[] activations; // sigmoid(z) [neurons[layer] X batchSize]

    private M[] wgrads; // weight gradients [neurons[layer] X features]
    private M[] bgrads; // bias gradients [neurons[layer] X 1]

    @FunctionalInterface
    public interface MatrixFactory<M> {
        M newMatrix(int rows, int columns);
    }

    public MatrixNeuralNetwork(MatrixFactory<M> matrixFactory, NeuralNetworkState state) {
        super(state);
        int layers = layers();
        this.matrixFactory = matrixFactory;
        weightMatrices = newMatrixArray(layers);
        biasesMatrices = newMatrixArray(layers);
        zs = newMatrixArray(layers);
        activations = newMatrixArray(layers);
        bgrads = newMatrixArray(layers);
        wgrads = newMatrixArray(layers);
        for (int l = 0; l < layers; l++) {
            final int layer = l;
            int neurons = neurons(layer);
            int features = features(layer);
            weightMatrices[layer] = newMatrix(neurons, features);
            biasesMatrices[layer] = newMatrix(neurons, 1);
            weightMatrices[layer].apply((row, column, value) -> weight(layer, row, column));
            biasesMatrices[layer].apply((row, column, value) -> bias(layer, row));
            wgrads[layer] = newMatrix(neurons, features);
            bgrads[layer] = newMatrix(neurons, 1);
        }
    }

    @Override
    public ArrayNetworkOutputs createOutpus(int samples) {
        return new ArrayNetworkOutputs(samples);
    }

    @Override
    protected SamplesSource<MatrixBatch<M>> createSampleSource() {
        return new MatrixSamplesSource<>(this);
    }

    @Override
    protected void updateState() {
        visitWeights((layer, neuron, feature) -> weight(layer, neuron, feature,
                weightMatrices[layer].get(neuron, feature)));
        visitBiases((layer, neuron) -> bias(layer, neuron, biasesMatrices[layer].get(neuron, 0)));
    }

    @Override
    protected void updateModel() {
        visitWeights((layer, neuron, feature) -> weightMatrices[layer].set(neuron, feature, weight(layer, neuron, feature)));
        visitBiases((layer, neuron) -> biasesMatrices[layer].set(neuron, 0, bias(layer, neuron)));
    }

    @Override
    protected void propagate(MatrixBatch<M> batch, ArrayNetworkOutputs outputs) {
        M outputsMatrix = forward(batch);
        for (int sample = 0; sample < batch.size(); sample++) {
            outputs.push(outputsMatrix.maxRowIndex(sample));
        }
    }

    @Override
    protected int evaluate(MatrixBatch<M> batch, ArrayNetworkOutputs outputs) {
        int matched = 0;
        M outputsMatrix = forward(batch);
        for (int sample = 0; sample < batch.size(); sample++) {
            int expected = batch.expectedIndex(sample);
            int actual = outputsMatrix.maxRowIndex(sample);
            if (outputs != null)
                outputs.push(actual);
            if (expected == actual) matched++;
        }
        return matched;
    }

    protected void train(MatrixBatch<M> batch, double learningRate) {
        // forward propagation
        setBatchSize(batch.size());
        M inputs = batch.inputs;

        for (int layer = 0; layer < layers(); layer++) {
            inputs = forward(layer, inputs);
        }

        // compute gradients
        // cost derivative = actual - expected
        activations[layers() - 1].apply((index, sample, value) -> value - (batch.expectedIndex(sample) == index ? 1 : 0));

        // backward propagation
        for (int layer = layers() - 1; layer > 0; layer--) {
            backward(layer, activations[layer - 1]);
        }
        backward(0, batch.inputs);

        // apply gradients
        for (int layer = 0; layer < layers(); layer++) {
            updateNetwork(layer, learningRate);
        }
    }

    private void setBatchSize(int batchSize) {
        if (batchSize == this.batchSize)
            return;
        if (batchSize > capacity) {
            for (int layer = 0; layer < layers(); layer++) {
                zs[layer] = newMatrix(neurons(layer), batchSize);
                activations[layer] = newMatrix(neurons(layer), batchSize);
            }
            capacity = batchSize;
        } else {
            for (int layer = 0; layer < layers(); layer++) {
                zs[layer].slide(0, batchSize);
                activations[layer].slide(0, batchSize);
            }
        }
        this.batchSize = batchSize;
    }

    private M forward(MatrixBatch<M> batch) {
        setBatchSize(batch.size());
        M inputs = batch.inputs;
        for (int layer = 0; layer < layers(); layer++) {
            inputs = forward(layer, inputs);
        }
        return inputs;
    }

    private M forward(int layer, M inputs) {
        // weight . inputs + bias
        weightMatrices[layer].dot(inputs, zs[layer]);
        zs[layer].addColumnVector(biasesMatrices[layer]);
        return zs[layer].sigmoid(activations[layer]);
    }


    private void backward(int layer, M inputs) {
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
            weightMatrices[layer].transpose_dot(activation, activations[layer - 1]);
        }
    }

    private void updateNetwork(int layer, double learningRate) {
        double lr = learningRate / batchSize;
        // w = w - (learningRate/batchSize) * wgrad
        weightMatrices[layer].sub(wgrads[layer].mul(lr));
        // b = b - (learningRate/batchSize) * bgrad
        biasesMatrices[layer].sub(bgrads[layer].mul(lr));
    }

    @SuppressWarnings("unchecked")
    private M[] newMatrixArray(int length) {
        return (M[]) new Matrix[length];
    }

    M newMatrix(int rows, int columns) {
        return matrixFactory.newMatrix(rows, columns);
    }

}
