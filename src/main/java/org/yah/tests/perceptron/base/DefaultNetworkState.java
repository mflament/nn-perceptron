package org.yah.tests.perceptron.base;

import org.yah.tests.perceptron.NeuralNetworkState;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.DoubleSupplier;

import static java.util.Objects.requireNonNull;

public final class DefaultNetworkState implements NeuralNetworkState {

    protected final int layers;

    protected final int maxNeurons;
    protected final int maxFeatures;
    protected final int totalNeurons;
    protected final int totalWeights;

    private final int[] layerSizes;

    private final double[][][] weights;
    private final double[][] biases;

    public DefaultNetworkState(NeuralNetworkState from) {
        requireNonNull(from,"from");
        layers = from.layers();
        layerSizes = new int[layers + 1];
        layerSizes[0] = from.features();
        for (int layer = 0; layer < layers; layer++) {
            layerSizes[layer + 1] = from.neurons(layer);
        }
        maxNeurons = NeuralNetworkState.maxNeurons(this);
        maxFeatures = NeuralNetworkState.maxFeatures(this);
        totalNeurons = NeuralNetworkState.totalNeurons(this);
        totalWeights = NeuralNetworkState.totalWeights(this);

        weights = new double[layers][][];
        biases = new double[layers][];
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            weights[layer] = new double[neurons][features];
            biases[layer] = new double[neurons];
            for (int neuron = 0; neuron < neurons; neuron++) {
                for (int feature = 0; feature < features; feature++) {
                    weights[layer][neuron][feature] = from.weight(layer, neuron, feature);
                }
                biases[layer][neuron] = from.bias(layer, neuron);
            }
        }
    }

    public DefaultNetworkState(int... layerSizes) {
        this(null, layerSizes);
    }

    public DefaultNetworkState(DoubleSupplier randomSource, int... layerSizes) {
        if (layerSizes.length < 2)
            throw new IllegalArgumentException("Invalid layers counts " + layerSizes.length);
        this.layerSizes = layerSizes;
        layers = layerSizes.length - 1;
        maxNeurons = NeuralNetworkState.maxNeurons(this);
        maxFeatures = NeuralNetworkState.maxFeatures(this);
        totalNeurons = NeuralNetworkState.totalNeurons(this);
        totalWeights = NeuralNetworkState.totalWeights(this);
        weights = new double[layers][][];
        biases = new double[layers][];
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            weights[layer] = new double[neurons][features];
            biases[layer] = new double[neurons];
            if (randomSource != null) {
                // He-et-al Initialization
                // https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
                double q = Math.sqrt(2.0 / features);
                randomize(weights[layer], randomSource, q);
            }
        }
    }

    private static void randomize(double[][] weights, DoubleSupplier randomSource, double q) {
        int features = weights[0].length;
        for (int neuron = 0; neuron < weights.length; neuron++) {
            for (int feature = 0; feature < features; feature++) {
                weights[neuron][feature] = randomSource.getAsDouble() * q;
            }
        }
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
    public int maxNeurons() {
        return maxNeurons;
    }

    @Override
    public int maxFeatures() {
        return maxFeatures;
    }

    @Override
    public int totalNeurons() {
        return totalNeurons;
    }

    @Override
    public int totalWeights() {
        return totalWeights;
    }

    @Override
    public double weight(int layer, int neuron, int feature) {
        return weights[layer][neuron][feature];
    }

    @Override
    public void weight(int layer, int neuron, int feature, double weight) {
        weights[layer][neuron][feature] = weight;
    }

    @Override
    public double bias(int layer, int neuron) {
        return biases[layer][neuron];
    }

    @Override
    public void bias(int layer, int neuron, double bias) {
        biases[layer][neuron] = bias;
    }

    @Override
    public String toString() {
        return Arrays.toString(layerSizes);
    }


}
