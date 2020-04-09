package org.yah.tests.perceptron;

import java.util.stream.IntStream;

public interface NeuralNetworkState {

    interface WeightVisitor {
        void visit(int layer, int neuron, int feature);
    }

    interface BiasVisitor {
        void visit(int layer, int neuron);
    }

    int layers();

    int features();

    int outputs();

    int features(int layer);

    int neurons(int layer);

    double weight(int layer, int neuron, int feature);

    void weight(int layer, int neuron, int feature, double weight);

    double bias(int layer, int neuron);

    void bias(int layer, int neuron, double bias);

    int maxNeurons();

    int maxFeatures();

    int totalNeurons();

    int totalWeights();


    default void visitWeights(int layer, WeightVisitor visitor) {
        int neurons = neurons(layer);
        int features = features(layer);
        for (int feature = 0; feature < features; feature++) {
            for (int neuron = 0; neuron < neurons; neuron++) {
                visitor.visit(layer, neuron, feature);
            }
        }
    }

    default void visitWeights(WeightVisitor visitor) {
        for (int layer = 0; layer < layers(); layer++) {
            visitWeights(layer, visitor);
        }
    }

    default void visitBiases(int layer, BiasVisitor visitor) {
        int neurons = neurons(layer);
        for (int neuron = 0; neuron < neurons; neuron++) {
            visitor.visit(layer, neuron);
        }
    }

    default void visitBiases(BiasVisitor visitor) {
        for (int layer = 0; layer < layers(); layer++) {
            visitBiases(layer, visitor);
        }
    }

    static int maxNeurons(NeuralNetworkState state) {
        return IntStream.range(0, state.layers())
                .map(state::neurons)
                .max()
                .orElseThrow(() -> new IllegalStateException("No layers"));
    }

    static int maxFeatures(NeuralNetworkState state) {
        return IntStream.range(0, state.layers())
                .map(state::features)
                .max()
                .orElseThrow(() -> new IllegalStateException("No layers"));
    }

    static int totalNeurons(NeuralNetworkState state) {
        int layers = state.layers();
        int res = 0;
        for (int layer = 0; layer < layers; layer++) {
            res += state.neurons(layer);
        }
        return res;
    }

    static int totalWeights(NeuralNetworkState state) {
        int layers = state.layers();
        int res = 0;
        for (int layer = 0; layer < layers; layer++) {
            res += state.neurons(layer) * state.features(layer);
        }
        return res;
    }

}
