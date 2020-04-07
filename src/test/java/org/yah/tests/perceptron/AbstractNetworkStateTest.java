package org.yah.tests.perceptron;

import org.junit.Before;
import org.junit.Test;
import org.yah.tests.perceptron.base.DefaultNetworkState;

import java.util.function.DoubleSupplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public abstract class AbstractNetworkStateTest {

    public static final double DELTA = 1E-6;

    protected static double[][][] createExepectedWeights(NeuralNetworkState state) {
        double[][][] res = new double[state.layers()][][];
        int index = 1;
        for (int layer = 0; layer < state.layers(); layer++) {
            int neurons = state.neurons(layer);
            int features = state.features(layer);
            res[layer] = new double[neurons][features];
            double q = Math.sqrt(2.0 / features);
            for (int n = 0; n < neurons; n++) {
                for (int f = 0; f < features; f++) {
                    res[layer][n][f] = q * index++;
                }
            }
        }
        return res;
    }

    protected DoubleSupplier randomSource;

    @Before
    public void setup() {
        randomSource = new DoubleSupplier() {
            double next = 1;

            @Override
            public double getAsDouble() {
                return next++;
            }
        };
    }

    protected final NeuralNetworkState newState(int... layers) {
        return newState(randomSource, layers);
    }

    protected abstract NeuralNetworkState newState(DoubleSupplier randomSource, int[] layers);

    protected abstract NeuralNetworkState newState(NeuralNetworkState from);

    @Test
    public void layers() {
        assertEquals(1, newState(2, 2).layers());
        assertEquals(2, newState(8, 4, 5).layers());
        assertEquals(3, newState(3, 4, 4, 8).layers());
    }

    @Test
    public void features() {
        assertEquals(2, newState(2, 2).features());
        assertEquals(8, newState(8, 4, 5).features());
        assertEquals(3, newState(3, 4, 4, 8).features());
    }

    @Test
    public void outputs() {
        assertEquals(2, newState(2, 2).outputs());
        assertEquals(5, newState(8, 4, 5).outputs());
        assertEquals(8, newState(3, 4, 4, 8).outputs());

    }

    @Test
    public void layers_features() {
        assertEquals(2, newState(2, 2).features(0));

        assertEquals(8, newState(8, 4, 5).features(0));
        assertEquals(4, newState(8, 4, 5).features(1));

        assertEquals(3, newState(3, 4, 4, 8).features(0));
        assertEquals(4, newState(3, 4, 4, 8).features(1));
        assertEquals(4, newState(3, 4, 4, 8).features(2));
    }

    @Test
    public void neurons() {
        assertEquals(2, newState(2, 2).neurons(0));

        assertEquals(4, newState(8, 4, 5).neurons(0));
        assertEquals(5, newState(8, 4, 5).neurons(1));

        assertEquals(4, newState(3, 4, 4, 8).neurons(0));
        assertEquals(4, newState(3, 4, 4, 8).neurons(1));
        assertEquals(8, newState(3, 4, 4, 8).neurons(2));
    }

    @Test
    public void maxNeurons() {
        assertEquals(2, newState(2, 2).maxNeurons());

        assertEquals(5, newState(8, 4, 5).maxNeurons());

        assertEquals(8, newState(3, 4, 4, 8).maxNeurons());
    }

    @Test
    public void maxFeatures() {
        assertEquals(2, newState(2, 2).maxFeatures());

        assertEquals(8, newState(8, 4, 5).maxFeatures());

        assertEquals(4, newState(3, 4, 4, 8).maxFeatures());
    }

    @Test
    public void totalNeurons() {
        assertEquals(2, newState(2, 2).totalNeurons());

        assertEquals(4 + 5, newState(8, 4, 5).totalNeurons());

        assertEquals(4 + 4 + 8, newState(3, 4, 4, 8).totalNeurons());
    }

    @Test
    public void totalWeights() {
        assertEquals(2 * 2, newState(2, 2).totalWeights());

        assertEquals(4 * 8 + 5 * 4, newState(8, 4, 5).totalWeights());

        assertEquals(4 * 3 + 4 * 4 + 8 * 4, newState(3, 4, 4, 8).totalWeights());
    }

    @Test
    public void set_get_weight() {
        NeuralNetworkState state = newState(2, 3, 2);
        state.weight(0, 1, 0, 1.5);
        state.weight(0, 2, 1, 2.2);
        state.weight(1, 0, 1, 2.5);

        assertEquals(1.5, state.weight(0, 1, 0), DELTA);
        assertEquals(2.2, state.weight(0, 2, 1), DELTA);
        assertEquals(2.5, state.weight(1, 0, 1), DELTA);
    }

    @Test
    public void get_set_bias() {
        NeuralNetworkState state = newState(2, 3, 2);
        state.bias(0, 1, 1.5);
        state.bias(0, 2, 2.2);
        state.bias(1, 0, 2.5);

        assertEquals(1.5, state.bias(0, 1), DELTA);
        assertEquals(2.2, state.bias(0, 2), DELTA);
        assertEquals(2.5, state.bias(1, 0), DELTA);
    }

    @Test
    public void bias_initialization() {
        NeuralNetworkState state = newState(2, 3, 2);
        state.visitBiases((layer, neuron) -> assertEquals(0, state.bias(layer, neuron), 0));
    }

    @Test
    public void weight_initialization() {
        NeuralNetworkState state = newState(2, 3, 2);
        double[][][] expecteds = createExepectedWeights(state);
        state.visitWeights((layer, neuron, feature) ->
                assertEquals(expecteds[layer][neuron][feature], state.weight(layer, neuron, feature), DELTA));
    }

    @Test
    public void copy_constructor() {
        NeuralNetworkState state = new DefaultNetworkState(randomSource, 2, 3, 2);
        state.visitBiases((layer, neuron) -> randomSource.getAsDouble());
        state.visitWeights((layer, neuron, feature) -> randomSource.getAsDouble());

        NeuralNetworkState copy = newState(state);
        assertEquals(state.layers(), copy.layers());
        assertEquals(state.features(), copy.features());
        assertEquals(state.outputs(), copy.outputs());
        for (int l = 0; l < state.layers(); l++) {
            assertEquals(state.neurons(l), copy.neurons(l));
            assertEquals(state.features(l), copy.features(l));
        }
        state.visitWeights((layer, neuron, feature) ->
                assertEquals(state.weight(layer, neuron, feature), copy.weight(layer, neuron, feature), DELTA));
        state.visitBiases((layer, neuron) ->
                assertEquals(state.bias(layer, neuron), copy.bias(layer, neuron), DELTA));

        copy.weight(0, 0, 0, 3.14);
        assertNotEquals(3.14, state.weight(0, 0, 0));

        copy.bias(0, 0, 3.14);
        assertNotEquals(3.14, state.bias(0, 0));
    }

}