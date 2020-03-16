package org.yah.tests.perceptron;

import static org.junit.Assert.assertEquals;

import java.util.function.Consumer;

import org.junit.Test;

public abstract class AbstractNeuralNetworkTest<N extends NeuralNetwork> {

    protected abstract void withNetwork(Consumer<N> consumer, int... layerSizes);

    @Test
    public void testLayers() {
        withNetwork(n -> assertEquals(1, n.layers()), 2, 2);
        withNetwork(n -> assertEquals(2, n.layers()), 8, 4, 5);
        withNetwork(n -> assertEquals(3, n.layers()), 3, 4, 4, 8);
    }

    @Test
    public void testFeatures() {
        withNetwork(n -> assertEquals(2, n.features()), 2, 2);
        withNetwork(n -> assertEquals(8, n.features()), 8, 4, 5);
        withNetwork(n -> assertEquals(3, n.features()), 3, 4, 4, 8);
    }

    @Test
    public void testOutputs() {
        withNetwork(n -> assertEquals(2, n.outputs()), 2, 2);
        withNetwork(n -> assertEquals(5, n.outputs()), 8, 4, 5);
        withNetwork(n -> assertEquals(8, n.outputs()), 3, 4, 4, 8);
    }

    @Test
    public void testLayerFeatures() {
        withNetwork(n -> assertEquals(2, n.features(0)), 2, 2);

        withNetwork(n -> assertEquals(8, n.features(0)), 8, 4, 5);
        withNetwork(n -> assertEquals(4, n.features(1)), 8, 4, 5);

        withNetwork(n -> assertEquals(3, n.features(0)), 3, 4, 4, 8);
        withNetwork(n -> assertEquals(4, n.features(1)), 3, 4, 4, 8);
        withNetwork(n -> assertEquals(4, n.features(2)), 3, 4, 4, 8);
    }

    @Test
    public void testNeurons() {
        withNetwork(n -> assertEquals(2, n.neurons(0)), 2, 2);

        withNetwork(n -> assertEquals(4, n.neurons(0)), 8, 4, 5);
        withNetwork(n -> assertEquals(5, n.neurons(1)), 8, 4, 5);

        withNetwork(n -> assertEquals(4, n.neurons(0)), 3, 4, 4, 8);
        withNetwork(n -> assertEquals(4, n.neurons(1)), 3, 4, 4, 8);
        withNetwork(n -> assertEquals(8, n.neurons(2)), 3, 4, 4, 8);
    }

}
