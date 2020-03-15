package org.yah.tests.perceptron;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.yah.tests.perceptron.array.CMArrayMatrix;

public class NeuralNetworkTest {

    protected NeuralNetwork<CMArrayMatrix> newNetwork(int... layerSizes) {
        return new MatrixNeuralNetwork<>(CMArrayMatrix::new, layerSizes);
    }

    @Test
    public void testLayers() {
        assertEquals(1, newNetwork(2, 2).layers());
        assertEquals(2, newNetwork(8, 4, 5).layers());
        assertEquals(3, newNetwork(3, 4, 4, 8).layers());
    }

    @Test
    public void testFeatures() {
        assertEquals(2, newNetwork(2, 2).features());
        assertEquals(8, newNetwork(8, 4, 5).features());
        assertEquals(3, newNetwork(3, 4, 4, 8).features());
    }

    @Test
    public void testOutputs() {
        assertEquals(2, newNetwork(2, 2).outputs());
        assertEquals(5, newNetwork(8, 4, 5).outputs());
        assertEquals(8, newNetwork(3, 4, 4, 8).outputs());
    }

    @Test
    public void testLayerFeatures() {
        assertEquals(2, newNetwork(2, 2).features(0));

        assertEquals(8, newNetwork(8, 4, 5).features(0));
        assertEquals(4, newNetwork(8, 4, 5).features(1));

        assertEquals(3, newNetwork(3, 4, 4, 8).features(0));
        assertEquals(4, newNetwork(3, 4, 4, 8).features(1));
        assertEquals(4, newNetwork(3, 4, 4, 8).features(2));
    }

    @Test
    public void testNeurons() {
        assertEquals(2, newNetwork(2, 2).neurons(0));

        assertEquals(4, newNetwork(8, 4, 5).neurons(0));
        assertEquals(5, newNetwork(8, 4, 5).neurons(1));

        assertEquals(4, newNetwork(3, 4, 4, 8).neurons(0));
        assertEquals(4, newNetwork(3, 4, 4, 8).neurons(1));
        assertEquals(8, newNetwork(3, 4, 4, 8).neurons(2));
    }

}
