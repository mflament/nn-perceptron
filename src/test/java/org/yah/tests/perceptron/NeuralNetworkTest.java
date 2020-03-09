package org.yah.tests.perceptron;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import static org.yah.tests.perceptron.Activation.*;
public class NeuralNetworkTest {

    private static final double EPSILON = 10E-5f;

    @Test
    public void testLayers() {
        assertEquals(1, new NeuralNetwork(2, 2).layers());
        assertEquals(2, new NeuralNetwork(8, 4, 5).layers());
        assertEquals(3, new NeuralNetwork(3, 4, 4, 8).layers());
    }

    @Test
    public void testFeatures() {
        assertEquals(2, new NeuralNetwork(2, 2).features());
        assertEquals(8, new NeuralNetwork(8, 4, 5).features());
        assertEquals(3, new NeuralNetwork(3, 4, 4, 8).features());
    }

    @Test
    public void testOutputs() {
        assertEquals(2, new NeuralNetwork(2, 2).outputs());
        assertEquals(5, new NeuralNetwork(8, 4, 5).outputs());
        assertEquals(8, new NeuralNetwork(3, 4, 4, 8).outputs());
    }

    @Test
    public void testLayerFeatures() {
        assertEquals(2, new NeuralNetwork(2, 2).features(0));

        assertEquals(8, new NeuralNetwork(8, 4, 5).features(0));
        assertEquals(4, new NeuralNetwork(8, 4, 5).features(1));

        assertEquals(3, new NeuralNetwork(3, 4, 4, 8).features(0));
        assertEquals(4, new NeuralNetwork(3, 4, 4, 8).features(1));
        assertEquals(4, new NeuralNetwork(3, 4, 4, 8).features(2));
    }

    @Test
    public void testNeurons() {
        assertEquals(2, new NeuralNetwork(2, 2).neurons(0));

        assertEquals(4, new NeuralNetwork(8, 4, 5).neurons(0));
        assertEquals(5, new NeuralNetwork(8, 4, 5).neurons(1));

        assertEquals(4, new NeuralNetwork(3, 4, 4, 8).neurons(0));
        assertEquals(4, new NeuralNetwork(3, 4, 4, 8).neurons(1));
        assertEquals(8, new NeuralNetwork(3, 4, 4, 8).neurons(2));
    }

    @Test
    public void testPropagateLayer() {
        NeuralNetwork nn = new NeuralNetwork(2, 1);
        double[][] inputs = new double[][] { { 0, 0.5f }, { 0, 1f } };
        double[][] outputs = new double[1][2];
        nn.propagate(inputs, outputs);
        double bias = nn.biases(0)[0];
        double w0 = nn.weights(0)[0][0];
        double w1 = nn.weights(0)[0][1];
        assertEquals(sigmoid(bias), outputs[0][0], EPSILON);
        assertEquals(sigmoid(0.5f * w0 + w1 + bias), outputs[0][1], EPSILON);
    }

}
