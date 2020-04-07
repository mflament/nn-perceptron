package org.yah.tests.perceptron.mt;

import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.NeuralNetworkState;
import org.yah.tests.perceptron.base.AbstractNeuralNetworkTest;
import org.yah.tests.perceptron.jni.NativeNeuralNetwork;

/**
 * @author Yah
 */
public class MTNeuralNetworkTest extends AbstractNeuralNetworkTest {

    @Override
    protected NeuralNetwork newNetwork(NeuralNetworkState state) {
        return new MTNeuralNetwork(state);
    }

    @Override
    protected void updateState(NeuralNetworkState network) {
        ((MTNeuralNetwork)network).updateState();
    }

    @Override
    protected void updateModel(NeuralNetworkState network) {
        ((MTNeuralNetwork)network).updateModel();
    }

}
