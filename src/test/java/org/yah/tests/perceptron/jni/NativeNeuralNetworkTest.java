package org.yah.tests.perceptron.jni;

import java.util.function.Consumer;

import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.NeuralNetworkState;
import org.yah.tests.perceptron.base.AbstractNeuralNetworkTest;
import org.yah.tests.perceptron.base.DefaultNetworkState;

import javax.management.JMX;
import javax.management.MBeanServer;

/**
 * @author Yah
 */
public class NativeNeuralNetworkTest extends AbstractNeuralNetworkTest {

    static {
        System.out.println("PID: " + ProcessHandle.current().pid());
    }
    @Override
    protected NeuralNetwork newNetwork(NeuralNetworkState state) {
        return new NativeNeuralNetwork(state);
    }

    @Override
    protected void updateState(NeuralNetworkState network) {
        ((NativeNeuralNetwork)network).updateState();
    }

    @Override
    protected void updateModel(NeuralNetworkState network) {
        ((NativeNeuralNetwork)network).updateModel();
    }
}
