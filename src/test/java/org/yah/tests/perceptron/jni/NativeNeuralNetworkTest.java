package org.yah.tests.perceptron.jni;

import java.util.function.Consumer;

import org.yah.tests.perceptron.AbstractNeuralNetworkTest;

/**
 * @author Yah
 *
 */
public class NativeNeuralNetworkTest extends AbstractNeuralNetworkTest<NativeNeuralNetwork> {


    @Override
    protected void withNetwork(Consumer<NativeNeuralNetwork> consumer, int... layerSizes) {
        try (NativeNeuralNetwork network = new NativeNeuralNetwork(layerSizes)) {
            consumer.accept(network);
        }
    }

}
