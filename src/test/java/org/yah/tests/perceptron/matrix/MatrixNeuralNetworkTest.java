/**
 * 
 */
package org.yah.tests.perceptron.matrix;

import java.util.function.Consumer;

import org.yah.tests.perceptron.AbstractNeuralNetworkTest;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;

/**
 * @author Yah
 *
 */
public class MatrixNeuralNetworkTest extends AbstractNeuralNetworkTest<MatrixNeuralNetwork<?>> {

    @Override
    protected void withNetwork(Consumer<MatrixNeuralNetwork<?>> consumer, int... layerSizes) {
        consumer.accept(new MatrixNeuralNetwork<>(CMArrayMatrix::new, layerSizes));
    }

}
