package org.yah.tests.perceptron.matrix;

import org.yah.tests.perceptron.matrix.array.RMArrayMatrix;

public class RMMatrixNeuralNetworkTest extends AbstractMatrixNeuralNetworkTest<RMArrayMatrix> {

    @Override
    protected MatrixNeuralNetwork.MatrixFactory<RMArrayMatrix> createMatrixFactory() {
        return RMArrayMatrix::new;
    }
}
