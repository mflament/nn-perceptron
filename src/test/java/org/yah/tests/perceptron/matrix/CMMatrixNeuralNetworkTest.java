package org.yah.tests.perceptron.matrix;

import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;

public class CMMatrixNeuralNetworkTest extends AbstractMatrixNeuralNetworkTest<CMArrayMatrix> {
    @Override
    protected MatrixNeuralNetwork.MatrixFactory<CMArrayMatrix> createMatrixFactory() {
        return CMArrayMatrix::new;
    }
}
