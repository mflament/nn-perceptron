package org.yah.tests.perceptron.matrix;

import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork.MatrixFactory;
import org.yah.tests.perceptron.matrix.flat.CMFlatMatrix;

public class FlatMatrixNeuralNetworkTest extends AbstractMatrixNeuralNetworkTest<CMFlatMatrix> {

    @Override
    protected MatrixFactory<CMFlatMatrix> createMatrixFactory() {
        return CMFlatMatrix::new;
    }
}
