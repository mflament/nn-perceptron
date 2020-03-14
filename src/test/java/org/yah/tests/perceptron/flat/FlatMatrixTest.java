package org.yah.tests.perceptron.flat;

import org.yah.tests.perceptron.AbstractMatrixTest;
import org.yah.tests.perceptron.Matrix;

public class FlatMatrixTest extends AbstractMatrixTest {

    @Override
    protected Matrix createMatrix(double[][] values) {
        return new FlatMatrix(values);
    }

}
