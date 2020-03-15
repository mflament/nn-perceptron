package org.yah.tests.perceptron.array;

import org.yah.tests.perceptron.AbstractMatrixTest;

public class CMArrayMatrixTest extends AbstractMatrixTest<CMArrayMatrix> {

    @Override
    protected CMArrayMatrix createMatrix(double[][] values) {
        return new CMArrayMatrix(values);
    }

    @Override
    protected CMArrayMatrix createMatrix(int rows, int columns) {
        return new CMArrayMatrix(rows, columns);
    }

}
