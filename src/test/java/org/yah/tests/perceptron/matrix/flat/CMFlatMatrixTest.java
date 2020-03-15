package org.yah.tests.perceptron.matrix.flat;

import org.yah.tests.perceptron.AbstractMatrixTest;
import org.yah.tests.perceptron.matrix.flat.CMFlatMatrix;

public class CMFlatMatrixTest extends AbstractMatrixTest<CMFlatMatrix> {

    @Override
    protected CMFlatMatrix createMatrix(int rows, int columns) {
        return new CMFlatMatrix(rows, columns);
    }

    @Override
    protected CMFlatMatrix createMatrix(double[][] values) {
        return new CMFlatMatrix(values);
    }

}
