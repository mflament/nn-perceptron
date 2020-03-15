package org.yah.tests.perceptron.array;

import org.yah.tests.perceptron.AbstractMatrixTest;

public class RMArrayMatrixTest extends AbstractMatrixTest<RMArrayMatrix> {

    @Override
    protected RMArrayMatrix createMatrix(double[][] values) {
        RMArrayMatrix res = createMatrix(values[0].length, values.length);
        res.apply((r, c, v) -> values[c][r]);
        return res;
    }

    @Override
    protected RMArrayMatrix createMatrix(int rows, int columns) {
        return new RMArrayMatrix(rows, columns);
    }

}
