package org.yah.tests.perceptron;

import static org.junit.Assert.assertEquals;

import java.util.Random;

import org.junit.Before;
import org.junit.Test;

public abstract class AbstractMatrixTest<M extends Matrix<M>> {

    protected M matrix;

    protected abstract M createMatrix(int rows, int columns);

    protected abstract M createMatrix(double[][] values);

    protected static final double[][] VALUES = { { 1, 4 }, { 2, 5 }, { 3, 6 } };

    protected Random random = new Random(12345);

    @Before
    public void setup() {
        matrix = createMatrix(VALUES);
    }

    @Test
    public void testGet() {
        assertMatrix(VALUES, matrix);
    }

    @Test
    public void testSub() {
        M m2 = createMatrix(new double[][] { { 1, 4 }, { 2, 5 }, { 3, 6 } });
        M result = createMatrix(new double[][] { { 10, 11 }, { 12, 13 }, { 14, 15 } });
        matrix.sub(m2, result);
        assertMatrix(VALUES, matrix);
        assertMatrix(VALUES, m2);
        assertMatrix(new double[][] { { 0, 0 }, { 0, 0 }, { 0, 0 } }, result);

        matrix.sub(m2);
        assertMatrix(VALUES, m2);
        assertMatrix(new double[][] { { 0, 0 }, { 0, 0 }, { 0, 0 } }, matrix);

        m2.sub(m2);
        assertMatrix(new double[][] { { 0, 0 }, { 0, 0 }, { 0, 0 } }, m2);
    }

    @Test
    public void testMul() {
        M m2 = createMatrix(new double[][] { { 1, 4 }, { 2, 5 }, { 3, 6 } });
        M result = createMatrix(new double[][] { { 10, 11 }, { 12, 13 }, { 14, 15 } });

        matrix.mul(m2, result);
        assertMatrix(VALUES, matrix);
        assertMatrix(VALUES, m2);
        assertMatrix(new double[][] { { 1, 16 }, { 4, 25 }, { 9, 36 } }, result);

        matrix.mul(m2);
        assertMatrix(VALUES, m2);
        assertMatrix(new double[][] { { 1, 16 }, { 4, 25 }, { 9, 36 } }, matrix);

        m2.mul(m2);
        assertMatrix(new double[][] { { 1, 16 }, { 4, 25 }, { 9, 36 } }, m2);
    }

    @Test
    public void testMulScalar() {
        // M result = createMatrix();
        M result = createMatrix(new double[][] { { 10, 11 }, { 12, 13 }, { 14, 15 } });

        matrix.mul(2, result);
        assertMatrix(VALUES, matrix);
        assertMatrix(new double[][] { { 2, 8 }, { 4, 10 }, { 6, 12 } }, result);

        matrix.mul(2);
        assertMatrix(new double[][] { { 2, 8 }, { 4, 10 }, { 6, 12 } }, matrix);
    }

    @Test
    public void testDot() {
        M m2 = createMatrix(new double[][] { { 7, 9, 11 }, { 8, 10, 12 } });
        M res = createMatrix(new double[][] { { 50, 55 }, { 80, 90 } });
        matrix.dot(m2, res);
        assertMatrix(new double[][] { { 58, 139 }, { 64, 154 } }, res);

        M res2 = createMatrix(new double[][] { { 50, 55 }, { 80, 90 } });
        assertMatrix(res, transpose(m2).dot(transpose(matrix), res2));
    }

    @Test
    public void testTransposeDot() {
        M m1 = createMatrix(2, 4);
        M m2 = createMatrix(m1.columns(), 2);
        for (int i = 0; i < 1; i++) {
            randomMatrix(m1);
            randomMatrix(m2);
            assertMatrix(transpose(m1).transpose_dot(m2), m1.dot(m2));
        }
    }

    @Test
    public void testDotTranspose() {
        M m1 = createMatrix(2, 4);
        M m2 = createMatrix(m1.columns(), 2);
        for (int i = 0; i < 1; i++) {
            randomMatrix(m1);
            randomMatrix(m2);
            assertMatrix(m1.dot_transpose(transpose(m2)), m1.dot(m2));
        }
    }

    protected M randomMatrix(M matrix) {
        matrix.apply((r, c, v) -> random.nextGaussian());
        return matrix;
    }

    protected M transpose(M matrix) {
        M tm = createMatrix(matrix.columns(), matrix.rows());
        tm.apply((r, c, v) -> matrix.get(c, r));
        return tm;
    }

    protected void assertMatrix(double[][] expected, M actual) {
        for (int col = 0; col < expected.length; col++) {
            for (int row = 0; row < expected[col].length; row++) {
                assertEquals(expected[col][row], actual.get(row, col), 0);
            }
        }
    }

    protected void assertMatrix(M expected, M actual) {
        assertEquals(expected.rows(), actual.rows());
        assertEquals(expected.columns(), actual.columns());
        for (int c = 0; c < expected.columns(); c++) {
            for (int r = 0; r < expected.rows(); r++) {
                assertEquals(expected.get(r, c), expected.get(r, c), 0);
            }
        }
    }
}