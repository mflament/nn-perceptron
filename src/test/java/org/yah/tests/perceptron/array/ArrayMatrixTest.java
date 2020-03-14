package org.yah.tests.perceptron.array;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.yah.tests.perceptron.AbstractMatrixTest;

public class ArrayMatrixTest extends AbstractMatrixTest<ArrayMatrix> {

    // protected static final double[][] VALUES = { { 1, 4 }, { 2, 5 }, { 3, 6 } };

    @Override
    protected ArrayMatrix createMatrix(double[][] values) {
        return new ArrayMatrix(values);
    }

    @Override
    protected ArrayMatrix createMatrix(int rows, int columns) {
        return new ArrayMatrix(rows, columns);
    }

    @Test
    public void testSlide() {
        assertEquals(1, matrix.slide(0, 1));
        assertMatrix(new double[][] { { 1, 4 } }, matrix);

        assertEquals(2, matrix.slide(1, 2));
        assertMatrix(new double[][] { { 2, 5 }, { 3, 6 } }, matrix);

        assertEquals(1, matrix.slide(2, 2));
        assertMatrix(new double[][] { { 3, 6 } }, matrix);

        assertEquals(0, matrix.slide(3, 1));
        assertMatrix(new double[][] {}, matrix);
    }

    @Test
    public void testSlidingSub() {
        ArrayMatrix m2 = createMatrix(new double[][] { { 1, 3 }, { 2, 5 } });
        ArrayMatrix result = createMatrix(new double[][] { { 10, 11 } });

        matrix.slide(1, 1);
        m2.slide(0, 1);
        matrix.sub(m2, result);
        assertMatrix(new double[][] { { 1, 2 } }, result); // 2 - 1 5 - 3

        m2.slide(1, 1);
        matrix.sub(m2);
        assertMatrix(new double[][] { { 0, 0 } }, matrix);
    }

}
