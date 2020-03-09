package org.yah.tests.perceptron;

import static org.junit.Assert.*;

import org.junit.Test;

public class ArrayMatrixTest {

    private Matrix matrix = new ArrayMatrix(new float[][] { { 1, 2, 3 }, { 4, 5, 6 } });

    @Test
    public void testGet() {
        assertEquals(1, matrix.get(0, 0), 0);
        assertEquals(2, matrix.get(0, 1), 0);
        assertEquals(3, matrix.get(0, 2), 0);
        assertEquals(4, matrix.get(1, 0), 0);
        assertEquals(5, matrix.get(1, 1), 0);
        assertEquals(6, matrix.get(1, 2), 0);
    }

    @Test
    public void testSet() {
        matrix.set(0, 0, 6);
        matrix.set(0, 1, 5);
        matrix.set(0, 2, 4);
        matrix.set(1, 0, 3);
        matrix.set(1, 1, 2);
        matrix.set(1, 2, 1);
        assertEquals(6, matrix.get(0, 0), 0);
        assertEquals(5, matrix.get(0, 1), 0);
        assertEquals(4, matrix.get(0, 2), 0);
        assertEquals(3, matrix.get(1, 0), 0);
        assertEquals(2, matrix.get(1, 1), 0);
        assertEquals(1, matrix.get(1, 2), 0);
    }

    @Test
    public void testRows() {
        assertEquals(2, matrix.rows());
    }

    @Test
    public void testColumns() {
        assertEquals(3, matrix.columns());
    }

}
