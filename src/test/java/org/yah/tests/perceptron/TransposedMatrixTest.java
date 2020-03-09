package org.yah.tests.perceptron;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class TransposedMatrixTest {


    private TransposedMatrix matrix = new TransposedMatrix(new ArrayMatrix(new float[][] { { 1, 2, 3 }, { 4, 5, 6 } }));
    
    /**
     * 1 2 3  ->  1 4
     * 4 5 6      2 5
     *            3 6  
     */

    @Test
    public void testGet() {
        assertEquals(1, matrix.get(0, 0), 0);
        assertEquals(4, matrix.get(0, 1), 0);
        assertEquals(2, matrix.get(1, 0), 0);
        assertEquals(5, matrix.get(1, 1), 0);
        assertEquals(3, matrix.get(2, 0), 0);
        assertEquals(6, matrix.get(2, 1), 0);
    }

    @Test
    public void testSet() {
        matrix.set(0, 0, 6);
        matrix.set(0, 1, 3);
        matrix.set(1, 0, 5);
        matrix.set(1, 1, 2);
        matrix.set(2, 0, 4);
        matrix.set(2, 1, 1);
        
        assertEquals(6, matrix.get(0, 0), 0);
        assertEquals(3, matrix.get(0, 1), 0);
        assertEquals(5, matrix.get(1, 0), 0);
        assertEquals(2, matrix.get(1, 1), 0);
        assertEquals(4, matrix.get(2, 0), 0);
        assertEquals(1, matrix.get(2, 1), 0);

        Matrix delegate = matrix.getDelegate();
        assertEquals(6, delegate.get(0, 0), 0);
        assertEquals(5, delegate.get(0, 1), 0);
        assertEquals(4, delegate.get(0, 2), 0);
        assertEquals(3, delegate.get(1, 0), 0);
        assertEquals(2, delegate.get(1, 1), 0);
        assertEquals(1, delegate.get(1, 2), 0);
    }

    @Test
    public void testRows() {
        assertEquals(3, matrix.rows());
    }

    @Test
    public void testColumns() {
        assertEquals(2, matrix.columns());
    }
}
