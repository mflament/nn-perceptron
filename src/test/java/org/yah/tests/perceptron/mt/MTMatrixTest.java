package org.yah.tests.perceptron.mt;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

public class MTMatrixTest {

    private double[] data;

    @Before
    public void setUp() throws Exception {
        data = new double[15];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }
    }

    /**
     * <code>
     * 0 5 10
     * 1 6 11
     * 2 7 12
     * 3 8 13
     * 4 9 14
     * </code>
     * 
     */
    @Test
    public void testGet() {
        MTMatrix matrix = new MTMatrix(data, 5, 3, 0, 5, 1);
        assertMatrix(new double[][] { { 0, 5, 10 }, { 1, 6, 11 }, { 2, 7, 12 }, { 3, 8, 13 }, { 4, 9, 14 } }, matrix);

        matrix = new MTMatrix(data, 5, 2, 5, 5, 1);
        assertMatrix(new double[][] { { 5, 10 }, { 6, 11 }, { 7, 12 }, { 8, 13 }, { 9, 14 } }, matrix);

        matrix = new MTMatrix(data, 3, 3, 2, 5, 1);
        assertMatrix(new double[][] { { 2, 7, 12 }, { 3, 8, 13 }, { 4, 9, 14 } }, matrix);
    }

    /**
     * <code>
     * 0 5 10
     * 1 6 11
     * 2 7 12
     * 3 8 13
     * 4 9 14
     * </code>
     * 
     * <code> 
     * 0  1  2  3  4 
     * 5  6  7  8  9 
     * 10 11 12 13 14 
     * </code>
     */
    @Test
    public void testTranspose() {
        MTMatrix matrix = new MTMatrix(data, 5, 3, 0, 5, 1).transpose();
        assertMatrix(new double[][] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 }, { 10, 11, 12, 13, 14 } }, matrix);

        matrix = new MTMatrix(data, 5, 2, 5, 5, 1).transpose();
        assertMatrix(new double[][] { { 5, 6, 7, 8, 9 }, { 10, 11, 12, 13, 14 } }, matrix);

        matrix = new MTMatrix(data, 3, 3, 2, 5, 1).transpose();
        assertMatrix(new double[][] { { 2, 3, 4 }, { 7, 8, 9 }, { 12, 13, 14 } }, matrix);
    }

    @Test
    public void testZero() {
        for (int r = 0; r < 100; r++) {
            for (int c = 0; c < 50; c++) {
                assertEquals(0, MTMatrix.ZERO.get(r, c), 0);
            }
        }
    }
    
    @Test
    public void testMaxRowIndex() {
        MTMatrix matrix = new MTMatrix(data, 5, 3, 0, 5, 1);
        assertEquals(4, matrix.maxRowIndex(0));
        assertEquals(4, matrix.maxRowIndex(1));
        assertEquals(4, matrix.maxRowIndex(2));
        
        matrix = matrix.transpose();
        
        assertEquals(2, matrix.maxRowIndex(0));
        assertEquals(2, matrix.maxRowIndex(1));
        assertEquals(2, matrix.maxRowIndex(2));
        assertEquals(2, matrix.maxRowIndex(3));
        assertEquals(2, matrix.maxRowIndex(4));
    }

    static void assertMatrix(double[][] expected, MTMatrix actual) {
        assertEquals(expected.length, actual.rows());
        assertEquals(expected[0].length, actual.columns());
        for (int r = 0; r < expected.length; r++) {
            for (int c = 0; c < expected[0].length; c++) {
                assertEquals(expected[r][c], actual.get(r, c), 0);
            }
        }
    }
}
