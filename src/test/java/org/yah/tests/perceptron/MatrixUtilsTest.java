package org.yah.tests.perceptron;

import static org.junit.Assert.*;

import org.junit.Test;

public class MatrixUtilsTest {
    private Matrix a = new ArrayMatrix(new float[][] { { 1, 2, 3 }, { 4, 5, 6 } });
    private Matrix b = new ArrayMatrix(new float[][] { { 1, 2, 3 }, { 4, 5, 6 } });

    @Test
    public void testAdd() {
        Matrix.add(a, b, a);
        assertEquals(2, a.get(0, 0), 0);
        assertEquals(4, a.get(0, 1), 0);
        assertEquals(6, a.get(0, 2), 0);
        assertEquals(8, a.get(1, 0), 0);
        assertEquals(10, a.get(1, 1), 0);
        assertEquals(12, a.get(1, 2), 0);
    }

    @Test
    public void testSub() {
        Matrix.sub(a, b, a);
        assertEquals(0, a.get(0, 0), 0);
        assertEquals(0, a.get(0, 1), 0);
        assertEquals(0, a.get(0, 2), 0);
        assertEquals(0, a.get(1, 0), 0);
        assertEquals(0, a.get(1, 1), 0);
        assertEquals(0, a.get(1, 2), 0);
    }

    @Test
    public void testMul() {
        Matrix.mul(a, 2, a);
        assertEquals(2, a.get(0, 0), 0);
        assertEquals(4, a.get(0, 1), 0);
        assertEquals(6, a.get(0, 2), 0);
        assertEquals(8, a.get(1, 0), 0);
        assertEquals(10, a.get(1, 1), 0);
        assertEquals(12, a.get(1, 2), 0);
    }

    @Test
    public void testApply() {
        Matrix.apply(a, v -> v + 1, b);
        assertEquals(2, b.get(0, 0), 0);
        assertEquals(3, b.get(0, 1), 0);
        assertEquals(4, b.get(0, 2), 0);
        assertEquals(5, b.get(1, 0), 0);
        assertEquals(6, b.get(1, 1), 0);
        assertEquals(7, b.get(1, 2), 0);
    }

    @Test
    public void testDot() {
        b = new ArrayMatrix(new float[][] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
        Matrix res = new ArrayMatrix(2, 2);
        Matrix.dot(a, b, res);
        assertEquals(58, res.get(0, 0), 0);
        assertEquals(64, res.get(0, 1), 0);
        assertEquals(139, res.get(1, 0), 0);
        assertEquals(154, res.get(1, 1), 0);
    }

}
