package org.yah.tests.perceptron.mt;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class MatrixIteratorTest {

    /**
     * <code>
     * 0 4 8
     * 1 5 9 
     * 2 6 10
     * 3 7 11 
     * </code>
     */
    private final double[] data = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

    @Test
    public void test() {
        MatrixIterator iterator = new MatrixIterator(0, 1, 4);
        assertIterator(iterator, 0, 1, 2, 3);

        iterator = new MatrixIterator(4, 1, 4);
        assertIterator(iterator, 4, 5, 6, 7);

        iterator = new MatrixIterator(0, 4, 2);
        assertIterator(iterator, 0, 4);

        iterator = new MatrixIterator(1, 4, 2);
        assertIterator(iterator, 1, 5);
    }

    private void assertIterator(MatrixIterator iterator, int... expecteds) {
        int index = 0;
        while (index < expecteds.length) {
            assertEquals(expecteds.length - index, iterator.remaining());
            assertTrue(iterator.hasNext());
            assertEquals(expecteds[index], data[iterator.next()], 0);
            index++;
        }
        assertEquals(0, iterator.remaining());
        assertFalse(iterator.hasNext());
    }

}
