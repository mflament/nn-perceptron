package org.yah.tests.perceptron.mt;

import java.util.NoSuchElementException;

/**
 * @author Yah
 *
 */
public class MatrixIterator {

    private int offset;
    private int stride;
    private int remaining;

    public MatrixIterator() {}

    public MatrixIterator(int offset, int stride, int remaining) {
        set(offset, stride, remaining);
    }

    public void set(int offset, int stride, int remaining) {
        this.offset = offset;
        this.stride = stride;
        this.remaining = remaining;

    }

    public int remaining() {
        return remaining;
    }

    public boolean hasNext() {
        return remaining > 0;
    }

    public int next() {
        if (!hasNext())
            throw new NoSuchElementException();
        int res = offset;
        offset += stride;
        remaining--;
        return res;
    }
}
