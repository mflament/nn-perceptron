package org.yah.tests.perceptron;

public class TransposedMatrix implements Matrix {
    private Matrix delegate;

    public TransposedMatrix() {}

    public TransposedMatrix(Matrix matrix) {
        this.delegate = matrix;
    }

    public Matrix getDelegate() { return delegate; }

    public void transpose(Matrix delegate) {
        this.delegate = delegate;
    }

    @Override
    public int rows() {
        return delegate.columns();
    }

    @Override
    public int columns() {
        return delegate.rows();
    }

    @Override
    public Matrix transpose() {
        return delegate;
    }

    @Override
    public float get(int row, int col) {
        return delegate.get(col, row);
    }

    @Override
    public void set(int row, int col, float v) {
        delegate.set(col, row, v);
    }

    @Override
    public String toString() {
        return Matrix.toString(this);
    }

}