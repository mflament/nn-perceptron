package org.yah.tests.perceptron;

public class ArrayMatrix implements Matrix {
    private final float[][] values;
    private TransposedMatrix transposedMatrix;

    public ArrayMatrix(float[][] values) {
        this.values = values;
    }

    public ArrayMatrix(int rows, int columns) {
        this(new float[rows][columns]);
    }

    public float get(int row, int col) {
        return values[row][col];
    }

    public void set(int row, int col, float v) {
        values[row][col] = v;
    }
    @Override
    public Matrix transpose() {
        if (transposedMatrix == null)
            transposedMatrix = new TransposedMatrix(this);
        return transposedMatrix;
    }

    @Override
    public int rows() {
        return values.length;
    }

    @Override
    public int columns() {
        return values[0].length;
    }

    @Override
    public String toString() {
        return Matrix.toString(this);
    }
}