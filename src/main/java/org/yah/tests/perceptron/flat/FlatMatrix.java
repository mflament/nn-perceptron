/**
 * 
 */
package org.yah.tests.perceptron.flat;

import org.yah.tests.perceptron.Matrix;

/**
 * @author Yah
 *
 */
public class FlatMatrix implements Matrix {

    private double[] data;
    private int rows, columns;
    private int offset, rowStride, columnStride;

    public FlatMatrix() {}

    public FlatMatrix(double[][] data) {
        rows = data.length;
        columns = data[0].length;
        offset = 0;
        rowStride = columns;
        columnStride = 1;
        this.data = new double[rows * columns];
        rowMajor((r, c, f) -> data[r][c]);
    }

    public FlatMatrix(double[] data, int rows, int columns, int offset, int rowStride,
            int columnStride) {
        this.data = data;
        this.rows = rows;
        this.columns = columns;
        this.offset = offset;
        this.rowStride = rowStride;
        this.columnStride = columnStride;
    }

    public FlatMatrix(FlatMatrix matrix) {
        this(matrix.data, matrix.rows, matrix.columns, matrix.offset, matrix.rowStride,
                matrix.columnStride);
    }

    public FlatMatrix(int rows, int columns) {
        this(new double[rows * columns], rows, columns, 0, columns, 1);
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int columns() {
        return columns;
    }

    @Override
    public double get(int row, int col) {
        return data[indexOf(row, col)];
    }

    @Override
    public void set(int row, int col, double value) {
        data[indexOf(row, col)] = value;
    }

    @Override
    public double update(int row, int column, MatrixFunction func) {
        int index = indexOf(row, column);
        double old = data[index];
        data[index] = func.apply(row, column, old);
        return old;
    }

    private int indexOf(int row, int column) {
        return offset + row * rowStride + column * columnStride;
    }

    @Override
    public String toString() {
        return Matrix.toString(this);
    }

}
