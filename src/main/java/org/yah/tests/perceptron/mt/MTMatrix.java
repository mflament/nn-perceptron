package org.yah.tests.perceptron.mt;

import java.util.Arrays;

public class MTMatrix {

    public static final MTMatrix ZERO = new MTMatrix(new double[] { 0 }, 1, 1, 0, 0, 0);

    private double[] data;

    private int rows, columns;
    private int offset, columnStride, rowStride;

    public MTMatrix() {}

    public MTMatrix(int rows, int columns) {
        this(new double[rows * columns], rows, columns, 0, rows, 1);
    }

    public MTMatrix(double[] data, int rows, int columns, int offset, int columnStride, int rowStride) {
        this.data = data;
        this.rows = rows;
        this.columns = columns;
        this.offset = offset;
        this.columnStride = columnStride;
        this.rowStride = rowStride;
    }

    public MTMatrix(MTMatrix from) {
        this.data = from.data;
        this.rows = from.rows;
        this.columns = from.columns;
        this.offset = from.offset;
        this.columnStride = from.columnStride;
        this.rowStride = from.rowStride;
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public int offset() {
        return offset;
    }

    public int index(int row, int col) {
        return offset + col * columnStride + row * rowStride;
    }

    public double get(int row, int col) {
        return data[index(row, col)];
    }

    public void set(int row, int col, double v) {
        data[index(row, col)] = v;
    }

    public void set(int index, double v) {
        data[index] = v;
    }

    public MTMatrix transpose() {
        return transpose(new MTMatrix());
    }

    public MTMatrix transpose(MTMatrix target) {
        target.data = data;
        target.offset = offset;

        // swap to handle transpose(this)
        int rows = this.rows;
        int columns = this.columns;
        int rowStride = this.rowStride;
        int columnStride = this.columnStride;

        target.rows = columns;
        target.columns = rows;
        target.columnStride = rowStride;
        target.rowStride = columnStride;
        return target;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                sb.append(String.format("%7.3f ", get(r, c)));
            }
            sb.append(System.lineSeparator());
        }
        return sb.toString();
    }

    public int capacity() {
        return data == null ? 0 : data.length;
    }

    public int size() {
        return columns * rows;
    }

    public void reshape(int rows, int columns) {
        int newSize = rows * columns;
        if (capacity() < newSize)
            data = new double[newSize];
        offset = 0;
        this.columns = columns;
        this.rows = rows;
        columnStride = rows;
        rowStride = 1;
    }

    public static MTMatrix fromRM(double[][] rmm) {
        MTMatrix res = new MTMatrix(rmm.length, rmm[0].length);
        for (int r = 0; r < rmm.length; r++) {
            for (int c = 0; c < rmm[0].length; c++) {
                res.set(r, c, rmm[r][c]);
            }
        }
        return res;
    }

    public void offset(int offset) {
        this.offset = offset;
    }

    public void columns(int columns) {
        this.columns = columns;
    }

    public int maxRowIndex(int col) {
        int res = 0;
        int offset = index(0, col);
        int resOffset = offset;
        for (int r = 1; r < rows; r++) {
            offset += rowStride;
            if (data[offset] > data[resOffset]) {
                resOffset = offset;
                res = r;
            }
        }
        return res;
    }

    public void sub(int row, int col, double v) {
        data[index(row, col)] -= v;
    }

    public double get(int index) {
        return data[index];
    }

    public double mul(int index, double s) {
        data[index] *= s;
        return data[index];
    }

    public void add(int row, int col, double delta) {
        data[index(row, col)] += delta;
    }

    public void add(int index, double delta) {
        data[index] += delta;
    }

    public void zero() {
        Arrays.fill(data, offset, size(), 0);
    }

}
