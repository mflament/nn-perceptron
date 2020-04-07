package org.yah.tests.perceptron.matrix.array;

import java.util.Arrays;

import org.yah.tests.perceptron.Activation;
import org.yah.tests.perceptron.matrix.Matrix;
import org.yah.tests.perceptron.matrix.MatrixFunction;

/**
 * @author Yah
 */
public class CMArrayMatrix implements Matrix<CMArrayMatrix> {

    private final double[][] data; // [col][rows]

    private int colOffset;
    private int columns;

    public CMArrayMatrix(int rows, int columns) {
        this.data = new double[columns][rows];
        this.columns = columns;
    }

    /**
     *
     */
    public CMArrayMatrix(double[][] _data) {
        colOffset = 0;
        columns = _data.length;
        int rows = _data[0].length;
        data = new double[columns][rows];
        for (int col = 0; col < columns; col++) {
            System.arraycopy(_data[col], 0, data[col], 0, rows);
        }
    }

    public CMArrayMatrix(CMArrayMatrix from) {
        this.data = from.data;
        this.columns = this.data.length;
        this.colOffset = 0;
    }

    @Override
    public int slide(int offset, int columns) {
        this.colOffset = offset;
        this.columns = Math.min(columns, data.length - colOffset);
        return this.columns;
    }

    @Override
    public CMArrayMatrix createView() {
        return new CMArrayMatrix(this);
    }

    @Override
    public CMArrayMatrix self() {
        return this;
    }

    @Override
    public int rows() {
        return data.length == 0 ? 0 : data[0].length;
    }

    @Override
    public int columns() {
        return columns;
    }

    @Override
    public void apply(MatrixFunction func) {
        int rows = rows();
        for (int col = 0; col < columns; col++) {
            double[] column = data[col + colOffset];
            for (int row = 0; row < rows; row++) {
                column[row] = func.apply(row, col, column[row]);
            }
        }
    }

    @Override
    public void sub(CMArrayMatrix b, CMArrayMatrix target) {
        int rows = rows();
        for (int col = 0; col < columns; col++) {
            double[] acols = data[col + colOffset];
            double[] bcols = b.data[col + b.colOffset];
            double[] tcols = target.data[col + target.colOffset];
            for (int row = 0; row < rows; row++) {
                tcols[row] = acols[row] - bcols[row];
            }
        }
    }

    @Override
    public void mul(CMArrayMatrix b, CMArrayMatrix target) {
        int rows = rows();
        for (int col = 0; col < columns; col++) {
            double[] acols = data[col + colOffset];
            double[] bcols = b.data[col + b.colOffset];
            double[] tcols = target.data[col + target.colOffset];
            for (int row = 0; row < rows; row++) {
                tcols[row] = acols[row] * bcols[row];
            }
        }
    }

    @Override
    public CMArrayMatrix mul(double s, CMArrayMatrix target) {
        int rows = rows();
        for (int col = 0; col < columns; col++) {
            double[] acols = data[col + colOffset];
            double[] tcols = target.data[col + target.colOffset];
            for (int row = 0; row < rows; row++) {
                tcols[row] = acols[row] * s;
            }
        }
        return target;
    }

    @Override
    public CMArrayMatrix dot(CMArrayMatrix b, CMArrayMatrix target) {
        assert columns == b.rows();
        assert target.rows() == rows() && target.columns == b.columns;

        int trows = target.rows();
        for (int tc = 0; tc < target.columns; tc++) {
            double[] tcol = target.data[tc + target.colOffset];
            double[] bcol = b.data[tc + b.colOffset];
            for (int tr = 0; tr < trows; tr++) {
                double v = 0;
                for (int i = 0; i < columns; i++) {
                    v += data[i + colOffset][tr] * bcol[i];
                }
                tcol[tr] = v;
            }
        }
        return target;
    }

    @Override
    public CMArrayMatrix transpose_dot(CMArrayMatrix b, CMArrayMatrix target) {
        assert rows() == b.rows();
        assert target.rows() == columns && target.columns == b.columns;

        int trows = target.rows();
        int pc = rows();
        for (int tc = 0; tc < target.columns; tc++) {
            double[] bcol = b.data[tc + b.colOffset];
            double[] tcol = target.data[tc + target.colOffset];
            for (int tr = 0; tr < trows; tr++) {
                double[] col = data[tr + colOffset];
                double v = 0;
                for (int i = 0; i < pc; i++) {
                    v += col[i] * bcol[i];
                }
                tcol[tr] = v;
            }
        }
        return target;
    }

    @Override
    public CMArrayMatrix dot_transpose(CMArrayMatrix b, CMArrayMatrix target) {
        assert columns == b.columns;
        assert target.rows() == rows() && target.columns == b.rows();

        int trows = target.rows();
        int pc = columns();
        for (int tc = 0; tc < target.columns; tc++) {
            double[] tcol = target.data[tc + target.colOffset];
            for (int tr = 0; tr < trows; tr++) {
                double v = 0;
                for (int i = 0; i < pc; i++) {
                    v += data[i + colOffset][tr] * b.data[i + b.colOffset][tc];
                }
                tcol[tr] = v;
            }
        }
        return target;
    }

    @Override
    public CMArrayMatrix dot(CMArrayMatrix b) {
        return dot(b, new CMArrayMatrix(rows(), b.columns()));
    }

    @Override
    public CMArrayMatrix transpose_dot(CMArrayMatrix b) {
        return transpose_dot(b, new CMArrayMatrix(columns(), b.columns()));
    }

    @Override
    public CMArrayMatrix dot_transpose(CMArrayMatrix b) {
        return dot_transpose(b, new CMArrayMatrix(rows(), b.rows()));
    }

    @Override
    public double get(int row, int col) {
        return data[col + colOffset][row];
    }

    @Override
    public void set(int row, int col, double value) {
        data[col + colOffset][row] = value;
    }

    @Override
    public void addColumnVector(CMArrayMatrix vector, CMArrayMatrix target) {
        int rows = rows();
        assert vector.columns == 1;
        assert vector.rows() == rows;

        double[] vectorCol = vector.data[vector.colOffset];
        for (int i = 0; i < columns; i++) {
            double[] col = data[i + colOffset];
            double[] targetCol = target.data[i + target.colOffset];
            for (int row = 0; row < rows; row++) {
                targetCol[row] = col[row] + vectorCol[row];
            }
        }
    }

    @Override
    public CMArrayMatrix sigmoid(CMArrayMatrix target) {
        int rows = rows();
        assert target.columns == columns;
        assert target.rows() == rows;
        for (int c = 0; c < columns; c++) {
            double[] col = data[c + colOffset];
            double[] tcol = target.data[c + target.colOffset];
            for (int r = 0; r < rows; r++) {
                tcol[r] = Activation.sigmoid(col[r]);
            }
        }
        return target;
    }

    @Override
    public void sigmoid_prime(CMArrayMatrix target) {
        int rows = rows();
        assert target.columns == columns;
        assert target.rows() == rows;
        for (int c = 0; c < columns; c++) {
            double[] col = data[c + colOffset];
            double[] tcol = target.data[c + target.colOffset];
            for (int r = 0; r < rows; r++) {
                tcol[r] = Activation.sigmoid_prime(col[r]);
            }
        }
    }

    @Override
    public void sumRows(CMArrayMatrix target) {
        int rows = rows();
        assert target.rows() == rows;
        assert target.columns == 1;
        double[] tcol = target.data[target.colOffset];
        Arrays.fill(tcol, 0);
        for (int c = 0; c < columns; c++) {
            double[] col = data[c + colOffset];
            for (int r = 0; r < rows; r++) {
                tcol[r] += col[r];
            }
        }
    }

    @Override
    public int maxRowIndex(int column) {
        int res = -1;
        double max = Double.MIN_VALUE;
        double[] col = data[column + colOffset];
        for (int i = 0; i < col.length; i++) {
            double v = col[i];
            if (v > max) {
                res = i;
                max = v;
            }
        }
        return res;
    }

    @Override
    public String toString() {
        return Matrix.toString(this);
    }

}
