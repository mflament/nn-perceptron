/**
 * 
 */
package org.yah.tests.perceptron.array;

import java.util.Arrays;

import org.yah.tests.perceptron.Activation;
import org.yah.tests.perceptron.Matrix;

/**
 * @author Yah
 *
 */
public class ArrayMatrix implements Matrix<ArrayMatrix> {

    private double[][] data; // [col][rows]

    private int colOffset;
    private int columns;

    public ArrayMatrix(int rows, int columns) {
        this.data = new double[columns][rows];
        this.columns = columns;
    }

    public ArrayMatrix(double[][] data) {
        set(data);
    }

    public ArrayMatrix(ArrayMatrix from) {
        this.data = from.data;
        this.columns = this.data.length;
    }

    public void set(double[][] data) {
        colOffset = 0;
        columns = data.length;
        int rows = data[0].length;
        this.data = new double[columns][rows];
        for (int col = 0; col < columns; col++) {
            System.arraycopy(data[col], 0, this.data[col], 0, rows);
        }
    }

    @Override
    public int slide(int offset, int columns) {
        this.colOffset = offset;
        this.columns = Math.min(columns, data.length - colOffset);
        return this.columns;
    }

    @Override
    public ArrayMatrix createView() {
        return new ArrayMatrix(this);
    }

    @Override
    public ArrayMatrix self() {
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
    public void zero() {
        int lastCol = colOffset + columns;
        for (int col = colOffset; col < lastCol; col++) {
            Arrays.fill(data[col], 0);
        }
    }

    @Override
    public ArrayMatrix sub(ArrayMatrix b, ArrayMatrix target) {
        int rows = rows();
        for (int col = 0; col < columns; col++) {
            double[] acols = data[col + colOffset];
            double[] bcols = b.data[col + b.colOffset];
            double[] tcols = target.data[col + target.colOffset];
            for (int row = 0; row < rows; row++) {
                tcols[row] = acols[row] - bcols[row];
            }
        }
        return target;
    }

    @Override
    public ArrayMatrix mul(ArrayMatrix b, ArrayMatrix target) {
        int rows = rows();
        for (int col = 0; col < columns; col++) {
            double[] acols = data[col + colOffset];
            double[] bcols = b.data[col + b.colOffset];
            double[] tcols = target.data[col + target.colOffset];
            for (int row = 0; row < rows; row++) {
                tcols[row] = acols[row] * bcols[row];
            }
        }
        return target;
    }

    @Override
    public ArrayMatrix mul(double s, ArrayMatrix target) {
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
    public ArrayMatrix dot(ArrayMatrix b, ArrayMatrix target) {
        assert columns == b.rows();
        assert target.rows() == rows() && target.columns == b.columns;

        int trows = target.rows();
        int pc = columns();
        for (int tc = 0; tc < target.columns; tc++) {
            double[] tcol = target.data[tc + target.colOffset];
            double[] bcol = b.data[tc + b.colOffset];
            for (int tr = 0; tr < trows; tr++) {
                tcol[tr] = 0;
                for (int i = 0; i < pc; i++) {
                    tcol[tr] += data[i + colOffset][tr] * bcol[i];
                }
            }
        }
        return target;
    }

    @Override
    public ArrayMatrix transpose_dot(ArrayMatrix b, ArrayMatrix target) {
        assert rows() == b.rows();
        assert target.rows() == columns && target.columns == b.columns;

        int trows = target.rows();
        int pc = rows();
        for (int tc = 0; tc < target.columns; tc++) {
            double[] bcol = b.data[tc + b.colOffset];
            double[] tcol = target.data[tc + target.colOffset];
            for (int tr = 0; tr < trows; tr++) {
                double[] col = data[tr + colOffset];
                tcol[tr] = 0;
                for (int i = 0; i < pc; i++) {
                    tcol[tr] += col[i] * bcol[i];
                }
            }
        }
        return target;
    }

    @Override
    public ArrayMatrix dot_transpose(ArrayMatrix b, ArrayMatrix target) {
        assert columns == b.columns;
        assert target.rows() == rows() && target.columns == b.rows();

        int trows = target.rows();
        int pc = columns();
        for (int tc = 0; tc < target.columns; tc++) {
            double[] tcol = target.data[tc + target.colOffset];
            for (int tr = 0; tr < trows; tr++) {
                tcol[tr] = 0;
                for (int i = 0; i < pc; i++) {
                    tcol[tr] += data[i + colOffset][tr] * b.data[i + colOffset][tc];
                }
            }
        }
        return target;
    }

    @Override
    public ArrayMatrix dot(ArrayMatrix b) {
        return dot(b, new ArrayMatrix(rows(), b.columns()));
    }

    @Override
    public ArrayMatrix transpose_dot(ArrayMatrix b) {
        return transpose_dot(b, new ArrayMatrix(columns(), b.columns()));
    }

    @Override
    public ArrayMatrix dot_transpose(ArrayMatrix b) {
        return dot_transpose(b, new ArrayMatrix(rows(), b.rows()));
    }

    @Override
    public double get(int row, int col) {
        return data[col + colOffset][row];
    }

    @Override
    public ArrayMatrix addColumnVector(ArrayMatrix vector, ArrayMatrix target) {
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
        return target;
    }

    @Override
    public ArrayMatrix sigmoid(ArrayMatrix target) {
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
    public ArrayMatrix sigmoid_prime(ArrayMatrix target) {
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
        return target;
    }

    @Override
    public ArrayMatrix sumRows(ArrayMatrix target) {
        int rows = rows();
        assert target.rows() == rows;
        assert target.columns == 1;
        double[] tcol = target.data[colOffset];
        Arrays.fill(tcol, 0);
        for (int c = 0; c < columns; c++) {
            double[] col = data[c + colOffset];
            for (int r = 0; r < rows; r++) {
                tcol[r] += col[r];
            }
        }
        return target;
    }

    @Override
    public int maxRowIndex(int column) {
        double[] col = data[column + colOffset];
        int res = -1;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < col.length; i++) {
            if (col[i] > max) {
                res = i;
                max = col[i];
            }
        }
        return res;
    }

    @Override
    public String toString() {
        return Matrix.toString(this);
    }

}
