/**
 * 
 */
package org.yah.tests.perceptron.matrix.array;

import java.util.Arrays;

import org.yah.tests.perceptron.Activation;
import org.yah.tests.perceptron.matrix.Matrix;

/**
 * @author Yah
 *
 */
public class RMArrayMatrix implements Matrix<RMArrayMatrix> {

    private double[][] data; // [rows][col]

    private int colOffset;
    private int columns;

    public RMArrayMatrix(int rows, int columns) {
        this.data = new double[rows][columns];
        this.columns = columns;
    }

    /**
     * @param data column major data (needs a standard ...)
     */
    public RMArrayMatrix(double[][] _data) {
        int rows = _data[0].length;
        columns = _data.length;
        colOffset = 0;
        this.data = new double[rows][columns];
        for (int r = 0; r < rows; r++) {
            double[] row = data[r];
            for (int c = 0; c < columns; c++) {
                row[c] = _data[c][r];
            }
        }
    }

    public RMArrayMatrix(RMArrayMatrix from) {
        this.data = from.data;
        this.columns = this.data[0].length;
    }

    @Override
    public int slide(int offset, int columns) {
        this.colOffset = offset;
        this.columns = Math.min(columns, data[0].length - colOffset);
        return this.columns;
    }

    @Override
    public RMArrayMatrix createView() {
        return new RMArrayMatrix(this);
    }

    @Override
    public RMArrayMatrix self() {
        return this;
    }

    @Override
    public int rows() {
        return data.length;
    }

    @Override
    public int columns() {
        return columns;
    }

    @Override
    public void apply(MatrixFunction func) {
        for (int r = 0; r < data.length; r++) {
            double[] row = data[r];
            for (int c = 0; c < columns; c++) {
                row[c + colOffset] = func.apply(r, c, row[c + colOffset]);
            }
        }
    }

    @Override
    public void zero() {
        int lastCol = colOffset + columns;
        for (int r = 0; r < data.length; r++) {
            Arrays.fill(data[r], colOffset, lastCol, 0);
        }
    }

    @Override
    public RMArrayMatrix sub(RMArrayMatrix b, RMArrayMatrix target) {
        for (int r = 0; r < data.length; r++) {
            double[] arow = data[r];
            double[] brow = b.data[r];
            double[] trow = target.data[r];
            for (int col = 0; col < columns; col++) {
                trow[col + target.colOffset] = arow[col + colOffset] - brow[col + b.colOffset];
            }
        }
        return target;
    }

    @Override
    public RMArrayMatrix mul(RMArrayMatrix b, RMArrayMatrix target) {
        for (int r = 0; r < data.length; r++) {
            double[] arow = data[r];
            double[] brow = b.data[r];
            double[] trow = target.data[r];
            for (int col = 0; col < columns; col++) {
                trow[col + target.colOffset] = arow[col + colOffset] * brow[col + b.colOffset];
            }
        }
        return target;
    }

    @Override
    public RMArrayMatrix mul(double s, RMArrayMatrix target) {
        for (int r = 0; r < data.length; r++) {
            double[] arow = data[r];
            double[] trow = target.data[r];
            for (int col = 0; col < columns; col++) {
                trow[col + target.colOffset] = arow[col + colOffset] * s;
            }
        }
        return target;
    }

    @Override
    public RMArrayMatrix dot(RMArrayMatrix b, RMArrayMatrix target) {
        assert columns == b.rows();
        assert target.rows() == rows() && target.columns == b.columns;

        for (int tr = 0; tr < target.data.length; tr++) {
            double[] row = data[tr];
            double[] trow = target.data[tr];
            for (int bc = 0; bc < b.columns; bc++) {
                double v = 0;
                for (int c = 0; c < columns; c++) {
                    v += row[c + colOffset] * b.data[c][bc + b.colOffset];
                }
                trow[bc + target.colOffset] = v;
            }
        }
        return target;
    }

    @Override
    public RMArrayMatrix transpose_dot(RMArrayMatrix b, RMArrayMatrix target) {
        assert rows() == b.rows();
        assert target.rows() == columns && target.columns == b.columns;

        for (int tr = 0; tr < target.data.length; tr++) {
            double[] trow = target.data[tr];
            for (int bc = 0; bc < b.columns; bc++) {
                double v = 0;
                for (int c = 0; c < data.length; c++) {
                    v += data[c][tr + colOffset] * b.data[c][bc + b.colOffset];
                }
                trow[bc + target.colOffset] = v;
            }
        }
        return target;
    }

    @Override
    public RMArrayMatrix dot_transpose(RMArrayMatrix b, RMArrayMatrix target) {
        assert columns == b.columns;
        assert target.rows() == rows() && target.columns == b.rows();

        for (int tr = 0; tr < target.data.length; tr++) {
            double[] row = data[tr];
            double[] trow = target.data[tr];
            for (int bc = 0; bc < b.data.length; bc++) {
                double v = 0;
                double[] brow = b.data[bc];
                for (int c = 0; c < columns; c++) {
                    v += row[c + colOffset] * brow[c + b.colOffset];
                }
                trow[bc + target.colOffset] = v;
            }
        }
        return target;
    }

    @Override
    public RMArrayMatrix dot(RMArrayMatrix b) {
        return dot(b, new RMArrayMatrix(rows(), b.columns()));
    }

    @Override
    public RMArrayMatrix transpose_dot(RMArrayMatrix b) {
        return transpose_dot(b, new RMArrayMatrix(columns(), b.columns()));
    }

    @Override
    public RMArrayMatrix dot_transpose(RMArrayMatrix b) {
        return dot_transpose(b, new RMArrayMatrix(rows(), b.rows()));
    }

    @Override
    public double get(int row, int col) {
        return data[row][col + colOffset];
    }

    @Override
    public RMArrayMatrix addColumnVector(RMArrayMatrix vector, RMArrayMatrix target) {
        int rows = rows();
        assert vector.columns == 1;
        assert vector.rows() == rows;

        for (int r = 0; r < data.length; r++) {
            double[] row = data[r];
            double[] trow = target.data[r];
            double v = vector.data[r][vector.colOffset];
            for (int col = 0; col < columns; col++) {
                trow[col + target.colOffset] = row[col + colOffset] + v;
            }
        }
        return target;
    }

    @Override
    public RMArrayMatrix sigmoid(RMArrayMatrix target) {
        assert target.columns == columns;
        assert target.rows() == rows();
        for (int r = 0; r < data.length; r++) {
            double[] row = data[r];
            double[] trow = target.data[r];
            for (int c = 0; c < columns; c++) {
                trow[c + target.colOffset] = Activation.sigmoid(row[c + colOffset]);
            }
        }
        return target;
    }

    @Override
    public RMArrayMatrix sigmoid_prime(RMArrayMatrix target) {
        int rows = rows();
        assert target.columns == columns;
        assert target.rows() == rows;
        for (int r = 0; r < data.length; r++) {
            double[] row = data[r];
            double[] trow = target.data[r];
            for (int c = 0; c < columns; c++) {
                trow[c + target.colOffset] = Activation.sigmoid_prime(row[c + colOffset]);
            }
        }
        return target;
    }

    @Override
    public RMArrayMatrix sumRows(RMArrayMatrix target) {
        int rows = rows();
        assert target.rows() == rows;
        assert target.columns == 1;
        for (int r = 0; r < data.length; r++) {
            double[] row = data[r];
            double[] trow = target.data[r];
            double v = 0;
            for (int c = 0; c < columns; c++) {
                v += row[c + colOffset];
            }
            trow[target.colOffset] = v;
        }
        return target;
    }

    @Override
    public int maxRowIndex(int column) {
        int res = -1;
        double max = Double.MIN_VALUE;
        for (int r = 0; r < data.length; r++) {
            double v = data[r][column + colOffset];
            if (v > max) {
                res = r;
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
