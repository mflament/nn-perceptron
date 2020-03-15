/**
 * 
 */
package org.yah.tests.perceptron.matrix.flat;

import java.util.Arrays;

import org.yah.tests.perceptron.Activation;
import org.yah.tests.perceptron.matrix.Matrix;

/**
 * @author Yah
 *
 */
public class CMFlatMatrix implements Matrix<CMFlatMatrix> {

    private final double[] data;
    private final int rows, totalColumns;
    private int colOffset, columns;

    /**
     * @param data column major data (needs a standard ...)
     */
    public CMFlatMatrix(double[][] _data) {
        this.totalColumns = _data.length;
        this.rows = _data[0].length;
        this.columns = totalColumns;
        this.colOffset = 0;
        this.data = new double[rows * columns];
        for (int c = 0; c < _data.length; c++) {
            System.arraycopy(_data[c], 0, this.data, c * rows, rows);
        }
    }

    public CMFlatMatrix(int rows, int columns) {
        this.totalColumns = columns;
        this.rows = rows;
        this.columns = totalColumns;
        this.colOffset = 0;
        this.data = new double[rows * columns];
    }

    private CMFlatMatrix(CMFlatMatrix from) {
        this.totalColumns = from.totalColumns;
        this.columns = totalColumns;
        this.rows = from.rows;
        this.data = from.data;
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
        return data[indexOf(col) + row];
    }

    @Override
    public CMFlatMatrix self() {
        return this;
    }

    @Override
    public void apply(MatrixFunction func) {
        for (int c = 0; c < columns; c++) {
            int ci = indexOf(c);
            for (int r = 0; r < rows; r++) {
                data[ci + r] = func.apply(r, c, data[ci + r]);
            }
        }
    }

    @Override
    public void zero() {
        int start = indexOf(0);
        Arrays.fill(data, start, start + columns, 0);
    }

    @Override
    public CMFlatMatrix sub(CMFlatMatrix b, CMFlatMatrix target) {
        for (int c = 0; c < columns; c++) {
            int ci = indexOf(c);
            int bci = b.indexOf(c);
            int tci = target.indexOf(c);
            for (int r = 0; r < rows; r++) {
                target.data[tci + r] = data[ci + r] - b.data[bci + r];
            }
        }
        return target;
    }

    @Override
    public CMFlatMatrix mul(CMFlatMatrix b, CMFlatMatrix target) {
        for (int c = 0; c < columns; c++) {
            int ci = indexOf(c);
            int bci = b.indexOf(c);
            int tci = target.indexOf(c);
            for (int r = 0; r < rows; r++) {
                target.data[tci + r] = data[ci + r] * b.data[bci + r];
            }
        }
        return target;
    }

    @Override
    public CMFlatMatrix mul(double s, CMFlatMatrix target) {
        for (int c = 0; c < columns; c++) {
            int ci = indexOf(c);
            int tci = target.indexOf(c);
            for (int r = 0; r < rows; r++) {
                target.data[tci + r] = data[ci + r] * s;
            }
        }
        return target;
    }

    @Override
    public CMFlatMatrix addColumnVector(CMFlatMatrix vector, CMFlatMatrix target) {
        assert vector.columns == 1;
        assert vector.rows() == rows;

        int vi = vector.indexOf(vector.colOffset);
        for (int c = 0; c < columns; c++) {
            int ci = indexOf(c);
            int tci = target.indexOf(c);
            for (int r = 0; r < rows; r++) {
                target.data[tci + r] = data[ci + r] + vector.data[vi + r];
            }
        }
        return target;
    }

    @Override
    public CMFlatMatrix sumRows(CMFlatMatrix target) {
        assert target.rows == rows;
        assert target.columns == 1;
        for (int r = 0; r < rows; r++) {
            double s = 0;
            for (int c = 0; c < columns; c++) {
                s += data[indexOf(c) + r];
            }
            target.data[target.indexOf(0) + r] = s;
        }
        return target;
    }

    @Override
    public CMFlatMatrix dot(CMFlatMatrix b, CMFlatMatrix target) {
        assert columns == b.rows();
        assert target.rows() == rows() && target.columns == b.columns;

        for (int tr = 0; tr < target.rows; tr++) {
            for (int tc = 0; tc < target.columns; tc++) {
                int bi = b.indexOf(tc);
                double v = 0;
                for (int c = 0; c < columns; c++) {
                    v += data[indexOf(c) + tr] * b.data[bi + c];
                }
                target.data[target.indexOf(tc) + tr] = v;
            }
        }
        return target;
    }

    @Override
    public CMFlatMatrix transpose_dot(CMFlatMatrix b, CMFlatMatrix target) {
        assert rows() == b.rows();
        assert target.rows() == columns && target.columns == b.columns;

        for (int tr = 0; tr < target.rows; tr++) {
            for (int tc = 0; tc < target.columns; tc++) {
                int bi = b.indexOf(tc);
                int i = indexOf(tr);
                double v = 0;
                for (int c = 0; c < rows; c++) {
                    v += data[i + c] * b.data[bi + c];
                }
                target.data[target.indexOf(tc) + tr] = v;
            }
        }
        return target;
    }

    @Override
    public CMFlatMatrix dot_transpose(CMFlatMatrix b, CMFlatMatrix target) {
        assert columns == b.columns;
        assert target.rows() == rows() && target.columns == b.rows();
        
        for (int tr = 0; tr < target.rows; tr++) {
            for (int tc = 0; tc < target.columns; tc++) {
                double v = 0;
                for (int c = 0; c < columns; c++) {
                    v += data[indexOf(c) + tr] * b.data[b.indexOf(c) + tc];
                }
                target.data[target.indexOf(tc) + tr] = v;
            }
        }
        return target;
    }

    @Override
    public CMFlatMatrix dot(CMFlatMatrix b) {
        return dot(b, new CMFlatMatrix(rows(), b.columns()));
    }

    @Override
    public CMFlatMatrix transpose_dot(CMFlatMatrix b) {
        return transpose_dot(b, new CMFlatMatrix(columns(), b.columns()));
    }

    @Override
    public CMFlatMatrix dot_transpose(CMFlatMatrix b) {
        return dot_transpose(b, new CMFlatMatrix(rows(), b.rows()));
    }

    @Override
    public CMFlatMatrix sigmoid(CMFlatMatrix target) {
        int ci = indexOf(colOffset);
        int tci = target.indexOf(target.colOffset);
        int count = rows * columns;
        for (int i = 0; i < count; i++) {
            target.data[tci + i] = Activation.sigmoid(data[ci + i]);
        }
        return target;
    }

    @Override
    public CMFlatMatrix sigmoid_prime(CMFlatMatrix target) {
        int ci = indexOf(colOffset);
        int tci = target.indexOf(target.colOffset);
        int count = rows * columns;
        for (int i = 0; i < count; i++) {
            target.data[tci + i] = Activation.sigmoid_prime(data[ci + i]);
        }
        return target;
    }

    @Override
    public int maxRowIndex(int column) {
        int ci = indexOf(column);
        int res = -1;
        double max = Double.MIN_VALUE;
        for (int r = 0; r < rows; r++) {
            double v = data[ci + r];
            if (v > max) {
                res = r;
                max = v;
            }
        }
        return res;
    }

    @Override
    public int slide(int offset, int columns) {
        this.colOffset = offset;
        this.columns = Math.min(columns, totalColumns - colOffset);
        return this.columns;
    }

    @Override
    public CMFlatMatrix createView() {
        return new CMFlatMatrix(this);
    }

    private int indexOf(int column) {
        return (column + colOffset) * rows;
    }

//    private int indexOf(int row, int column) {
//        return offset + row * rowStride + column * columnStride;
//    }

    @Override
    public String toString() {
        return Matrix.toString(this);
    }

}
