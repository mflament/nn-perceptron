package org.yah.tests.perceptron;

import java.util.Random;

public interface Matrix {

    int rows();

    int columns();

    float get(int row, int col);

    void set(int row, int col, float v);
    
    Matrix transpose();
    
    default Matrix add(Matrix m) {
        Matrix.add(this, m, this);
        return this;
    }

    default Matrix sub(Matrix m) {
        Matrix.sub(this, m, this);
        return this;
    }

    default Matrix mul(Matrix m) {
        Matrix.mul(this, m, this);
        return this;
    }

    default Matrix mul(float s) {
        Matrix.mul(this, s, this);
        return this;
    }

    default Matrix zero() {
        Matrix.zero(this);
        return this;
    }

    default Matrix apply(MatrixValueFunction func) {
        Matrix.apply(this, func, this);
        return this;
    }

    default Matrix random() {
        Matrix.random(this);
        return this;
    }

    default int maxRowIndex(int column) {
        return Matrix.maxRowIndex(this, column);
    }

    interface MatrixValueFunction {
        float apply(float value);
    }

    static void add(Matrix a, Matrix b, Matrix result) {
        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.columns(); c++) {
                result.set(r, c, a.get(r, c) + b.get(r, c));
            }
        }
    }

    static void sub(Matrix a, Matrix b, Matrix result) {
        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.columns(); c++) {
                result.set(r, c, a.get(r, c) - b.get(r, c));
            }
        }
    }

    static void mul(Matrix a, Matrix b, Matrix result) {
        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.columns(); c++) {
                result.set(r, c, a.get(r, c) * b.get(r, c));
            }
        }
    }

    static void mul(Matrix m, float s, Matrix result) {
        for (int r = 0; r < m.rows(); r++) {
            for (int c = 0; c < m.columns(); c++) {
                result.set(r, c, m.get(r, c) * s);
            }
        }
    }

    static void apply(Matrix m, MatrixValueFunction function, Matrix result) {
        for (int r = 0; r < m.rows(); r++) {
            for (int c = 0; c < m.columns(); c++) {
                result.set(r, c, function.apply(m.get(r, c)));
            }
        }
    }

    static void dot(Matrix a, Matrix b, Matrix result) {
        assert a != result && b != result;
        assert a.columns() == b.rows();
        assert result.rows() == a.rows();
        assert result.columns() == b.columns();
        Matrix.zero(result);
        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.columns(); c++) {
                float av = a.get(r, c);
                for (int bc = 0; bc < b.columns(); bc++) {
                    float v = result.get(r, bc) + av * b.get(c, bc);
                    result.set(r, bc, v);
                }
            }
        }
    }

    static int maxRowIndex(Matrix m, int column) {
        float max = Float.MIN_VALUE;
        int res = -1;
        for (int r = 0; r < m.rows(); r++) {
            float v = m.get(r, column);
            if (v > max) {
                res = r;
                max = v;
            }
        }
        return res;
    }

    static void zero(Matrix m) {
        for (int r = 0; r < m.rows(); r++) {
            for (int c = 0; c < m.columns(); c++) {
                m.set(r, c, 0);
            }
        }
    }

    static void random(Matrix m) {
        apply(m, v -> (float) RANDOM.nextGaussian(), m);
    }

    static final Random RANDOM = createRandom();

    static Random createRandom() {
        long seed = seed();
        return seed < 0 ? new Random() : new Random(seed);
    }

    static long seed() {
        long seed = -1;
        String prop = System.getProperty("seed");
        if (prop != null) {
            try {
                seed = Long.parseLong(prop);
            } catch (NumberFormatException e) {}
        }
        return seed;
    }

    static String toString(Matrix m) {
        StringBuilder sb = new StringBuilder();
        for (int row = 0; row < m.rows(); row++) {
            for (int col = 0; col < m.columns(); col++) {
                float v = m.get(row, col);
                sb.append(String.format("%5.3f ", v));
            }
            sb.append(System.lineSeparator());
        }
        return sb.toString();
    }
}