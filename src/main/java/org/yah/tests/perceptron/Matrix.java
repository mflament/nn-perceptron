package org.yah.tests.perceptron;

public class Matrix {
    public static double[][] matrix(int rows, int cols) {
        return new double[rows][cols];
    }

    public static double[][] transpose(double[][] m) {
        double[][] t = matrix(m[0].length, m.length);
        for (int r = 0; r < m.length; r++) {
            for (int c = 0; c < m[r].length; c++) {
                t[c][r] = m[r][c];
            }
        }
        return t;
    }

    public static int maxRowIndex(double[][] m, int col) {
        int res = -1;
        double max = Double.MIN_VALUE;
        for (int row = 0; row < m.length; row++) {
            double v = m[row][col];
            if (v > max) {
                max = v;
                res = row;
            }
        }
        return res;
    }

    public static void zero(double[][] m) {
        for (int r = 0; r < m.length; r++) {
            for (int c = 0; c < m[r].length; c++) {
                m[r][c] = 0f;
            }
        }
    }

    public static double accuracy(double[][] outputs, double[][] expected) {
        int samples = outputs[0].length;
        int[] indices = new int[samples];
        Labels.toExpectedIndex(expected, indices);
        return accuracy(outputs, indices);
    }

    public static double accuracy(double[][] outputs, int[] expected) {
        int samples = outputs[0].length;
        int macthed = 0;
        for (int sample = 0; sample < samples; sample++) {
            if (expected[sample] == maxRowIndex(outputs, sample))
                macthed++;
        }
        return macthed / (double) samples;
    }

}