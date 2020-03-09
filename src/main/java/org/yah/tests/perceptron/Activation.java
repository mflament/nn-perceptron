package org.yah.tests.perceptron;

public class Activation {

    public static void sigmoid(double[][] in) {
        sigmoid(in, in);
    }

    public static void sigmoid(double[][] in, double[][] out) {
        for (int r = 0; r < in.length; r++) {
            for (int c = 0; c < in[r].length; c++) {
                out[r][c] = sigmoid(in[r][c]);
            }
        }
    }

    public static double sigmoid(double v) {
        return 1.0 / (1.0 + exp(-v));
    }

    public static double sigmoid_prime(double v) {
        double sv = sigmoid(v);
        return sv * (1.0f - sv);
    }

    public static double exp(double val) {
        final long tmp = (long) (1512775 * val + (1072693248 - 60801));
        return Double.longBitsToDouble(tmp << 32);
    }
}