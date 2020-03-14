package org.yah.tests.perceptron;

public class Activation {

    public static double sigmoid(double v) {
        return 1.0 / (1.0 + exp(-v));
    }

    public static double sigmoid_prime(double v) {
        double sv = sigmoid(v);
        return sv * (1.0 - sv);
    }

    public static double exp(double val) {
        final long tmp = (long) (1512775 * val + (1072693248 - 60801));
        return Double.longBitsToDouble(tmp << 32);
    }
}