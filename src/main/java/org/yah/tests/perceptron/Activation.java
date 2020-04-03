package org.yah.tests.perceptron;

public class Activation {

    public static double sigmoid(double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }

    public static double sigmoid_prime(double v) {
        double s = sigmoid(v);
        return s * (1.0 - s);
    }

    /** @noinspection unused*/
    public static double exp(double val) {
        final long tmp = (long) (1512775 * val + (1072693248 - 60801));
        return Double.longBitsToDouble(tmp << 32);
    }
    
}