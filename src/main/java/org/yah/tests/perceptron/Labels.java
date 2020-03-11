package org.yah.tests.perceptron;

public class Labels {
    public static double[][] toExpectedMatrix(int[] indices, int indexOffset, int outputs) {
        double[][] m = new double[outputs][indices.length];
        toExpectedMatrix(indices, indexOffset, m);
        return m;
    }

    public static void toExpectedMatrix(int[] indices, int indexOffset, double[][] m) {
        for (int i = indexOffset, col = 0; i < indices.length
                && col < m[0].length; i++, col++) {
            int index = indices[i];
            for (int row = 0; row < m.length; row++) {
                m[row][col] = index == row ? 1 : 0;
            }
        }
    }

    public static void toExpectedIndex(double[][] m, int[] indices) {
        for (int i = 0; i < m[0].length; i++) {
            indices[i] = Matrix.maxRowIndex(m, i);
        }
    }

    public static int countMatched(int[] expected, int[] actuals) {
        assert expected.length == actuals.length;
        int matched = 0;
        for (int i = 0; i < expected.length; i++) {
            if (actuals[i] == expected[i])
                matched++;
        }
        return matched;
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
            if (expected[sample] == Matrix.maxRowIndex(outputs, sample))
                macthed++;
        }
        return macthed / (double) samples;
    }
    
    public static double accuracy(int[] outputs, int[] expected) {
        int matched = 0;
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == outputs[i]) matched++;
        }
        return matched / outputs.length;
    }

}