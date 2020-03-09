package org.yah.tests.perceptron;

import static org.yah.tests.perceptron.Matrix.transpose;

import org.yah.tests.perceptron.NeuralNetwork.Batch;

public class NeuralNetworkSandbox {

    private static final double[][] INPUTS = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    private static final int[] OUTPUTS = { 0, 1, 1, 0 };

    private static final int LOG_INTERVAL = 2000000;
    private static final double NS_S = 1E-9;

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 2, 2);
        Batch batch = new Batch(transpose(INPUTS), OUTPUTS, nn.outputs());
        double score = nn.evaluate(batch);
        System.out.println(score);
        int count = 0;
        long start = System.nanoTime();
        while (true) {
            nn.train(batch, 0.1f);
            count++;
            if (count == LOG_INTERVAL) {
                double elapsed = (System.nanoTime() - start) * NS_S;
                System.out.println(String.format("bps: %.3f", LOG_INTERVAL / elapsed));
                count = 0;
                start = System.nanoTime();
            }
            // score = nn.evaluate(batch);
            // System.out.println(score);
        }
    }
}
