package org.yah.tests.perceptron;

import org.yah.tests.perceptron.MatrixNeuralNetwork.MatrixFactory;
import org.yah.tests.perceptron.array.ArrayMatrix;

public class NeuralNetworkSandbox<M extends Matrix<M>> {

    private static final double[][] INPUTS = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    private static final int[] OUTPUTS = { 0, 1, 1, 0 };

    private static final int LOG_INTERVAL = 2000000;
    private static final double NS_MS = 1E-6;

    private MatrixNeuralNetwork<M> network;
    private BatchSource<M> batchSource;

    private NeuralNetworkSandbox(MatrixFactory<M> matrixFactory, int... layerSizes) {
        network = new MatrixNeuralNetwork<>(matrixFactory, layerSizes);
        batchSource = network.createBatchSource();
    }

    public void run() throws InterruptedException {
        Batch<M> batch = batchSource.createBatch(INPUTS, OUTPUTS, true);
        long start = System.nanoTime();
        System.out.println(network.evaluate(batch));
        int count = 0;
        while (true) {
            double score = network.train(batch, 0.1f);
            count++;
            if (count == LOG_INTERVAL) {
                double elapsed = (System.nanoTime() - start) * NS_MS;
                System.out.println(
                        String.format("score: %.2f b/ms: %.3f", score, LOG_INTERVAL / elapsed));
                count = 0;
                start = System.nanoTime();
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        NeuralNetworkSandbox<ArrayMatrix> sb = new NeuralNetworkSandbox<>(ArrayMatrix::new, 2, 2, 2);
        sb.run();
    }
}
