package org.yah.tests.perceptron;

import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.matrix.Matrix;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork.MatrixFactory;
import org.yah.tests.perceptron.matrix.MatrixSamplesSource;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;

public class NeuralNetworkSandbox<M extends Matrix<M>> {

    private static final long LOG_INTERVAL = 1000;
    private static final double NS_MS = 1E-6;

    private MatrixNeuralNetwork<M> network;
    private MatrixSamplesSource<M> samplesSource;

    private NeuralNetworkSandbox(MatrixFactory<M> matrixFactory, int... layerSizes) {
        network = new MatrixNeuralNetwork<>(matrixFactory, layerSizes);
        samplesSource = network.createSampleSource();
    }

    public void runXOR() throws InterruptedException {
        run(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new int[] { 0, 1, 1, 0 });
    }

    public void runNAND() throws InterruptedException {
        run(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new int[] { 1, 1, 1, 0 });
    }

    public void runAND() throws InterruptedException {
        run(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new int[] { 0, 0, 0, 1 });
    }

    public void runOR() throws InterruptedException {
        run(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new int[] { 0, 1, 1, 1 });
    }

    public void run(double[][] inputs, int[] outputIndices) throws InterruptedException {
        TrainingSamplesProvider provider = SamplesProviders.newTrainingProvider(inputs, false, outputIndices);
        TrainingSamples samples = samplesSource.createTraining(provider, 0);
        long start = System.nanoTime();
        System.out.println(network.evaluate(samples, null));
        int count = 0;
        while (true) {
            network.train(samples, 0.1f);
            count++;
            double elapsed = (System.nanoTime() - start) * NS_MS;
            if (elapsed > LOG_INTERVAL) {
                double score = network.evaluate(samples, null);
                System.out.println(
                        String.format("score: %.2f b/ms: %.3f", score, count / elapsed));
                count = 0;
                start = System.nanoTime();
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        NeuralNetworkSandbox<CMArrayMatrix> sb = new NeuralNetworkSandbox<>(CMArrayMatrix::new, 2,
                2,
                2);
        sb.runNAND();
    }
}
