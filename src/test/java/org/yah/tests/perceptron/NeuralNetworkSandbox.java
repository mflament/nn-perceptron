package org.yah.tests.perceptron;

import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.jni.NativeNeuralNetwork;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;
import org.yah.tests.perceptron.mt.MTNeuralNetwork;
import org.yah.tests.perceptron.opencl.CLNeuralNetwork;

public class NeuralNetworkSandbox implements AutoCloseable {

    static {
        Runtime.getRuntime().loadLibrary("neuralnetwork");
    }

    private static final long LOG_INTERVAL = 1000;
    private static final double NS_MS = 1E-6;

    private NeuralNetwork network;
    private SamplesSource samplesSource;

    private NeuralNetworkSandbox(NeuralNetwork network) {
        this.network = network;
        samplesSource = network.createSampleSource();
    }

    @Override
    public void close() throws Exception {
        if (network instanceof AutoCloseable)
            ((AutoCloseable) network).close();
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
        TrainingSamplesProvider provider = SamplesProviders.newTrainingProvider(inputs, false,
                outputIndices);
        TrainingSamples samples = null;
        try {
            samples = samplesSource.createTraining(provider, 2);
            long start = System.nanoTime();
            double last = network.evaluate(samples);
            System.out.println(last);
            int count = 0;
            while (true) {
                network.train(samples, 0.1f);
                double s = network.evaluate(samples);
                if (s != last) {
                    System.out.println(s);
                    last = s;
                }
                count++;
                double elapsed = (System.nanoTime() - start) * NS_MS;
                if (elapsed > LOG_INTERVAL) {
                    double score = network.evaluate(samples);
                    System.out.println(
                            String.format("score: %.2f b/ms: %.3f", score, count / elapsed));
                    count = 0;
                    start = System.nanoTime();
                }
            }
        } finally {
            if (samples instanceof AutoCloseable) {
                try {
                    ((AutoCloseable) samples).close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        int[] layers = new int[] { 2, 2, 2 };
        NeuralNetwork network = null;
        if (args.length > 0) {
            switch (args[0]) {
            case "native":
                network = new NativeNeuralNetwork(layers);
                break;
            case "cl":
                network = new CLNeuralNetwork(layers);
                break;
            case "mt":
                network = new MTNeuralNetwork(layers);
                break;
            case "matrix":
                network = new MatrixNeuralNetwork<>(CMArrayMatrix::new, layers);
            }
        }
        if (network == null)
            network = new MatrixNeuralNetwork<>(CMArrayMatrix::new, layers);

        try (NeuralNetworkSandbox sb = new NeuralNetworkSandbox(network)) {
            sb.runNAND();
        }
    }
}
