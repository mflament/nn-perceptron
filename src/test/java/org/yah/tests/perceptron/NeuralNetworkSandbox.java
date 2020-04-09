package org.yah.tests.perceptron;

import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.base.DefaultNetworkState;
import org.yah.tests.perceptron.jni.NativeNeuralNetwork;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;
import org.yah.tests.perceptron.mt.MTNeuralNetwork;
import org.yah.tests.perceptron.opencl.CLNeuralNetwork;

import java.io.IOException;

public class NeuralNetworkSandbox {

    private static final long LOG_INTERVAL = 1000;
    private static final double NS_MS = 1E-6;

    static final double[][] INPTUS = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    static final TrainingSamplesProvider NAND = SamplesProviders.newTrainingProvider(
            INPTUS, false,
            new int[]{1, 1, 1, 0});
    static final TrainingSamplesProvider XOR = SamplesProviders.newTrainingProvider(
            INPTUS, false,
            new int[]{0, 1, 1, 0});
    static final TrainingSamplesProvider AND = SamplesProviders.newTrainingProvider(
            INPTUS, false,
            new int[]{0, 0, 0, 1});
    static final TrainingSamplesProvider OR = SamplesProviders.newTrainingProvider(
            INPTUS, false,
            new int[]{0, 1, 1, 1});

    /**
     * @noinspection InfiniteLoopStatement
     */
    public void run(NeuralNetwork network, TrainingSamplesProvider provider) {
        TrainingSamples samples = null;
        try {
            samples = network.createTraining(provider, 2);
            double last = network.evaluate(samples);
            long trainingTime = 0;
            long evaluationTime = 0;
            System.out.println("score: " + last);
            int count = 0;
            long lastLog = System.nanoTime();
            while (true) {
                long start = System.nanoTime();
                network.train(samples, 0.1);
                trainingTime += System.nanoTime() - start;

                start = System.nanoTime();
                double s = network.evaluate(samples);
                evaluationTime += System.nanoTime() - start;
                if (s != last) {
                    System.out.println(s);
                    last = s;
                }
                count++;
                double elapsed = (System.nanoTime() - lastLog) * NS_MS;
                if (elapsed > LOG_INTERVAL) {
                    double score = network.evaluate(samples);
                    double trainingMs = (trainingTime / (double) count) * NS_MS;
                    double evaluationMs = (evaluationTime / (double) count) * NS_MS;
                    System.out.println(
                            String.format("score: %.2f training: %.3fms; evaluation: %.3fms", score, trainingMs,
                                    evaluationMs));
                    count = 0;
                    trainingTime = evaluationTime = 0;
                    lastLog = System.nanoTime();
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
        // dump(new MatrixNeuralNetwork<>(CMArrayMatrix::new, new DefaultNetworkState(RandomUtils.newRandomSource(12356), 2, 3, 3)));
        run(args);
        //compare();
    }

    private static void run(String[] args) throws Exception {
        NeuralNetworkState state = new DefaultNetworkState(RandomUtils.newRandomSource(), 2, 2, 2);
        NeuralNetwork network = null;
        String type = args.length > 0 ? args[0] : "matrix";
        try {
            switch (args[0]) {
                case "native":
                    network = new NativeNeuralNetwork(state);
                    break;
                case "cl":
                    network = new CLNeuralNetwork(state);
                    break;
                case "mt":
                    network = new MTNeuralNetwork(state);
                    break;
                case "matrix":
                    network = new MatrixNeuralNetwork<>(CMArrayMatrix::new, state);
                    break;
                default:
                    throw new IllegalArgumentException("Invalid type " + type);
            }
            new NeuralNetworkSandbox().run(network, NAND);
        } finally {
            if (network instanceof AutoCloseable)
                ((AutoCloseable) network).close();
        }
    }
}
