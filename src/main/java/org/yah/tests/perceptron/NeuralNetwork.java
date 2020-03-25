package org.yah.tests.perceptron;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

public interface NeuralNetwork {

    int layers();

    int features();

    int outputs();

    int features(int layer);

    int neurons(int layer);

    SamplesSource createSampleSource();

    void propagate(InputSamples samples, int[] outputs);

    void propagate(InputSamples samples, IntBuffer outputs);

    default double evaluate(TrainingSamples samples) {
        return evaluate(samples, (IntBuffer) null);
    }

    double evaluate(TrainingSamples samples, int[] outputs);

    double evaluate(TrainingSamples samples, IntBuffer outputs);

    void train(TrainingSamples samples, double learningRate);

    void snapshot(int layer, DoubleBuffer buffer);

    default ByteBuffer snapshot() {
        int totalSize = 0;
        for (int layer = 0; layer < layers(); layer++) {
            totalSize += neurons(layer) * features(layer) + neurons(layer);
        }
        ByteBuffer res = ByteBuffer.allocate(totalSize * Double.BYTES);
        DoubleBuffer buffer = res.asDoubleBuffer();
        for (int layer = 0; layer < layers(); layer++) {
            snapshot(layer, buffer);
        }
        return res;
    }
}