package org.yah.tests.perceptron;

import java.util.Iterator;

public interface NeuralNetwork {

    double[][] weights(int layer);

    double[] biases(int layer);

    int layers();

    int features();

    int outputs();

    int features(int layer);

    int neurons(int layer);

    void propagate(double[][] inputs, int[] outputs);

    double train(Batch batch, double learningRate);

    double train(Iterator<Batch> batchIter, double learningRate);

    double accuracy();

}