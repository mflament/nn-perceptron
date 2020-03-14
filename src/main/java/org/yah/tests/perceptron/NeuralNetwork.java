package org.yah.tests.perceptron;

import java.util.Iterator;

public interface NeuralNetwork<M extends Matrix<M>> {

    int layers();

    int features();

    int outputs();

    int features(int layer);

    int neurons(int layer);

    void propagate(M inputs, int[] outputs);

    double evaluate(Batch<M> batch, int[] outputs);

    default double evaluate(Batch<M> batch) {
        return evaluate(batch, null);
    }

    double evaluate(Iterator<Batch<M>> batches);

    void train(Batch<M> batch, double learningRate);

    void train(Iterator<Batch<M>> batches, double learningRate);

}