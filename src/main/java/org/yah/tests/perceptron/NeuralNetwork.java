package org.yah.tests.perceptron;

import java.util.Iterator;

public interface NeuralNetwork<B extends Batch> {

    int layers();

    int features();

    int outputs();

    int features(int layer);

    int neurons(int layer);
    
    BatchSource<B> createBatchSource();

    void propagate(B batch, int[] outputs);

    double evaluate(B batch, int[] outputs);

    default double evaluate(B batch) {
        return evaluate(batch, null);
    }

    double evaluate(Iterator<B> batches);

    void train(B batch, double learningRate);

    void train(Iterator<B> batches, double learningRate);

}