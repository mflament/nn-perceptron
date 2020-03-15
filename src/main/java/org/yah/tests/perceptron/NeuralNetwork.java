package org.yah.tests.perceptron;

public interface NeuralNetwork {

    int layers();

    int features();

    int outputs();

    int features(int layer);

    int neurons(int layer);
    
    SamplesSource createSampleSource();

    void propagate(InputSamples samples, int[] outputs);

    double evaluate(TrainingSamples samples, int[] outputs);

    void train(TrainingSamples samples, double learningRate);

}