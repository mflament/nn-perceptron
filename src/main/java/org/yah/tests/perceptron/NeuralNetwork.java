package org.yah.tests.perceptron;

import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;

public interface NeuralNetwork extends NeuralNetworkState {

    NeuralNetworkState getState();

    InputSamples createInputs(SamplesProvider provider, int batchSize);

    TrainingSamples createTraining(TrainingSamplesProvider provider, int batchSize);

    NetworkOutputs createOutpus(int samples);

    void propagate(InputSamples samples, NetworkOutputs outputs);

    double evaluate(TrainingSamples samples, NetworkOutputs outputs);

    default double evaluate(TrainingSamples samples) {
        return evaluate(samples, null);
    }

    void train(TrainingSamples samples, double learningRate);

}