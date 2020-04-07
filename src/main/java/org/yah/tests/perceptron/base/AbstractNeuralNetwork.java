package org.yah.tests.perceptron.base;

import org.yah.tests.perceptron.*;

public abstract class AbstractNeuralNetwork<O extends NetworkOutputs> implements NeuralNetwork {

    protected final NeuralNetworkState state;

    private boolean stateDirty, modelDirty;

    public AbstractNeuralNetwork(NeuralNetworkState state) {
        this.state = new DefaultNetworkState(state);
    }

    @Override
    public int layers() {
        return state.layers();
    }

    @Override
    public int features() {
        return state.features();
    }

    @Override
    public int outputs() {
        return state.outputs();
    }

    @Override
    public int features(int layer) {
        return state.features(layer);
    }

    @Override
    public int neurons(int layer) {
        return state.neurons(layer);
    }


    @Override
    public int maxNeurons() {
        return state.maxNeurons();
    }

    @Override
    public int maxFeatures() {
        return state.maxFeatures();
    }

    @Override
    public int totalNeurons() {
        return state.totalNeurons();
    }

    @Override
    public int totalWeights() {
        return state.totalWeights();
    }

    @Override
    public NeuralNetworkState getState() {
        return new DefaultNetworkState(this);
    }

    @Override
    public double weight(int layer, int neuron, int feature) {
        checkState();
        return state.weight(layer, neuron, feature);
    }

    @Override
    public double bias(int layer, int neuron) {
        checkState();
        return state.bias(layer, neuron);
    }

    @Override
    public void weight(int layer, int neuron, int feature, double weight) {
        state.weight(layer, neuron, feature, weight);
        stateChanged();
    }

    @Override
    public void bias(int layer, int neuron, double bias) {
        state.bias(layer, neuron, bias);
        stateChanged();
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + state.toString();
    }

    protected final void checkState() {
        if (stateDirty) {
            updateState();
            stateDirty = false;
        }
    }

    protected final void checkModel() {
        if (modelDirty) {
            updateModel();
            modelDirty = false;
        }
    }

    protected final void stateChanged() {
        modelDirty = true;
    }

    protected final void modelChanged() {
        stateDirty = true;
    }

    @SuppressWarnings("unchecked")
    @Override
    public final void propagate(InputSamples samples, NetworkOutputs outputs) {
        checkModel();
        doPropagate(samples, (O) outputs);
    }

    @SuppressWarnings("unchecked")
    @Override
    public final double evaluate(TrainingSamples samples, NetworkOutputs outputs) {
        checkModel();
        return doEvaluate(samples, (O) outputs);
    }

    @Override
    public final void train(TrainingSamples samples, double learningRate) {
        checkModel();
        doTrain(samples, learningRate);
        modelChanged();
    }

    @Override
    public abstract O createOutpus(int samples);

    protected abstract void updateState();

    protected abstract void updateModel();

    protected abstract void doPropagate(InputSamples samples, O outputs);

    protected abstract double doEvaluate(InputSamples samples, O outputs);

    protected abstract void doTrain(TrainingSamples samples, double learningRate);
}
