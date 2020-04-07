package org.yah.tests.perceptron.base;

import org.yah.tests.perceptron.*;

public abstract class AbstractBatchedNeuralNetwork<B extends TrainingBatch, O extends NetworkOutputs>
        extends AbstractNeuralNetwork<O> {

    protected SamplesSource<B> samplesSource;

    public AbstractBatchedNeuralNetwork(NeuralNetworkState state) {
        super(state);
    }

    @Override
    public InputSamples createInputs(SamplesProviders.SamplesProvider provider, int batchSize) {
        SamplesSource<B> source = getSamplesSource();
        return source.createInputs(provider, batchSize);
    }

    @Override
    public TrainingSamples createTraining(SamplesProviders.TrainingSamplesProvider provider, int batchSize) {
        SamplesSource<B> source = getSamplesSource();
        return source.createTraining(provider, batchSize);
    }

    @SuppressWarnings("unchecked")
    @Override
    protected void doPropagate(InputSamples samples, O outputs) {
        assert outputs != null && outputs.samples() == samples.size();
        checkModel();
        for (B batch : (BatchedSamples<B>) samples) {
            propagate(batch, outputs);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    protected double doEvaluate(InputSamples samples, O outputs) {
        assert outputs == null || outputs.samples() == samples.size();
        checkModel();
        int matched = 0;
        for (B batch : (BatchedSamples<B>) samples) {
            matched += evaluate(batch, outputs);
        }
        return matched / (double) samples.size();
    }

    @SuppressWarnings("unchecked")
    @Override
    protected void doTrain(TrainingSamples samples, double learningRate) {
        checkModel();
        for (B batch : (BatchedSamples<B>) samples) {
            train(batch, learningRate);
        }
        modelChanged();
    }

    protected abstract SamplesSource<B> createSampleSource();

    private synchronized SamplesSource<B> getSamplesSource() {
        if (samplesSource == null)
            samplesSource = createSampleSource();
        return samplesSource;
    }

    protected abstract void propagate(B batch, O outputs);

    protected abstract int evaluate(B batch, O outputs);

    protected abstract void train(B batch, double learningRate);
}
