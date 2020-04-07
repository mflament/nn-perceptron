package org.yah.tests.perceptron.mt;

import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.base.SamplesSource;

/**
 * @author Yah
 */
public class MTSamplesSource implements SamplesSource<MTBatch> {

    private final MTNeuralNetwork network;

    public MTSamplesSource(MTNeuralNetwork network) {
        this.network = network;
    }

    @Override
    public MTTrainingSamples createInputs(SamplesProvider provider, int batchSize) {
        MTMatrix inputs = createInputs(provider);
        return new MTTrainingSamples(inputs, batchSize);
    }

    @Override
    public MTTrainingSamples createTraining(TrainingSamplesProvider provider, int batchSize) {
        MTMatrix inputs = createInputs(provider);
        int samples = provider.samples();
        int[] expectedIndices = new int[samples];
        for (int sample = 0; sample < samples; sample++) {
            int expectedIndex = provider.outputIndex(sample);
            expectedIndices[sample] = expectedIndex;
        }
        return new MTTrainingSamples(inputs, expectedIndices, batchSize);
    }

    private MTMatrix createInputs(SamplesProvider provider) {
        int samples = provider.samples();
        int features = network.features();
        MTMatrix inputs = new MTMatrix(features, samples);
        for (int sample = 0; sample < samples; sample++) {
            for (int feature = 0; feature < features; feature++) {
                inputs.set(sample, feature, provider.input(sample, feature));
            }
        }
        return inputs;
    }

}
