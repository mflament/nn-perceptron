package org.yah.tests.perceptron.base;

import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.TrainingSamples;

/**
 * @author Yah
 */
public interface SamplesSource<B extends TrainingBatch> {

    BatchedSamples<B> createInputs(SamplesProvider provider, int batchSize);

    BatchedSamples<B> createTraining(TrainingSamplesProvider provider, int batchSize);

}
