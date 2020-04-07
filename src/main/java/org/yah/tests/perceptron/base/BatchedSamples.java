package org.yah.tests.perceptron.base;

import org.yah.tests.perceptron.TrainingSamples;

public interface BatchedSamples<B extends  TrainingBatch> extends TrainingSamples, Iterable<B> {
}
