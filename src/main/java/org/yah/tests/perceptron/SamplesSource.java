/**
 * 
 */
package org.yah.tests.perceptron;

import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;

/**
 * @author Yah
 */
public interface SamplesSource {

    InputSamples createInputs(SamplesProvider provider, int batchSize);

    TrainingSamples createTraining(TrainingSamplesProvider provider, int batchSize);

}
