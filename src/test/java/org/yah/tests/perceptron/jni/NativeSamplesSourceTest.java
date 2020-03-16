package org.yah.tests.perceptron.jni;

import static org.junit.Assert.assertEquals;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.yah.tests.perceptron.SamplesProviders;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;

public class NativeSamplesSourceTest {

    private static final double[][] INPUTS = { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } };
    private static final int[] OUTPUT_INDICES = { 1, 0, 2, 1, 2 };

    private static final TrainingSamplesProvider PROVIDER = SamplesProviders
            .newTrainingProvider(INPUTS, true, OUTPUT_INDICES);

    private NativeNeuralNetwork network;
    private NativeSamplesSource source;

    static {
        Runtime.getRuntime().loadLibrary("neuralnetwork");
    }

    @Before
    public void setup() {
        network = new NativeNeuralNetwork(2, 3);
        source = new NativeSamplesSource(network);
    }

    @After
    public void delete() {
        network.close();
    }

    @Test
    public void testCreateInputs() {
        try (NativeTrainingSamples samples = source.createInputs(PROVIDER, 2)) {
            assertEquals(5, samples.size());
            assertEquals(2, samples.batchSize());
            assertEquals(3, samples.batchCount());
        }

        try (NativeTrainingSamples samples = source.createInputs(PROVIDER, 0)) {
            assertEquals(5, samples.size());
            assertEquals(5, samples.batchSize());
            assertEquals(1, samples.batchCount());
        }
    }

    @Test
    public void testCreateTraining() {
        try (NativeTrainingSamples samples = source.createInputs(PROVIDER, 2)) {
            assertEquals(5, samples.size());
            assertEquals(2, samples.batchSize());
            assertEquals(3, samples.batchCount());
        }
    }

}
