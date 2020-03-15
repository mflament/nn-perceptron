package org.yah.tests.perceptron.jni;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.junit.Before;
import org.junit.Test;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.SamplesProviders;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;

public class NativeSamplesSourceTest {

    private static final double[][] INPUTS = { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } };
    private static final int[] OUTPUT_INDICES = { 1, 0, 2, 1, 2 };

    private static final TrainingSamplesProvider PROVIDER = SamplesProviders
            .newTrainingProvider(INPUTS, true, OUTPUT_INDICES);

    private NativeSamplesSource source;

    @Before
    public void setup() {
        NeuralNetwork nn = mock(NeuralNetwork.class);
        when(nn.features()).thenReturn(2);
        when(nn.outputs()).thenReturn(3);
        source = new NativeSamplesSource(nn);
    }

    @Test
    public void testCreateInputs() {
        NativeTrainingSamples samples = source.createInputs(PROVIDER, 2);
        assertEquals(5, samples.size());
        assertEquals(2, samples.batchSize());
        assertEquals(3, samples.batchCount());
        samples.delete();
    }

    @Test
    public void testCreateTraining() {
        NativeTrainingSamples samples = source.createTraining(PROVIDER, 2);
        assertEquals(5, samples.size());
        assertEquals(2, samples.batchSize());
        assertEquals(3, samples.batchCount());
        samples.delete();
    }

}
