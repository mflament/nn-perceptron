package org.yah.tests.perceptron.matrix;

import org.junit.Before;
import org.junit.Test;
import org.yah.tests.perceptron.RandomUtils;
import org.yah.tests.perceptron.SamplesProviders;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tests.perceptron.base.DefaultNetworkState;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;

import java.util.Iterator;

import static org.junit.Assert.*;

public class MatrixSamplesSourceTest {

    private static final double[][] INPUTS = {{1, 2}, {3, 4}, {5, 6}};
    private static final double[][] TINPUTS = {{1, 3, 5}, {2, 4, 6}};
    private static final int[] EXPECTEDS = {0, 2, 1};

    private static final int[] EXPECTED_INDICES = {0, 2, 1};

    private final TrainingSamplesProvider provider = SamplesProviders.newTrainingProvider(INPUTS, false,
            EXPECTEDS);

    private MatrixSamplesSource<CMArrayMatrix> source;

    @Before
    public void setup() {
        MatrixNeuralNetwork<CMArrayMatrix> network = new MatrixNeuralNetwork<>(CMArrayMatrix::new,
                new DefaultNetworkState(RandomUtils.newRandomSource(), 2, 4, 3));
        source = new MatrixSamplesSource<>(network);
    }

    @Test
    public void testCreateTraining() {
        TrainingSamplesProvider provider = SamplesProviders.newTrainingProvider(INPUTS, false,
                EXPECTEDS);
        MatrixSamples<CMArrayMatrix> samples = source.createTraining(provider, 2);
        for (int i = 0; i < 2; i++) {
            Iterator<MatrixBatch<CMArrayMatrix>> iter = samples.iterator();
            assertTrue(iter.hasNext());
            MatrixBatch<?> batch = iter.next();

            assertEquals(2, batch.size());
            assertMatrix(new double[][]{{1, 2}, {3, 4}}, batch.inputs);
            assertEquals(0, batch.expectedIndex(0));
            assertEquals(2, batch.expectedIndex(1));

            assertTrue(iter.hasNext());
            batch = iter.next();

            assertEquals(1, batch.size());
            assertMatrix(new double[][]{{5, 6}}, batch.inputs);
            assertEquals(1, batch.expectedIndex(0));

            assertFalse(iter.hasNext());
        }

    }

    @Test
    public void testCreateInputs() {
        TrainingSamples samples = source.createInputs(provider, 0);
        assertEquals(3, samples.size());
        assertEquals(3, samples.batchSize());
        assertEquals(1, samples.batchCount());

        samples = source.createInputs(provider, 2);
        assertEquals(3, samples.size());
        assertEquals(2, samples.batchSize());
        assertEquals(2, samples.batchCount());
    }

    @Test
    public void testCreateTransposedInputs() {
        TrainingSamplesProvider tprovider = SamplesProviders.newTrainingProvider(TINPUTS,
                true, EXPECTEDS);
        MatrixSamples<CMArrayMatrix> samples = source.createTraining(tprovider, 3);
        assertMatrix(INPUTS, samples.inputs);
        assertArrayEquals(EXPECTED_INDICES, samples.expectedIndices);
    }

    private void assertMatrix(double[][] expected, Matrix<?> actual) {
        assertEquals(expected.length, actual.columns());
        assertEquals(expected[0].length, actual.rows());
        actual.apply((r, c, v) -> {
            assertEquals(expected[c][r], v, 0);
            return v;
        });
    }

}
