package org.yah.tests.perceptron.matrix;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;

import org.junit.Before;
import org.junit.Test;
import org.yah.tests.perceptron.SamplesProviders;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tests.perceptron.matrix.MatrixSamplesSource.MatrixBatch;
import org.yah.tests.perceptron.matrix.MatrixSamplesSource.MatrixSamples;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;

public class MatrixSamplesSourceTest {

    private static final double[][] INPUTS = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
    private static final double[][] TINPUTS = { { 1, 3, 5 }, { 2, 4, 6 } };
    private static final int[] EXPECTEDS = { 0, 2, 1 };

    private static final double[][] EXPECTED_OUTPUTS = { { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } };
    private static final double[][] EXPECTED_INDICES = { { 0 }, { 2 }, { 1 } };

    private TrainingSamplesProvider provider = SamplesProviders.newTrainingProvider(INPUTS, false,
            EXPECTEDS);

    private MatrixNeuralNetwork<CMArrayMatrix> network;
    private MatrixSamplesSource<CMArrayMatrix> source;

    @Before
    public void setup() {
        network = new MatrixNeuralNetwork<>(CMArrayMatrix::new, 2, 4, 3);
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
            assertMatrix(new double[][] { { 1, 2 }, { 3, 4 } }, batch.inputs());
            assertMatrix(new double[][] { { 1, 0, 0 }, { 0, 0, 1 } }, batch.expectedOutputs());
            assertMatrix(new double[][] { { 0 }, { 2 } }, batch.expectedIndices());

            assertTrue(iter.hasNext());
            batch = iter.next();

            assertEquals(1, batch.size());
            assertMatrix(new double[][] { { 5, 6 } }, batch.inputs());
            assertMatrix(new double[][] { { 0, 1, 0 } }, batch.expectedOutputs());
            assertMatrix(new double[][] { { 1 } }, batch.expectedIndices());

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
        assertMatrix(INPUTS, samples.inputs());
        assertMatrix(EXPECTED_OUTPUTS, samples.expectedOutputs());
        assertMatrix(EXPECTED_INDICES, samples.expectedIndices());
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
