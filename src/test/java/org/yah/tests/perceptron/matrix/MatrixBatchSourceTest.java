package org.yah.tests.perceptron.matrix;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;

import org.junit.Before;
import org.junit.Test;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;

public class MatrixBatchSourceTest {

    private static final double[][] INPUTS = { { 1, 2, 3 }, { 4, 5, 6 } };
    private static final double[][] TINPUTS = { { 1, 4 }, { 2, 5 }, { 3, 6 } };
    private static final int[] EXPECTEDS = { 0, 2, 1 };

    private static final double[][] EXPECTED_OUTPUTS = { { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } };
    private static final double[][] EXPECTED_INDICES = { { 0, 2, 1 } };

    private MatrixNeuralNetwork<CMArrayMatrix> network;
    private MatrixBatchSource<CMArrayMatrix> source;

    @Before
    public void setup() {
        network = new MatrixNeuralNetwork<>(CMArrayMatrix::new, 2, 4, 3);
        source = new MatrixBatchSource<>(network);
    }

    @Test
    public void testCreateBatches() {
        Iterable<MatrixBatch<CMArrayMatrix>> batches = source.createBatches(INPUTS, EXPECTEDS, 2);
        for (int i = 0; i < 2; i++) {
            Iterator<MatrixBatch<CMArrayMatrix>> iter = batches.iterator();
            assertTrue(iter.hasNext());
            MatrixBatch<CMArrayMatrix> batch = iter.next();

            assertEquals(2, batch.size());
            assertMatrix(new double[][] { { 1, 2 }, { 4, 5 } }, batch.inputs());
            assertMatrix(new double[][] { { 1, 0 }, { 0, 0 }, { 0, 1 } }, batch.expectedOutputs());
            assertMatrix(new double[][] { { 0, 2 } }, batch.expectedIndices());

            assertTrue(iter.hasNext());
            batch = iter.next();

            assertEquals(1, batch.size());
            assertMatrix(new double[][] { { 3 }, { 6 } }, batch.inputs());
            assertMatrix(new double[][] { { 0 }, { 1 }, { 0 } }, batch.expectedOutputs());
            assertMatrix(new double[][] { { 1 } }, batch.expectedIndices());

            assertFalse(iter.hasNext());
        }
    }

    @Test
    public void testCreateBatch() {
        MatrixBatch<CMArrayMatrix> batch = source.createBatch(INPUTS, EXPECTEDS);
        assertEquals(3, batch.size());
        assertMatrix(INPUTS, batch.inputs());
        assertMatrix(EXPECTED_OUTPUTS, batch.expectedOutputs());
        assertMatrix(EXPECTED_INDICES, batch.expectedIndices());
    }

    @Test
    public void testCreateTransposedBatch() {
        MatrixBatch<CMArrayMatrix> batch = source.createBatch(TINPUTS, EXPECTEDS, true);
        assertEquals(3, batch.size());
        assertMatrix(INPUTS, batch.inputs());
        assertMatrix(EXPECTED_OUTPUTS, batch.expectedOutputs());
        assertMatrix(EXPECTED_INDICES, batch.expectedIndices());
    }

    private void assertMatrix(double[][] expected, Matrix<?> actual) {
        assertEquals(expected.length, actual.rows());
        assertEquals(expected[0].length, actual.columns());
        actual.apply((r, c, v) -> {
            assertEquals(expected[r][c], v, 0);
            return v;
        });
    }

}
