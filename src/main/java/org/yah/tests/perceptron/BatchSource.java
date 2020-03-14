/**
 * 
 */
package org.yah.tests.perceptron;

/**
 * @author Yah
 *
 */
public interface BatchSource<M extends Matrix<M>> {

    Iterable<Batch<M>> createBatches(double[][] inputs, int[] expecteds, int batchSize,
            boolean transposeInputs);

    default Iterable<Batch<M>> createBatches(double[][] inputs, int[] expecteds, int batchSize) {
        return createBatches(inputs, expecteds, batchSize, false);
    }

    Batch<M> createBatch(double[][] inputs, int[] expecteds, boolean transposeInputs);

    default Batch<M> createBatch(double[][] inputs, int[] expecteds) {
        return createBatch(inputs, expecteds, false);
    }
}
