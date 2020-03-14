/**
 * 
 */
package org.yah.tests.perceptron;

/**
 * @author Yah
 *
 */
public interface BatchSource<M extends Matrix<M>> {

    public interface TrainingSet<M extends Matrix<M>> extends Iterable<Batch<M>> {
        int samples();

        int batchSize();

        default int batchCount() {
            return (int) Math.ceil(samples() / (double) batchSize());
        }
    }

    TrainingSet<M> createBatches(double[][] inputs, int[] expecteds, int batchSize,
            boolean transposeInputs);

    default TrainingSet<M> createBatches(double[][] inputs, int[] expecteds, int batchSize) {
        return createBatches(inputs, expecteds, batchSize, false);
    }

    Batch<M> createBatch(double[][] inputs, int[] expecteds, boolean transposeInputs);

    default Batch<M> createBatch(double[][] inputs, int[] expecteds) {
        return createBatch(inputs, expecteds, false);
    }
}
