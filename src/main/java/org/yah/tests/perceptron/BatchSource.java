/**
 * 
 */
package org.yah.tests.perceptron;

/**
 * @author Yah
 */
public interface BatchSource<B extends Batch> {

    public interface TrainingSet<B extends Batch> extends Iterable<B> {
        int samples();

        int batchSize();

        default int batchCount() {
            return (int) Math.ceil(samples() / (double) batchSize());
        }
    }

    TrainingSet<B> createBatches(double[][] inputs, int[] expecteds, int batchSize,
            boolean transposeInputs);

    default TrainingSet<B> createBatches(double[][] inputs, int[] expecteds, int batchSize) {
        return createBatches(inputs, expecteds, batchSize, false);
    }

    B createBatch(double[][] inputs, boolean transposeInputs);

    B createBatch(double[][] inputs, int[] expecteds, boolean transposeInputs);

    default B createBatch(double[][] inputs, int[] expecteds) {
        return createBatch(inputs, expecteds, false);
    }
}
