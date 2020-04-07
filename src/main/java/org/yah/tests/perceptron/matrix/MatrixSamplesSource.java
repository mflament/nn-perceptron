package org.yah.tests.perceptron.matrix;

import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.base.SamplesSource;

/**
 * Note: all inputs are expected to be column major. They can be transposed
 * using the corresponding parameter if necessary.
 *
 * @author Yah
 */
final class MatrixSamplesSource<M extends Matrix<M>> implements SamplesSource<MatrixBatch<M>> {

    private final MatrixNeuralNetwork<M> network;

    public MatrixSamplesSource(MatrixNeuralNetwork<M> network) {
        this.network = network;
    }

    @Override
    public MatrixSamples<M> createInputs(SamplesProvider provider, int batchSize) {
        M inputsMatrix = createInputs(provider);
        return new MatrixSamples<>(batchSize, inputsMatrix);
    }

    @Override
    public MatrixSamples<M> createTraining(TrainingSamplesProvider provider, int batchSize) {
        M inputsMatrix = createInputs(provider);
        checkExpecteds(provider, inputsMatrix.columns());
        return new MatrixSamples<>(batchSize, inputsMatrix, provider.createExpectedIndices());
    }

    private M createInputs(SamplesProvider provider) {
        M res = network.newMatrix(network.features(), provider.samples());
        res.apply((r, c, v) -> provider.input(c, r));
        return res;
    }

    private void checkExpecteds(TrainingSamplesProvider provider, int samples) {
        for (int i = 0; i < samples; i++) {
            int index = provider.outputIndex(i);
            if (index < 0 || index >= network.outputs())
                throw new IllegalArgumentException("Invalid expected index " + index);
        }
    }

}
