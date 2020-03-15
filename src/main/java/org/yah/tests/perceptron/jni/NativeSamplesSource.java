package org.yah.tests.perceptron.jni;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.SamplesSource;

class NativeSamplesSource implements SamplesSource {

    private final NeuralNetwork network;

    public NativeSamplesSource(NeuralNetwork network) {
        this.network = network;
    }

    @Override
    public NativeTrainingSamples createInputs(SamplesProvider provider, int batchSize) {
        ByteBuffer inputsBuffer = createInputs(provider);
        return new NativeTrainingSamples(batchSize, inputsBuffer, null, null);
    }

    @Override
    public NativeTrainingSamples createTraining(TrainingSamplesProvider provider, int batchSize) {
        ByteBuffer inputsBuffer = createInputs(provider);
        checkExpecteds(provider);
        ByteBuffer outputsMatrix = createOutputs(provider);
        ByteBuffer expectedIndices = createExpectedIndices(provider);
        return new NativeTrainingSamples(batchSize, inputsBuffer, outputsMatrix, expectedIndices);
    }

    private ByteBuffer createInputs(SamplesProvider provider) {
        if (provider.features() != network.features()) {
            throw new IllegalArgumentException(
                    "Invalid inputs features: " + provider.features() + ", expected "
                            + network.features());
        }
        return NativeMatrix.create(provider.features(), provider.samples(),
                (r, c, v) -> provider.input(c, r));
    }

    private void checkExpecteds(TrainingSamplesProvider provider) {
        for (int i = 0; i < provider.samples(); i++) {
            int index = provider.outputIndex(i);
            if (index < 0 || index >= network.outputs())
                throw new IllegalArgumentException("Invalid expected index " + index);
        }
    }

    private ByteBuffer createOutputs(TrainingSamplesProvider provider) {
        return NativeMatrix.create(network.outputs(), provider.samples(),
                (r, c, v) -> provider.outputIndex(c) == r ? 1 : 0);
    }

    private ByteBuffer createExpectedIndices(TrainingSamplesProvider provider) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(provider.samples() * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int i = 0; i < provider.samples(); i++) {
            buffer.putInt(provider.outputIndex(i));
        }
        return buffer;
    }
}
