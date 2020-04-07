package org.yah.tests.perceptron.opencl;

import org.lwjgl.BufferUtils;
import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.base.BatchedSamples;
import org.yah.tests.perceptron.base.SamplesSource;
import org.yah.tools.opencl.mem.BufferProperties;
import org.yah.tools.opencl.mem.CLBuffer;

import java.nio.ByteBuffer;

/**
 * @author Yah
 */
class CLSamplesSource implements SamplesSource<CLTrainingBatch> {

    private final CLNeuralNetwork network;

    public CLSamplesSource(CLNeuralNetwork network) {
        this.network = network;
    }

    @Override
    public BatchedSamples<CLTrainingBatch> createInputs(SamplesProvider provider, int batchSize) {
        CLBuffer inputsBuffer = createInputsBuffer(provider);
        return new CLTrainingSamples(provider.samples(), batchSize, inputsBuffer);
    }

    @Override
    public BatchedSamples<CLTrainingBatch> createTraining(TrainingSamplesProvider provider, int batchSize) {
        CLBuffer inputsBuffer = createInputsBuffer(provider);

        int samples = provider.samples();
        ByteBuffer buffer = BufferUtils.createByteBuffer(samples * Integer.BYTES);
        for (int i = 0; i < samples; i++) {
            buffer.putInt(provider.outputIndex(i));
        }
        buffer.flip();
        CLBuffer expectedIndicesBuffer = network.environment.mem(buffer,
                BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_ONLY,
                BufferProperties.MEM_HOST_WRITE_ONLY);
        return new CLTrainingSamples(samples, batchSize, inputsBuffer, expectedIndicesBuffer);
    }

    private CLBuffer createInputsBuffer(SamplesProvider provider) {
        return network.createMatrixBuffer(network.features(), provider.samples(),
                (r, c) -> provider.input(c, r), BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_ONLY, BufferProperties.MEM_HOST_NO_ACCESS);
    }


}
