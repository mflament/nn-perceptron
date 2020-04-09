package org.yah.tests.perceptron.jni;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryUtil;
import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.TrainingSamples;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

class NativeTrainingSamples implements TrainingSamples {

    private final int size;
    private final int features;
    private final int batchSize;
    final DoubleBuffer inputs;
    final IntBuffer expectedIndices;

    final ByteBuffer struct;

    NativeTrainingSamples(NativeNeuralNetwork network, SamplesProvider provider, int batchSize) {
        this.size = provider.samples();
        this.batchSize = batchSize == 0 ? size : batchSize;
        this.features = network.features();
        this.inputs = createInputs(network, provider);
        if (provider instanceof TrainingSamplesProvider) {
            TrainingSamplesProvider trainingProvider = (TrainingSamplesProvider) provider;
            checkExpecteds(network, trainingProvider);
            this.expectedIndices = createExpectedIndices(trainingProvider);
        } else
            this.expectedIndices = null;
        struct = serialize();
    }

    private ByteBuffer serialize() {
        int size = Integer.BYTES; // size
        size += Integer.BYTES; // batch size
        size += Integer.BYTES; // features

        if (PointerBuffer.POINTER_SIZE == 8)
            size += Integer.BYTES; // padding

        size += PointerBuffer.POINTER_SIZE; // inputs address
        size += PointerBuffer.POINTER_SIZE; // expected indices address


        ByteBuffer buffer = BufferUtils.createByteBuffer(size);
        buffer.putInt(size());
        buffer.putInt(batchSize);
        buffer.putInt(features);

        if (PointerBuffer.POINTER_SIZE == 8)
            buffer.putInt(0);

        PointerBuffer.put(buffer, MemoryUtil.memAddress(inputs));
        PointerBuffer.put(buffer, expectedIndices == null ? 0 : MemoryUtil.memAddress(expectedIndices));
        return buffer.flip();
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int batchSize() {
        return batchSize;
    }

    private DoubleBuffer createInputs(NativeNeuralNetwork network, SamplesProvider provider) {
        DoubleBuffer buffer = BufferUtils.createDoubleBuffer(provider.samples() * network.features());
        for (int sample = 0; sample < provider.samples(); sample++) {
            for (int feature = 0; feature < network.features(); feature++) {
                buffer.put(provider.input(sample, feature));
            }
        }
        return buffer.flip();
    }

    private void checkExpecteds(NativeNeuralNetwork network, TrainingSamplesProvider provider) {
        for (int i = 0; i < provider.samples(); i++) {
            int index = provider.outputIndex(i);
            if (index < 0 || index >= network.outputs())
                throw new IllegalArgumentException("Invalid expected index " + index);
        }
    }

    private IntBuffer createExpectedIndices(TrainingSamplesProvider provider) {
        IntBuffer buffer = BufferUtils.createIntBuffer(provider.samples());
        for (int sample = 0; sample < provider.samples(); sample++) {
            buffer.put(provider.outputIndex(sample));
        }
        return buffer.flip();
    }

}