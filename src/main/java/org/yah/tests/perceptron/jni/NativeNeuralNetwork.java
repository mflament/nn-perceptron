package org.yah.tests.perceptron.jni;

import org.lwjgl.BufferUtils;
import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.NeuralNetworkState;
import org.yah.tests.perceptron.SamplesProviders.SamplesProvider;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tests.perceptron.base.AbstractNeuralNetwork;
import org.yah.tests.perceptron.base.DirectBufferOutputs;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

/**
 * @author Yah
 */
public class NativeNeuralNetwork extends AbstractNeuralNetwork<DirectBufferOutputs>
        implements AutoCloseable {

    static {
        Runtime.getRuntime().loadLibrary("neuralnetwork");
    }

    private long reference;

    private final ByteBuffer stateBuffer;

    public NativeNeuralNetwork(NeuralNetworkState state) {
        super(state);
        stateBuffer = createStateBuffer(state);
        reference = create(stateBuffer);
        if (reference == 0)
            throw new IllegalStateException("Error creating native neuralnetwork");
    }

    @Override
    public void close() {
        if (reference != 0) {
            delete(reference);
            reference = 0;
        }
    }

    @Override
    public InputSamples createInputs(SamplesProvider provider, int batchSize) {
        return new NativeTrainingSamples(this, provider, batchSize);
    }

    @Override
    public TrainingSamples createTraining(TrainingSamplesProvider provider, int batchSize) {
        return new NativeTrainingSamples(this, provider, batchSize);
    }

    @Override
    public DirectBufferOutputs createOutpus(int samples) {
        return new DirectBufferOutputs(samples);
    }


    @Override
    protected void doPropagate(InputSamples samples, DirectBufferOutputs outputs) {
        propagate((NativeTrainingSamples) samples, outputs);
    }

    @Override
    protected double doEvaluate(InputSamples samples, DirectBufferOutputs outputs) {
        return evaluate((NativeTrainingSamples) samples, outputs);
    }

    @Override
    protected void doTrain(TrainingSamples samples, double learningRate) {
        train((NativeTrainingSamples) samples, learningRate);
    }

    private void propagate(NativeTrainingSamples samples, DirectBufferOutputs outputs) {
        propagate(reference, samples.struct, outputs.buffer());
    }

    private double evaluate(NativeTrainingSamples samples, DirectBufferOutputs outputs) {
        return evaluate(reference, samples.struct, outputs != null ? outputs.buffer() : null);
    }

    private void train(NativeTrainingSamples samples, double learningRate) {
        train(reference, samples.struct, learningRate);
        modelChanged();
    }

    @Override
    protected void updateState() {
        stateBuffer.position((2 + layers()) * Integer.BYTES);
        visitWeights((layer, neuron, feature) -> weight(layer, neuron, feature, stateBuffer.getDouble()));
        visitBiases((layer, neuron) -> bias(layer, neuron, stateBuffer.getDouble()));
        stateBuffer.position(0);
    }

    @Override
    protected void updateModel() {
        stateBuffer.position((2 + layers()) * Integer.BYTES);
        visitWeights((layer, neuron, feature) -> stateBuffer.putDouble(weight(layer, neuron, feature)));
        visitBiases((layer, neuron) -> stateBuffer.putDouble(bias(layer, neuron)));
        stateBuffer.position(0);
    }

    private static ByteBuffer createStateBuffer(NeuralNetworkState state) {
        int size = Integer.BYTES; // layers count
        size += (state.layers() + 1) * Integer.BYTES; // layer sizes
        size += state.totalWeights() * Double.BYTES; // weights
        size += state.totalNeurons() * Double.BYTES; // neurons
        ByteBuffer buffer = BufferUtils.createByteBuffer(size);
        buffer.putInt(state.layers());
        buffer.putInt(state.features());
        for (int layer = 0; layer < state.layers(); layer++) {
            buffer.putInt(state.neurons(layer));
        }
        for (int layer = 0; layer < state.layers(); layer++) {
            int neurons = state.neurons(layer);
            int features = state.features(layer);
            for (int feature = 0; feature < features; feature++) {
                for (int neuron = 0; neuron < neurons; neuron++) {
                    buffer.putDouble(state.weight(layer, neuron, feature));
                }
            }
        }
        for (int layer = 0; layer < state.layers(); layer++) {
            int neurons = state.neurons(layer);
            for (int neuron = 0; neuron < neurons; neuron++) {
                buffer.putDouble(state.bias(layer, neuron));
            }
        }
        return buffer.flip();
    }

    protected native void delete(long networkReference);

    private static native long create(ByteBuffer state);

    private static native void propagate(long networkReference, ByteBuffer samples, IntBuffer outputs);

    private static native double evaluate(long networkReference, ByteBuffer samples, IntBuffer outputs);

    private static native void train(long networkReference, ByteBuffer samples, double learningRate);

}
