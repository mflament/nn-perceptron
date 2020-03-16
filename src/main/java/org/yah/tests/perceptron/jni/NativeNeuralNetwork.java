/**
 * 
 */
package org.yah.tests.perceptron.jni;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;

/**
 * @author Yah
 *
 */
public class NativeNeuralNetwork extends NativeObject implements NeuralNetwork {

    static {
        Runtime.getRuntime().loadLibrary("neuralnetwork");
    }

    private final ThreadLocal<IntBuffer> outputsBuffers = new ThreadLocal<>();

    public NativeNeuralNetwork(int... layerSizes) {
        reference = create(layerSizes);
        if (reference == 0)
            throw new IllegalStateException("Error creating native neuralnetwork");
    }

    private IntBuffer getOutputsBuffer(int capacity) {
        IntBuffer buffer = outputsBuffers.get();
        if (buffer == null || buffer.capacity() < capacity) {
            buffer = ByteBuffer.allocateDirect(capacity * Integer.BYTES)
                    .order(ByteOrder.nativeOrder()).asIntBuffer();
            outputsBuffers.set(buffer);
        } else {
            buffer.position(0);
            buffer.limit(capacity);
        }
        return buffer;
    }

    @Override
    public int layers() {
        return layers(reference);
    }

    @Override
    public int features() {
        return features(reference);
    }

    @Override
    public int outputs() {
        return outputs(reference);
    }

    @Override
    public int features(int layer) {
        return features(reference, layer);
    }

    @Override
    public int neurons(int layer) {
        return neurons(reference, layer);
    }

    @Override
    public SamplesSource createSampleSource() {
        return new NativeSamplesSource(this);
    }

    @Override
    public String toString() {
        int layers = layers();
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int l = 0; l < layers + 1; l++) {
            sb.append(features(l));
            if (l < layers)
                sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public void propagate(InputSamples samples, int[] outputs) {
        NativeTrainingSamples nativeSamples = (NativeTrainingSamples) samples;
        IntBuffer outputsBuffer = getOutputsBuffer(outputs.length);
        propagate(reference, nativeSamples.reference, outputsBuffer);
        outputsBuffer.get(outputs);
    }

    @Override
    public double evaluate(TrainingSamples samples, int[] outputs) {
        NativeTrainingSamples nativeSamples = (NativeTrainingSamples) samples;
        IntBuffer outputsBuffer = outputs != null ? getOutputsBuffer(outputs.length) : null;
        double accuracy = evaluate(reference, nativeSamples.reference, outputsBuffer);
        if (outputsBuffer != null)
            outputsBuffer.get(outputs);
        return accuracy;
    }

    @Override
    public void train(TrainingSamples samples, double learningRate) {
        NativeTrainingSamples nativeSamples = (NativeTrainingSamples) samples;
        train(reference, nativeSamples.reference, learningRate);
    }

    @Override
    protected native void delete(long networkReference);

    private static native long create(int[] layerSizes);

    private static native int layers(long networkReference);

    private static native int features(long networkReference);

    private static native int outputs(long networkReference);

    private static native int features(long networkReference, int layer);

    private static native int neurons(long networkReference, int layer);

    private static native void propagate(long networkReference, long samplesReference,
            IntBuffer outputs);

    private static native double evaluate(long networkReference, long samplesReference,
            IntBuffer outputs);

    private static native void train(long networkReference, long samplesReference,
            double learningRate);
}
