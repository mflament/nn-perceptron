/**
 * 
 */
package org.yah.tests.perceptron.jni;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.RandomUtils;
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

    private IntBuffer outputsBuffer;

    private final int layers;
    private final int[] layerSizes;

    public NativeNeuralNetwork(int... layerSizes) {
        if (layerSizes.length < 2)
            throw new IllegalArgumentException("Invalid layers counts " + layerSizes.length);
        this.layerSizes = layerSizes;
        this.layers = layerSizes.length - 1;
        if (RandomUtils.SEED >= 0)
            seed(RandomUtils.SEED);
        reference = create(layerSizes);
        if (reference == 0)
            throw new IllegalStateException("Error creating native neuralnetwork");
    }

    private IntBuffer getOutputsBuffer(int capacity) {
        if (outputsBuffer == null || outputsBuffer.capacity() < capacity) {
            outputsBuffer = ByteBuffer.allocateDirect(capacity * Integer.BYTES)
                    .order(ByteOrder.nativeOrder()).asIntBuffer();
        } else {
            outputsBuffer.position(0);
            outputsBuffer.limit(capacity);
        }
        return outputsBuffer;
    }

    @Override
    public int layers() {
        return layers;
    }

    @Override
    public int features() {
        return layerSizes[0];
    }

    @Override
    public int outputs() {
        return layerSizes[layers];
    }

    @Override
    public int features(int layer) {
        return layerSizes[layer];
    }

    @Override
    public int neurons(int layer) {
        return layerSizes[layer + 1];
    }

    @Override
    public void snapshot(int layer, DoubleBuffer buffer) {
        throw new UnsupportedOperationException();
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
        propagate(samples, IntBuffer.wrap(outputs));
    }

    @Override
    public void propagate(InputSamples samples, IntBuffer outputs) {
        NativeTrainingSamples nativeSamples = (NativeTrainingSamples) samples;
        IntBuffer buffer = outputs.isDirect() ? outputs : getOutputsBuffer(samples.size());
        propagate(reference, nativeSamples.reference, buffer);
        if (buffer != outputs) {
            outputs.put(buffer);
            outputs.flip();
        }
    }

    @Override
    public double evaluate(TrainingSamples samples, int[] outputs) {
        return evaluate(samples, outputs != null ? IntBuffer.wrap(outputs) : null);
    }

    @Override
    public double evaluate(TrainingSamples samples, IntBuffer outputs) {
        NativeTrainingSamples nativeSamples = (NativeTrainingSamples) samples;
        IntBuffer buffer = null;
        if (outputs != null)
            buffer = outputs.isDirect() ? outputs : getOutputsBuffer(samples.size());
        double accuracy = evaluate(reference, nativeSamples.reference, buffer);
        if (buffer != outputs) {
            outputs.put(buffer);
            outputs.flip();
        }
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

    private static native void seed(long seed);

}
