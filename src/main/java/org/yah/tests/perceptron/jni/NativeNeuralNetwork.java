/**
 * 
 */
package org.yah.tests.perceptron.jni;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Iterator;

import org.yah.tests.perceptron.Batch;
import org.yah.tests.perceptron.JavaNeuralNetwork;
import org.yah.tests.perceptron.NeuralNetwork;

/**
 * @author Yah
 *
 */
public class NativeNeuralNetwork implements NeuralNetwork {

    private final JavaNeuralNetwork network;

    private boolean dirty;

    private double accuracy = Double.NaN;

    public NativeNeuralNetwork(int... layerSizes) {
        network = new JavaNeuralNetwork(layerSizes);
    }

    @Override
    public double[][] weights(int layer) {
        return network.weights(layer);
    }

    @Override
    public double[] biases(int layer) {
        return network.biases(layer);
    }

    @Override
    public int layers() {
        return network.layers();
    }

    @Override
    public int features() {
        return network.features();
    }

    @Override
    public int outputs() {
        return network.outputs();
    }

    @Override
    public int features(int layer) {
        return network.features(layer);
    }

    @Override
    public int neurons(int layer) {
        return network.neurons(layer);
    }

    @Override
    public double accuracy() {
        return accuracy ;
    }

    @Override
    public void propagate(double[][] inputs, int[] outputs) {
        propagate(inputs, outputs);
    }

    @Override
    public double train(Iterator<Batch> batchIter, double learningRate) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public double train(Batch batch, double learningRate) {
        int size = Integer.BYTES;// layers count
        size += (network.layers() + 1) * Integer.BYTES; // layers sizes
        int layers = network.layers();
        for (int layer = 0; layer < layers; layer++) {
            // weights and biases per layer
            size += bytes(network.weights(layer));
            size += bytes(network.biases(layer));
        }
        size += bytes(batch.inputs);
        size += bytes(batch.expectedMatrix);
        ByteBuffer buffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
        buffer.putInt(layers);
        for (int layer = 0; layer < layers + 1; layer++) {
            buffer.putInt(network.features(layer));
        }
        int networkData = buffer.position();
        for (int layer = 0; layer < layers; layer++) {
            put(network.weights(layer), buffer);
            put(network.biases(layer), buffer);
        }
        put(batch.inputs, buffer);
        // put(batch.expectedMatrix, buffer);
        train(buffer, learningRate);

        buffer.position(networkData);
        for (int layer = 0; layer < layers; layer++) {
            get(buffer, network.weights(layer));
            get(buffer, network.biases(layer));
        }
    }

    private static ByteBuffer serializeNetwork(NeuralNetwork network) {
        int size = Integer.BYTES;// layers count
        size += (network.layers() + 1) * Integer.BYTES; // features + layers sizes
        int layers = network.layers();
        for (int layer = 0; layer < layers; layer++) {
            // weights and biases per layer
            size += bytes(network.weights(layer));
            size += bytes(network.biases(layer));
        }
        ByteBuffer buffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
        buffer.putInt(layers);
        for (int layer = 0; layer < layers + 1; layer++) {
            buffer.putInt(network.features(layer));
        }
        for (int layer = 0; layer < layers; layer++) {
            put(network.weights(layer), buffer);
            put(network.biases(layer), buffer);
        }
        buffer.flip();
        return buffer;
    }

    private static ByteBuffer serializeBatch(NeuralNetwork network) {

    }

    private static void get(ByteBuffer buffer, double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            get(buffer, matrix[i]);
        }
    }

    private static void get(ByteBuffer buffer, double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = buffer.getDouble();
        }
    }

    private static int bytes(double[][] matrix) {
        return matrix.length * bytes(matrix[0]);
    }

    private static int bytes(double[] v) {
        return v.length * Double.BYTES;
    }

    private static void put(double[][] matrix, ByteBuffer buffer) {
        for (int row = 0; row < matrix.length; row++) {
            put(matrix[row], buffer);
        }
    }

    private static void put(double[] v, ByteBuffer buffer) {
        for (int i = 0; i < v.length; i++) {
            buffer.putDouble(v[i]);
        }
    }

    private static native void network(ByteBuffer network);

    private static native void train(ByteBuffer batch, double learningRate);

    private static native void propagate(ByteBuffer inputs, ByteBuffer outputs);

}
