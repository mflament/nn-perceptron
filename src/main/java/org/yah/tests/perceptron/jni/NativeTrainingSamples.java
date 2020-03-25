package org.yah.tests.perceptron.jni;

import java.nio.Buffer;

import org.yah.tests.perceptron.TrainingSamples;

class NativeTrainingSamples extends NativeObject implements TrainingSamples {

    private final int size;
    private final int batchSize;
    private Buffer inputsMatrix, outputsMatrix, outputIndices;

    public NativeTrainingSamples(int size, int batchSize, Buffer inputsMatrix, Buffer outputsMatrix,
            Buffer outputIndices) {
        this.size = size;
        this.batchSize = batchSize == 0 ? size : batchSize;
        this.inputsMatrix = inputsMatrix;
        this.outputsMatrix = outputsMatrix;
        this.outputIndices = outputIndices;
        this.reference = create(batchSize, inputsMatrix, outputsMatrix, outputIndices);
        if (reference == 0)
            throw new IllegalStateException("Error creating NativeTrainingSamples");
    }

    @Override
    public void close() {
        super.close();
        inputsMatrix = null;
        outputsMatrix = null;
        outputIndices = null;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int batchSize() {
        return batchSize;
    }

    private static native long create(int batchSize, Buffer inputsMatrix,
            Buffer outputsMatrix, Buffer outputIndices);

    @Override
    protected native void delete(long reference);

    private static native int size(long reference);

    private static native int batchSize(long reference);

}