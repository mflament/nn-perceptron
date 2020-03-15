package org.yah.tests.perceptron.jni;

import java.nio.Buffer;

import org.yah.tests.perceptron.TrainingSamples;

class NativeTrainingSamples implements TrainingSamples, NativeObject {
    private long reference;
    private Buffer inputsMatrix, outputsMatrix, outputIndices;

    public NativeTrainingSamples(int batchSize, Buffer inputsMatrix, Buffer outputsMatrix,
            Buffer outputIndices) {
        this.inputsMatrix = inputsMatrix;
        this.outputsMatrix = outputsMatrix;
        this.outputIndices = outputIndices;
        this.reference = create(batchSize, inputsMatrix, outputsMatrix, outputIndices);
        if (reference == 0) throw new IllegalStateException("Error creating NativeTrainingSamples");
    }

    @Override
    public long reference() {
        return reference;
    }

    @Override
    public void delete() {
        if (reference != 0) {
            delete(reference);
            reference = 0;
            inputsMatrix = null;
            outputsMatrix = null;
            outputIndices = null;
        }
    }

    @Override
    public int size() {
        return size(reference);
    }

    @Override
    public int batchSize() {
        return batchSize(reference);
    }

    private static native long create(int batchSize, Buffer inputsMatrix,
            Buffer outputsMatrix, Buffer outputIndices);

    private static native void delete(long reference);

    private static native int size(long reference);

    private static native int batchSize(long reference);

}