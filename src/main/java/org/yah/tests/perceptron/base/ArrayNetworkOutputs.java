package org.yah.tests.perceptron.base;

import org.yah.tests.perceptron.NetworkOutputs;

import java.nio.IntBuffer;

public class ArrayNetworkOutputs implements NetworkOutputs {

    private int[] outputs;
    private int offset;

    public ArrayNetworkOutputs(int size) {
        this.outputs = new int[size];
    }

    @Override
    public int samples() {
        return outputs.length;
    }

    @Override
    public int outputIndex(int sample) {
        return outputs[sample];
    }

    public void push(int index) {
        outputs[offset++] = index;
    }

    public void set(int index, int value) {
        outputs[index] = value;
    }

    @Override
    public void reset() {
        offset = 0;
    }

    @Override
    public void copy(IntBuffer target) {
        target.put(outputs).flip();
    }
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        int count = Math.min(outputs.length, 128);
        sb.append('[');
        for (int i = 0; i < count; i++) {
            sb.append(outputs[i]);
            if (i < count - 1) sb.append(", ");
        }
        if (count < outputs.length)
            sb.append(" ... (").append(outputs.length - count).append(" more)");
        sb.append(']');
        return sb.toString();
    }

}
