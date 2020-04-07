package org.yah.tests.perceptron.base;

import org.lwjgl.BufferUtils;
import org.yah.tests.perceptron.NetworkOutputs;

import java.nio.IntBuffer;

public class DirectBufferOutputs implements NetworkOutputs {

    private IntBuffer buffer;

    public DirectBufferOutputs(int size) {
        this.buffer = BufferUtils.createIntBuffer(size);
    }

    @Override
    public void reset() {
        buffer.position(0);
    }

    @Override
    public int samples() {
        return buffer.capacity();
    }

    @Override
    public int outputIndex(int sample) {
        return buffer.get(sample);
    }

    public IntBuffer buffer() {
        return buffer;
    }

    public IntBuffer position(int newPosition) {
        return buffer.position(newPosition);
    }

    public IntBuffer limit(int newLimit) {
        return buffer.limit(newLimit);
    }

    public int position() {
        return buffer.position();
    }

    public int limit() {
        return buffer.limit();
    }

    @Override
    public void copy(IntBuffer target) {
        target.put(buffer.position(0)).flip();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        int count = Math.min(buffer.capacity(), 128);
        sb.append('[');
        for (int i = 0; i < count; i++) {
            sb.append(buffer.get(i));
            if (i < count - 1) sb.append(", ");
        }
        if (count < buffer.capacity())
            sb.append(" ... (").append(buffer().capacity() - count).append(" more)");
        sb.append(']');
        return sb.toString();
    }
}
