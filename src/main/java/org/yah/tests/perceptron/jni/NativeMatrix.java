package org.yah.tests.perceptron.jni;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.yah.tests.perceptron.matrix.Matrix.MatrixFunction;

class NativeMatrix {

    private NativeMatrix() {}

    public static ByteBuffer create(int rows, int columns, MatrixFunction function) {
        int size = 2 * Integer.BYTES + rows * columns * Double.BYTES;
        ByteBuffer buffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
        buffer.putInt(rows);
        buffer.putInt(columns);
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                buffer.putDouble(function.apply(r, c, 0));
            }
        }
        buffer.flip();
        return buffer;
    }

}
