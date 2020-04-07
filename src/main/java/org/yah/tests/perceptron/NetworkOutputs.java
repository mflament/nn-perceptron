package org.yah.tests.perceptron;

import java.nio.IntBuffer;

public interface NetworkOutputs {

    int samples();

    int outputIndex(int sample);

    void reset();

    void copy(IntBuffer target);
}
