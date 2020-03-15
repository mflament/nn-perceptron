package org.yah.tests.perceptron;

public interface InputSamples {

    /**
     * @return the total number of samples
     */
    int size();

    int batchSize();
    
    default int batchCount() {
        return (int) Math.ceil(size() / (double) batchSize());
    }
}
