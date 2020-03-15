package org.yah.tests.perceptron;

public interface Batch{

    /**
     * @return the number of samples
     */
    int size();

    /**
     * @return index of this batch in dataset
     */
    int index();

}
