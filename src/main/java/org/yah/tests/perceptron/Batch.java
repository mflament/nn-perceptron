package org.yah.tests.perceptron;

public interface Batch<M extends Matrix<M>> {

    /**
     * @return the number of samples
     */
    default int size() {
        return inputs().columns();
    }

    /**
     * @return index of this batch in dataset
     */
    int index();

    /**
     * @return Matrix[network.features x samples count]
     */
    M inputs();

    /**
     * @return Matrix[network.outputs x samples count]
     */
    M expectedOutputs();

    /**
     * @return Matrix[1 x samples count] with indices of expected outputs neuron
     */
    M expectedIndices();

    default double accuracy(M outputs, int[] outputIndices) {
        int matched = 0;
        int samples = outputs.columns();
        M expectedIndices = expectedIndices();
        assert samples == expectedIndices.columns();
        for (int sample = 0; sample < samples; sample++) {
            int outputIndex = outputs.maxRowIndex(sample);
            if (expectedIndices.get(0, sample) == outputIndex)
                matched++;
            if (outputIndices != null)
                outputIndices[sample] = outputIndex;
        }
        return matched / (double) samples;
    }

}
