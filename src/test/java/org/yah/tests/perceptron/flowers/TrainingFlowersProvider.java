package org.yah.tests.perceptron.flowers;

import java.util.Random;

import org.yah.tests.perceptron.RandomUtils;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;

/**
 * @author Yah
 *
 */
class TrainingFlowersProvider implements TrainingSamplesProvider {

    private final AllFlowersProvider allFlowersProvider;
    private final int[] sampleIndices;

    public TrainingFlowersProvider(AllFlowersProvider allFlowersProvider, int samples) {
        this.allFlowersProvider = allFlowersProvider;
        this.sampleIndices = randomizeFlowers(allFlowersProvider.samples(),
                Math.min(samples, allFlowersProvider.samples()));
    }

    @Override
    public int samples() {
        return sampleIndices.length;
    }

    @Override
    public double input(int sample, int feature) {
        return allFlowersProvider.input(sampleIndices[sample], feature);
    }

    @Override
    public int outputIndex(int sample) {
        return allFlowersProvider.outputIndex(sampleIndices[sample]);
    }

    private static int[] randomizeFlowers(int total, int count) {
        Random random = RandomUtils.RANDOM;
        int[] flowers = new int[total];
        for (int i = 0; i < total; i++)
            flowers[i] = i;
        for (int i = 0; i < total; i++)
            swap(flowers, i, random.nextInt(flowers.length));
        int[] res = new int[count];
        System.arraycopy(flowers, 0, res, 0, count);
        return res;
    }

    private static void swap(int[] array, int a, int b) {
        int buff = array[a];
        array[a] = array[b];
        array[b] = buff;
    }
}
