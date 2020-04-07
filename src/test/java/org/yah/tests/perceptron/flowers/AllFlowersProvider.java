package org.yah.tests.perceptron.flowers;

import org.yah.tests.perceptron.RandomUtils;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;

/**
 * @author Yah
 */
class AllFlowersProvider implements TrainingSamplesProvider {

    private final int width, height;
    private final int samples;
    private final double noiseScale;
    private final OpenSimplexNoise noise;

    public AllFlowersProvider(int width, int height, double noiseScale) {
        this(width, height, width * height, noiseScale);
    }

    public AllFlowersProvider(int width, int height, int samples, double noiseScale) {
        this.width = width;
        this.height = height;
        this.samples = samples;
        this.noiseScale = noiseScale;
        noise = new OpenSimplexNoise(RandomUtils.SEED < 0 ? System.currentTimeMillis() : RandomUtils.SEED);
    }

    @Override
    public int samples() {
        return samples;
    }

    @Override
    public double input(int sample, int feature) {
        if (feature == 0) {
            int x = sample % width;
            return x / (double) width;
        } else if (feature == 1) {
            int y = sample / width;
            return y / (double) height;
        }
        throw new IllegalArgumentException("Invalid feature " + feature);
    }

    @Override
    public int outputIndex(int sample) {
        double dx = input(sample, 0);
        double dy = input(sample, 1);
        double n = noise.eval(dx * noiseScale, dy * noiseScale);
        return n < 0 ? 0 : 1;
    }

}
