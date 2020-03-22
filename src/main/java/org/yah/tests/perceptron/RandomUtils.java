package org.yah.tests.perceptron;

import java.util.Random;

public final class RandomUtils {

    public static final Random RANDOM = createRandom();

    public static final long SEED = seed();

    private RandomUtils() {}

    private static Random createRandom() {
        long seed = seed();
        return seed < 0 ? new Random() : new Random(seed);
    }

    private static long seed() {
        long seed = -1;
        String prop = System.getProperty("seed");
        if (prop != null) {
            try {
                seed = Long.parseLong(prop);
            } catch (NumberFormatException e) {}
        }
        return seed;
    }

    public static double nextGaussian() {
        return RANDOM.nextGaussian();
    }

}
