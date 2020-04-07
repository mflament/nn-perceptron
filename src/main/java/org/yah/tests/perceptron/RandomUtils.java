package org.yah.tests.perceptron;

import java.util.Random;
import java.util.function.DoubleSupplier;

public final class RandomUtils {

    public static final long SEED = seed();

    private RandomUtils() {}

    public static Random newRandom() {
        return newRandom(seed());
    }

    public static Random newRandom(long seed) {
        return seed < 0 ? new Random() : new Random(seed);
    }

    private static long seed() {
        long seed = -1;
        String prop = System.getProperty("seed");
        if (prop != null) {
            try {
                seed = Long.parseLong(prop);
            } catch (NumberFormatException ignored) {}
        }
        return seed;
    }

    public static DoubleSupplier newRandomSource() {
        return newRandomSource(seed());
    }

    public static DoubleSupplier newRandomSource(long seed) {
        Random random = newRandom(seed);
        return () -> random.nextGaussian();
    }

}
