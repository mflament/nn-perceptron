package org.yah.tests.perceptron;

public final class SamplesProviders {

    private SamplesProviders() {
    }

    public interface SamplesProvider {

        int samples();

        double input(int sample, int feature);
    }

    public interface TrainingSamplesProvider extends SamplesProvider {
        int outputIndex(int sample);
        default int[] createExpectedIndices() {
            int[] res = new int[samples()];
            for (int i = 0; i < res.length; i++) {
                res[i] = outputIndex(i);
            }
            return res;
        }
    }

    private static abstract class AbstractArraySamplesProvider implements TrainingSamplesProvider {
        protected final int samples;
        protected final double[][] inputs;
        protected final int[] outputIndices;

        protected AbstractArraySamplesProvider(int samples, double[][] inputs, int[] outputIndices) {
            this.samples = samples;
            this.inputs = inputs;
            this.outputIndices = outputIndices;
        }

        @Override
        public int samples() {
            return samples;
        }

        @Override
        public int outputIndex(int sample) {
            return outputIndices[sample];
        }
    }

    private static class CMArraySamplesProvider extends AbstractArraySamplesProvider {

        public CMArraySamplesProvider(double[][] inputs, int[] outputIndices) {
            super(inputs.length, inputs, outputIndices);
        }

        @Override
        public double input(int sample, int feature) {
            return inputs[sample][feature];
        }
    }

    private static class RMArraySamplesProvider extends AbstractArraySamplesProvider {

        public RMArraySamplesProvider(double[][] inputs, int[] outputIndices) {
            super(inputs[0].length, inputs, outputIndices);
        }

        @Override
        public double input(int sample, int feature) {
            return inputs[feature][sample];
        }
    }

    public static SamplesProvider newSamplesProvider(double[][] inputs, boolean transpose) {
        return newTrainingProvider(inputs, transpose, null);
    }

    public static TrainingSamplesProvider newTrainingProvider(double[][] inputs, boolean transpose,
                                                              int[] outputIndices) {
        return transpose ? new RMArraySamplesProvider(inputs, outputIndices)
                : new CMArraySamplesProvider(inputs, outputIndices);
    }
}