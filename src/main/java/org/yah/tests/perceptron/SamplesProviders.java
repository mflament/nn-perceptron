package org.yah.tests.perceptron;

public final class SamplesProviders {

    private SamplesProviders() {}

    public interface SamplesProvider {

        int samples();

        int features();

        double input(int sample, int feature);
    }

    public interface TrainingSamplesProvider extends SamplesProvider {
        int outputIndex(int sample);
    }

    private static abstract class AbstractArraySamplesProvider implements TrainingSamplesProvider {
        protected final int samples;
        protected final int features;
        protected final double[][] inputs;
        protected final int[] outputIndices;

        /**
         * @param inputs        if !transpose : column major samples, else row major
         *                      samples
         * @param outputIndices null or int[samples]
         * @param transpose     transpose input from row major to column major
         */
        protected AbstractArraySamplesProvider(int samples, int features, double[][] inputs,
                int[] outputIndices) {
            this.samples = samples;
            this.features = features;
            this.inputs = inputs;
            this.outputIndices = outputIndices;
        }

        @Override
        public int samples() {
            return samples;
        }

        @Override
        public int features() {
            return features;
        }

        @Override
        public int outputIndex(int sample) {
            return outputIndices[sample];
        }
    }

    private static class CMArraySamplesProvider extends AbstractArraySamplesProvider {

        public CMArraySamplesProvider(double[][] inputs, int[] outputIndices) {
            super(inputs.length, inputs.length == 0 ? 0 : inputs[0].length, inputs, outputIndices);
        }

        @Override
        public double input(int sample, int feature) {
            return inputs[sample][feature];
        }
    }

    private static class RMArraySamplesProvider extends AbstractArraySamplesProvider {

        public RMArraySamplesProvider(double[][] inputs, int[] outputIndices) {
            super(inputs.length == 0 ? 0 : inputs[0].length, inputs.length, inputs, outputIndices);
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