package org.yah.tests.perceptron.mt;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.Iterator;

import org.yah.tests.perceptron.Activation;
import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.RandomUtils;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tests.perceptron.mt.ChunkExecutor.ChunkHandler;
import org.yah.tests.perceptron.mt.MTSamplesSource.MTBatch;
import org.yah.tests.perceptron.mt.MTSamplesSource.MTTrainingSamples;

/**
 * @author Yah
 *
 */
public class MTNeuralNetwork implements NeuralNetwork, AutoCloseable {

    private final int[] layerSizes;
    private final int layers;
    private final ChunkExecutor executor;

    private final MTMatrix[] weights;
    private final MTMatrix[] biases;

    private final MTMatrix[] zs;
    private final MTMatrix[] activations;
    private final MTMatrix[] wgrads;
    private final MTMatrix[] bgrads;

    private final MTMatrix outputsMatrix;

    private final int totalWeights;

    private int[] batchOutputs;

    private final ForwardHandler forwardHandler = new ForwardHandler();
    private final CostDerivativeHandler costHandler = new CostDerivativeHandler();
    private final OutputsIndexer outputsIndexer = new OutputsIndexer();
    private final SigmoidPrimeHandler sigmoidPrimeHandler = new SigmoidPrimeHandler();
    private final DotHandler dotHandler = new DotHandler();
    private final ModelUpdateHandler modelUpdateHandler = new ModelUpdateHandler();

    private final MTMatrix transposed = new MTMatrix();

    public MTNeuralNetwork(int... layerSizes) {
        if (layerSizes.length < 2)
            throw new IllegalArgumentException("Invalid layers counts " + layerSizes.length);
        this.layerSizes = layerSizes;
        layers = layerSizes.length - 1;
        executor = new ChunkExecutor(Runtime.getRuntime().availableProcessors());
        weights = new MTMatrix[layers];
        biases = new MTMatrix[layers];
        zs = new MTMatrix[layers];
        activations = new MTMatrix[layers];
        wgrads = new MTMatrix[layers];
        bgrads = new MTMatrix[layers];
        int tw = 0;
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            weights[layer] = new MTMatrix(neurons, features);
            randomize(weights[layer], Math.sqrt(2.0 / features));
            biases[layer] = new MTMatrix(neurons, 1);
            tw += weights[layer].size();

            zs[layer] = new MTMatrix();
            activations[layer] = new MTMatrix();
            wgrads[layer] = new MTMatrix(neurons, features);
            bgrads[layer] = new MTMatrix(neurons, 1);
        }
        totalWeights = tw;
        outputsMatrix = activations[layers - 1];
    }

    @Override
    public void snapshot(int layer, DoubleBuffer buffer) {
        int neurons = neurons(layer);
        int features = features(layer);
        for (int n = 0; n < neurons; n++) {
            for (int f = 0; f < features; f++) {
                buffer.put(weights[layer].get(n, f));
            }
        }
        for (int n = 0; n < neurons; n++) {
            buffer.put(biases[layer].get(n, 0));
        }
    }

    private void randomize(MTMatrix m, double q) {
        int size = m.size();
        for (int i = 0; i < size; i++) {
            m.set(i, RandomUtils.nextGaussian() * q);
        }
    }

    @Override
    public void close() {
        executor.close();
    }

    @Override
    public int layers() {
        return layers;
    }

    @Override
    public int features() {
        return layerSizes[0];
    }

    @Override
    public int outputs() {
        return layerSizes[layers];
    }

    @Override
    public int features(int layer) {
        return layerSizes[layer];
    }

    @Override
    public int neurons(int layer) {
        return layerSizes[layer + 1];
    }

    @Override
    public SamplesSource createSampleSource() {
        return new MTSamplesSource(this);
    }

    @Override
    public void propagate(InputSamples samples, int[] outputIndices) {
        assert outputIndices.length == samples.size();

        MTTrainingSamples mtsamples = (MTTrainingSamples) samples;
        for (MTBatch batch : mtsamples) {
            forward(batch);
            indexOutputs(null, outputIndices, batch.offset());
        }
    }

    @Override
    public void propagate(InputSamples samples, IntBuffer outputIndices) {
        MTTrainingSamples mtsamples = (MTTrainingSamples) samples;
        for (MTBatch batch : mtsamples) {
            forward(batch);
            prepareBatchOutputs(batch.batchSize());
            indexOutputs(batch, batchOutputs, 0);
            outputIndices.put(batchOutputs);
        }
    }

    @Override
    public double evaluate(TrainingSamples samples, int[] outputIndices) {
        MTTrainingSamples mtsamples = (MTTrainingSamples) samples;
        Iterator<MTBatch> batchIter = mtsamples.iterator();
        int matched = 0;
        while (batchIter.hasNext()) {
            MTBatch batch = batchIter.next();
            forward(batch);
            matched += indexOutputs(batch, outputIndices, batch.offset());
        }
        return matched / (double) samples.size();
    }

    @Override
    public double evaluate(TrainingSamples samples, IntBuffer outputIndices) {
        MTTrainingSamples mtsamples = (MTTrainingSamples) samples;
        Iterator<MTBatch> batchIter = mtsamples.iterator();
        int matched = 0;
        while (batchIter.hasNext()) {
            MTBatch batch = batchIter.next();
            forward(batch);

            if (outputIndices != null) {
                prepareBatchOutputs(batch.batchSize());
                matched += indexOutputs(batch, batchOutputs, 0);
                outputIndices.put(batchOutputs);
            } else
                matched += indexOutputs(batch, null, 0);
        }
        return matched / (double) samples.size();
    }

    private int indexOutputs(MTBatch batch, int[] outputIndices, int indicesOffset) {
        outputsIndexer.prepare(batch, outputIndices, indicesOffset);
        executor.distribute(outputsMatrix.columns(), outputsIndexer);
        return outputsIndexer.matched;
    }

    @Override
    public void train(TrainingSamples samples, double learningRate) {
        MTTrainingSamples mtsamples = (MTTrainingSamples) samples;
        for (MTBatch batch : mtsamples) {
            // forward propagation
            forward(batch);

            // cost derivative = actual - expected
            costHandler.prepare(batch.expectedOutputs());
            executor.distribute(outputsMatrix.size(), costHandler);

            // backward propagation
            for (int layer = layers - 1; layer > 0; layer--) {
                backward(layer, activations[layer - 1]);
            }
            backward(0, batch.inputs());

            // update model
            modelUpdateHandler.prepare(learningRate / batch.batchSize());
            executor.distribute(totalWeights, modelUpdateHandler);
        }
    }

    private void forward(MTBatch batch) {
        prepareBatchSize(batch.batchSize());
        MTMatrix inputs = batch.inputs();
        for (int layer = 0; layer < layers; layer++) {
            forwardHandler.prepare(inputs, layer);
            executor.distribute(activations[layer].size(), forwardHandler);
            inputs = forwardHandler.a;
        }
    }

    private void backward(int layer, MTMatrix inputs) {
        // activation = activation * sigmoid_prime(z)
        sigmoidPrimeHandler.prepare(layer, bgrads[layer]);
        executor.distribute(sigmoidPrimeHandler.a.size(), sigmoidPrimeHandler);

        // wgrad = delta . T(inputs)
        MTMatrix wgrad = wgrads[layer];
        dotHandler.prepare(activations[layer], inputs.transpose(transposed), wgrad);
        executor.distribute(wgrad.size(), dotHandler);

        if (layer > 0) {
            // activation[layer-1] (next inputs) = T(weight[layer]) . delta
            MTMatrix nextInputs = activations[layer - 1];

            dotHandler.prepare(weights[layer].transpose(transposed), activations[layer], nextInputs);
            executor.distribute(nextInputs.size(), dotHandler);
        }
    }

    private void prepareBatchSize(int batchSize) {
        for (int layer = 0; layer < layers; layer++) {
            zs[layer].reshape(neurons(layer), batchSize);
            activations[layer].reshape(neurons(layer), batchSize);
        }
    }

    private int maxNeurons() {
        return Arrays.stream(layerSizes, 1, layers).max().orElse(0);
    }

    private void prepareBatchOutputs(int batchSize) {
        if (batchOutputs == null || batchOutputs.length < batchSize)
            batchOutputs = new int[batchSize];
    }

    private static abstract class MatrixHandler implements ChunkHandler {
        protected int rows;

        @Override
        public final void handle(int chunkIndex, int offset, int size) {
            int col = offset / rows;
            int row = offset % rows;
            for (int i = 0; i < size; i++) {
                handleElement(chunkIndex, row, col);
                row++;
                if (row == rows) {
                    row = 0;
                    col++;
                }
            }
        }

        protected abstract void handleElement(int chunkIndex, int row, int col);
    }

    private class ForwardHandler extends MatrixHandler {
        private MTMatrix i, a, z, w, b;

        public void prepare(MTMatrix inputs, int layer) {
            this.i = inputs;
            this.a = activations[layer];
            this.w = weights[layer];
            this.b = biases[layer];
            this.z = zs[layer];
            this.rows = w.rows();
        }

        // weight . inputs + bias
        @Override
        protected void handleElement(int chunkIndex, int row, int col) {
            double dot = 0;
            for (int c = 0; c < w.columns(); c++) {
                dot += w.get(row, c) * i.get(c, col);
            }
            dot += b.get(row, 0);
            z.set(row, col, dot);
            a.set(row, col, Activation.sigmoid(dot));
        }
    }

    private class OutputsIndexer implements ChunkHandler {
        private MTBatch batch;

        private int[] indices;
        private int indicesOffset;

        private int[] matcheds;
        private int matched;

        @Override
        public void start(int chunksCount) {
            if (this.matcheds == null || this.matcheds.length < chunksCount)
                this.matcheds = new int[chunksCount];
        }

        public void prepare(MTBatch batch, int[] indices, int indicesOffset) {
            this.batch = batch;
            this.indices = indices;
            this.indicesOffset = indicesOffset;
        }

        @Override
        public void handle(int chunkIndex, int offset, int size) {
            int matched = 0;
            for (int c = 0; c < size; c++, offset++) {
                int index = outputsMatrix.maxRowIndex(offset);
                if (batch.hasExpecteds()) {
                    int expectedIndex = batch.expectedIndex(offset);
                    if (expectedIndex == index)
                        matched++;
                }
                if (indices != null)
                    indices[indicesOffset + offset] = index;
            }
            matcheds[chunkIndex] = matched;
        }

        @Override
        public void complete() {
            matched = 0;
            for (int value : matcheds) {
                matched += value;
            }
        }
    }

    private class CostDerivativeHandler extends MatrixHandler {
        private MTMatrix expectedOutputs;

        public void prepare(MTMatrix expectedOutputs) {
            this.expectedOutputs = expectedOutputs;
            this.rows = expectedOutputs.rows();
        }

        @Override
        protected void handleElement(int chunkIndex, int row, int col) {
            outputsMatrix.sub(row, col, expectedOutputs.get(row, col));
        }
    }

    private static class DotHandler extends MatrixHandler {
        private MTMatrix a, b, res;

        public void prepare(MTMatrix a, MTMatrix b, MTMatrix res) {
            this.a = a;
            this.b = b;
            this.res = res;
            this.rows = res.rows();
        }

        @Override
        protected void handleElement(int chunkIndex, int row, int col) {
            // res = a . b
            int cols = a.columns();
            double sum = 0;
            for (int c = 0; c < cols; c++) {
                sum += a.get(row, c) * b.get(c, col);
            }
            res.set(row, col, sum);
        }
    }

    private class SigmoidPrimeHandler implements ChunkHandler {
        private MTMatrix a, z, bgrad;
        private MTMatrix[] chunkBgrads;
        private int rows;

        @Override
        public void start(int chunksCount) {
            if (chunkBgrads == null || chunkBgrads.length < chunksCount) {
                chunkBgrads = new MTMatrix[chunksCount];
                for (int i = 0; i < chunkBgrads.length; i++) {
                    chunkBgrads[i] = new MTMatrix(maxNeurons(), 1);
                }
            }
            for (MTMatrix chunkBgrad : chunkBgrads) {
                chunkBgrad.reshape(bgrad.rows(), 1);
                chunkBgrad.zero();
            }
        }

        public void prepare(int layer, MTMatrix bragd) {
            this.a = activations[layer];
            this.z = zs[layer];
            this.bgrad = bgrads[layer];
            rows = a.rows();
        }

        @Override
        public void handle(int chunkIndex, int offset, int size) {
            int row = offset % rows;
            for (int i = 0; i < size; i++, offset++) {
                // activation = activation * sigmoid_prime(z)
                double delta = a.mul(offset, Activation.sigmoid_prime(z.get(offset)));

                // sum activation row to thread bgrads
                chunkBgrads[chunkIndex].add(row, delta);
                if (++row == rows)
                    row = 0;
            }
        }

        @Override
        public void complete() {
            // sum chunk grads to layer bgrads
            for (int r = 0; r < rows; r++) {
                double s = 0;
                for (MTMatrix chunkBgrad : chunkBgrads) {
                    s += chunkBgrad.get(r);
                }
                bgrad.set(r, s);
            }
        }
    }

    private class ModelUpdateHandler implements ChunkHandler {

        private int[] layerOffsets;

        private double lr;

        @Override
        public void start(int chunksCount) {
            if (layerOffsets == null) {
                layerOffsets = new int[layers];
                layerOffsets[0] = weights[0].size();
                for (int i = 1; i < layers; i++) {
                    layerOffsets[i] = layerOffsets[i - 1] + weights[i].size();
                }
            }
        }

        public void prepare(double lr) {
            this.lr = lr;
        }

        @Override
        public void handle(int chunkIndex, int offset, int size) {
            int layer = layerIndex(offset);
            int layerOffset = offset - (layer > 0 ? layerOffsets[layer - 1] : 0);
            for (int i = 0; i < size; i++) {
                int neurons = neurons(layer);
                weights[layer].add(layerOffset, -lr * wgrads[layer].get(layerOffset));
                if (layerOffset < neurons) {
                    // first col
                    biases[layer].add(layerOffset, -lr * bgrads[layer].get(layerOffset));
                }
                layerOffset++;
                if (layerOffset == layerOffsets[layer]) {
                    layerOffset = 0;
                    layer++;
                }
            }

        }

        private int layerIndex(int offset) {
            for (int layer = 0; layer < layerOffsets.length; layer++) {
                if (offset < layerOffsets[layer])
                    return layer;
            }
            throw new IllegalArgumentException("offset " + offset + " is out of bound (" + totalWeights + ")");
        }

    }

}
