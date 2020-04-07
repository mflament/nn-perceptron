package org.yah.tests.perceptron.mt;

import org.yah.tests.perceptron.Activation;
import org.yah.tests.perceptron.NeuralNetworkState;
import org.yah.tests.perceptron.base.AbstractBatchedNeuralNetwork;
import org.yah.tests.perceptron.base.ArrayNetworkOutputs;
import org.yah.tests.perceptron.base.SamplesSource;
import org.yah.tests.perceptron.mt.ChunkExecutor.ChunkHandler;

/**
 * @author Yah
 */
public final class MTNeuralNetwork extends AbstractBatchedNeuralNetwork<MTBatch, ArrayNetworkOutputs> implements AutoCloseable {

    private final ChunkExecutor executor;

    private final MTMatrix[] weights;
    private final MTMatrix[] biases;
    private final MTMatrix[] zs;
    private final MTMatrix[] activations;
    private final MTMatrix[] wgrads;
    private final MTMatrix[] bgrads;

    private final ForwardHandler forwardHandler = new ForwardHandler();
    private final CostDerivativeHandler costHandler = new CostDerivativeHandler();
    private final OutputsIndexer outputsIndexer = new OutputsIndexer();
    private final SigmoidPrimeHandler sigmoidPrimeHandler = new SigmoidPrimeHandler();
    private final DotHandler dotHandler = new DotHandler();
    private final ModelUpdateHandler modelUpdateHandler = new ModelUpdateHandler();

    private final MTMatrix transposed = new MTMatrix();

    public MTNeuralNetwork(NeuralNetworkState state) {
        super(state);
        int layers = layers();
        executor = new ChunkExecutor(Runtime.getRuntime().availableProcessors());
        weights = new MTMatrix[layers];
        biases = new MTMatrix[layers];
        zs = new MTMatrix[layers];
        activations = new MTMatrix[layers];
        wgrads = new MTMatrix[layers];
        bgrads = new MTMatrix[layers];
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            weights[layer] = new MTMatrix(neurons, features);
            copyWeights(layer);
            biases[layer] = new MTMatrix(neurons, 1);
            copyBiases(layer);
            zs[layer] = new MTMatrix();
            activations[layer] = new MTMatrix();
            wgrads[layer] = new MTMatrix(neurons, features);
            bgrads[layer] = new MTMatrix(neurons, 1);
        }
    }

    private void copyWeights(int layer) {
        for (int neuron = 0; neuron < neurons(layer); neuron++) {
            for (int feature = 0; feature < features(layer); feature++) {
                weights[layer].set(neuron, feature, state.weight(layer, neuron, feature));
            }
        }
    }

    private void copyBiases(int layer) {
        for (int neuron = 0; neuron < neurons(layer); neuron++) {
            biases[layer].set(neuron, 0, state.bias(layer, neuron));
        }
    }

    @Override
    public void close() {
        executor.close();
    }

    @Override
    public SamplesSource<MTBatch> createSampleSource() {
        return new MTSamplesSource(this);
    }

    @Override
    public ArrayNetworkOutputs createOutpus(int samples) {
        return new ArrayNetworkOutputs(samples);
    }

    @Override
    protected void propagate(MTBatch batch, ArrayNetworkOutputs outputs) {
        MTMatrix outputsMatrix = forward(batch);
        indexOutputs(outputsMatrix, batch, outputs);
    }

    @Override
    protected int evaluate(MTBatch batch, ArrayNetworkOutputs outputs) {
        MTMatrix outputsMatrix = forward(batch);
        return indexOutputs(outputsMatrix, batch, outputs);
    }

    @Override
    protected void train(MTBatch batch, double learningRate) {
        // forward propagation
        MTMatrix outputs = forward(batch);

        // cost derivative = actual - expected
        costHandler.prepare(outputs, batch);
        executor.distribute(outputs.size(), costHandler);

        // backward propagation
        for (int layer = layers() - 1; layer > 0; layer--) {
            backward(layer, activations[layer - 1]);
        }
        backward(0, batch.inputs);

        // update model
        modelUpdateHandler.prepare(learningRate / batch.size());
        executor.distribute(totalWeights(), modelUpdateHandler);
    }

    @Override
    protected void updateState() {
        visitWeights((layer, neuron, feature) -> weight(layer, neuron, feature, weights[layer].get(neuron, feature)));
        visitBiases((layer, neuron) -> bias(layer, neuron, biases[layer].get(neuron, 0)));
    }

    @Override
    protected void updateModel() {
        visitWeights((layer, neuron, feature) -> weights[layer].set(neuron, feature, weight(layer, neuron, feature)));
        visitBiases((layer, neuron) -> biases[layer].set(neuron, 0, bias(layer, neuron)));
    }

    private int indexOutputs(MTMatrix outputsMatrix, MTBatch batch, ArrayNetworkOutputs networkOutputs) {
        outputsIndexer.prepare(outputsMatrix, batch, networkOutputs);
        executor.distribute(outputsMatrix.columns(), outputsIndexer);
        return outputsIndexer.matched;
    }

    private MTMatrix forward(MTBatch batch) {
        prepareBatchSize(batch.size());
        MTMatrix inputs = batch.inputs;
        for (int layer = 0; layer < layers(); layer++) {
            forwardHandler.prepare(inputs, layer);
            executor.distribute(activations[layer].size(), forwardHandler);
            inputs = forwardHandler.a;
        }
        return inputs;
    }

    private void backward(int layer, MTMatrix inputs) {
        // activation = activation * sigmoid_prime(z)
        sigmoidPrimeHandler.prepare(layer);
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
        for (int layer = 0; layer < layers(); layer++) {
            zs[layer].reshape(neurons(layer), batchSize);
            activations[layer].reshape(neurons(layer), batchSize);
        }
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

    private static class OutputsIndexer implements ChunkHandler {
        private MTBatch batch;
        private MTMatrix outputsMatrix;

        private ArrayNetworkOutputs outputsIndices;

        private int[] matcheds;
        private int matched;

        @Override
        public void start(int chunksCount) {
            if (this.matcheds == null || this.matcheds.length < chunksCount)
                this.matcheds = new int[chunksCount];
        }

        public void prepare(MTMatrix outputsMatrix,
                            MTBatch batch,
                            ArrayNetworkOutputs outputsIndices) {
            this.outputsMatrix = outputsMatrix;
            this.batch = batch;
            this.outputsIndices = outputsIndices;
        }

        @Override
        public void handle(int chunkIndex, int offset, int size) {
            int matched = 0;
            for (int c = 0; c < size; c++, offset++) {
                int index = outputsMatrix.maxRowIndex(offset);
                int expected = batch.expectedIndex(offset);
                if (expected == index)
                    matched++;
                if (outputsIndices != null)
                    outputsIndices.push(index);
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

    private static class CostDerivativeHandler extends MatrixHandler {
        private MTBatch batch;
        private MTMatrix outputs;

        public void prepare(MTMatrix outputs, MTBatch batch) {
            this.outputs = outputs;
            this.batch = batch;
            this.rows = outputs.rows();
        }

        @Override
        protected void handleElement(int chunkIndex, int row, int col) {
            outputs.sub(row, col, batch.expectedIndex(col) == row ? 1 : 0);
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

        public void prepare(int layer) {
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
                layerOffsets = new int[layers()];
                layerOffsets[0] = weights[0].size();
                for (int i = 1; i < layers(); i++) {
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
            throw new IllegalArgumentException("offset " + offset + " is out of bound (" + totalWeights() + ")");
        }
    }

}
