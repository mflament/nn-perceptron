package org.yah.tests.perceptron.opencl;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.lwjgl.BufferUtils;
import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.RandomUtils;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tests.perceptron.opencl.CLSamplesSource.CLTrainingBatch;
import org.yah.tests.perceptron.opencl.CLSamplesSource.CLTrainingSamples;
import org.yah.tools.opencl.CLEnvironment;
import org.yah.tools.opencl.cmdqueue.CLCommandQueue.KernelNDRange;
import org.yah.tools.opencl.cmdqueue.CLEvent;
import org.yah.tools.opencl.context.CLContext;
import org.yah.tools.opencl.kernel.CLKernel;
import org.yah.tools.opencl.mem.BufferProperties;
import org.yah.tools.opencl.mem.CLBuffer;
import org.yah.tools.opencl.mem.CLMemObject;

public class CLNeuralNetwork implements NeuralNetwork, AutoCloseable {

    interface MatrixElementProvider {
        float get(int row, int col);
    }

    private final CLEnvironment environment;

    private final int[] layerSizes;
    private final int layers;

    private final CLBuffer[] weightBuffers;
    private final CLBuffer[] biasBuffers;

    private final CLBuffer[] zBuffers;
    private final CLBuffer[] activationBuffers;
    private final CLBuffer[] wgrads;
    private final CLBuffer[] bgrads;
    private int batchSize;

    private final CLEvent[] layerEvents;

    private final CLKernel forwardKernel;
    private final CLKernel indexerKernel;
    private final CLKernel costKernel;
    private final CLKernel sigmoidPrimeKernel;
    private final CLKernel dotTransposeKernel;
    private final CLKernel transposeDotKernel;
    private final CLKernel updateLayerKernel;

    private final KernelNDRange range;

    private IntBuffer outputs;
    private CLBuffer outputBuffer;

    public CLNeuralNetwork(int... layerSizes) throws IOException {
        this(null, layerSizes);
    }

    public CLNeuralNetwork(CLContext clContext, int... layerSizes) throws IOException {
        this.environment = new CLEnvironment(clContext, "cl/neuralnetwork.cl", ""); //-DDEBUG
        this.layerSizes = layerSizes;
        this.layers = layerSizes.length - 1;
        this.weightBuffers = new CLBuffer[layers];
        this.biasBuffers = new CLBuffer[layers];
        this.layerEvents = new CLEvent[layers];
        this.wgrads = new CLBuffer[layers];
        this.bgrads = new CLBuffer[layers];
        for (int layer = 0; layer < layers; layer++) {
            this.weightBuffers[layer] = createWeights(layer);
            this.biasBuffers[layer] = createBiases(layer);
            this.layerEvents[layer] = environment.event();
            this.wgrads[layer] = createMatrixBuffer(neurons(layer), features(layer), null,
                    BufferProperties.MEM_READ_WRITE, BufferProperties.MEM_HOST_NO_ACCESS);
            this.bgrads[layer] = createMatrixBuffer(neurons(layer), 1, null,
                    BufferProperties.MEM_READ_WRITE, BufferProperties.MEM_HOST_NO_ACCESS);
        }

        this.zBuffers = new CLBuffer[layers];
        this.activationBuffers = new CLBuffer[layers];

        forwardKernel = environment.kernel("forward");
        indexerKernel = environment.kernel("indices");
        costKernel = environment.kernel("cost");
        sigmoidPrimeKernel = environment.kernel("sigmoid_prime");
        dotTransposeKernel = environment.kernel("dot_transpose");
        transposeDotKernel = environment.kernel("transpose_dot");
        updateLayerKernel = environment.kernel("update_layer");

        range = environment.createKernelRange();
    }

    @Override
    public void close() {
        environment.close();
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
        return layerSizes[layerSizes.length - 1];
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
        return new CLSamplesSource(this);
    }

    @Override
    public void propagate(InputSamples samples, int[] outputs) {
        CLTrainingSamples clsamples = (CLTrainingSamples) samples;
        assert outputs.length == samples.size();
        try (SamplesContext context = new SamplesContext(layers * samples.batchCount())) {
            Iterator<CLTrainingBatch> batchIterator = clsamples.iterator();
            while (batchIterator.hasNext()) {
                CLTrainingBatch batch = batchIterator.next();
                context.propagate(batch);
                environment.finish();
                // copy outputs
                this.outputs.get(outputs, batch.offset, batch.batchSize);
            }
        }
    }

    @Override
    public double evaluate(TrainingSamples samples, int[] outputs) {
        CLTrainingSamples clsamples = (CLTrainingSamples) samples;
        assert outputs.length == samples.size();
        int matched = 0;
        try (SamplesContext context = new SamplesContext(layers * samples.batchCount())) {
            Iterator<CLTrainingBatch> batchIterator = clsamples.iterator();
            while (batchIterator.hasNext()) {
                CLTrainingBatch batch = batchIterator.next();
                context.propagate(batch);
                environment.finish();
                // copy outputs
                if (outputs != null)
                    this.outputs.get(outputs, batch.offset, batch.batchSize);
                // count matched sample
                for (int sample = 0; sample < batch.batchSize; sample++) {
                    if (this.outputs.get(sample) == batch.getExpectedIndex(sample))
                        matched++;
                }
            }
        }

        return matched / (double) samples.size();
    }

    @Override
    public void train(TrainingSamples samples, double learningRate) {
        CLTrainingSamples clsamples = (CLTrainingSamples) samples;
        try (SamplesContext context = new SamplesContext(layers * samples.batchCount())) {
            Iterator<CLTrainingBatch> batchIterator = clsamples.iterator();
            while (batchIterator.hasNext()) {
                CLTrainingBatch batch = batchIterator.next();
                setBatchSize(batch.batchSize);
                CLEvent event = context.forward(batch, null);

                // compute cost derivative after last layer forward
                event = context.costDerivative(batch, event);

                // backward propagation
                for (int layer = layers - 1; layer > 0; layer--) {
                    event = context.backward(layer, batch.batchSize, activationBuffers[layer - 1], 0, event);
                }
                event = context.backward(0, batch.batchSize, batch.getInputs(), batch.offset, event);

                // update model
                float lr = (float) (learningRate / batch.batchSize);
                for (int layer = 0; layer < layers; layer++) {
                    context.prepareMatrixRange(neurons(layer), features(layer), event);
                    event = null; // no need to synchronize layers update
                    int arg = 0;
                    updateLayerKernel.setArg(arg++, neurons(layer));
                    updateLayerKernel.setArg(arg++, batch.batchSize);
                    updateLayerKernel.setArg(arg++, wgrads[layer]);
                    updateLayerKernel.setArg(arg++, bgrads[layer]);
                    updateLayerKernel.setArg(arg++, weightBuffers[layer]);
                    updateLayerKernel.setArg(arg++, biasBuffers[layer]);
                    updateLayerKernel.setArg(arg++, lr);
                    environment.run(updateLayerKernel, range);
                }
                environment.finish();
            }
        }
    }

    private void setBatchSize(int batchSize) {
        if (this.batchSize < batchSize) {
            for (int layer = 0; layer < layers; layer++) {
                if (zBuffers[layer] != null)
                    zBuffers[layer].close();
                int neurons = neurons(layer);
                zBuffers[layer] = createMatrixBuffer(neurons, batchSize, null, BufferProperties.MEM_READ_WRITE,
                        BufferProperties.MEM_HOST_NO_ACCESS);
                if (activationBuffers[layer] != null)
                    activationBuffers[layer].close();
                activationBuffers[layer] = createMatrixBuffer(neurons, batchSize, null, BufferProperties.MEM_READ_WRITE,
                        BufferProperties.MEM_HOST_NO_ACCESS);
            }

            if (outputBuffer != null)
                outputBuffer.close();
            outputs = BufferUtils.createIntBuffer(batchSize);
            outputBuffer = environment.mem(batchSize * Integer.BYTES,
                    BufferProperties.MEM_WRITE_ONLY,
                    BufferProperties.MEM_ALLOC_HOST_PTR,
                    BufferProperties.MEM_HOST_READ_ONLY);
        }
    }

    private CLBuffer createBiases(int layer) {
        return createMatrixBuffer(neurons(layer), 1, (r, c) -> 0,
                BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_WRITE,
                BufferProperties.MEM_HOST_WRITE_ONLY);
    }

    private CLBuffer createWeights(int layer) {
        int features = features(layer);
        float q = (float) Math.sqrt(2.0 / features);
        return createMatrixBuffer(neurons(layer), features,
                (r, c) -> (float) RandomUtils.nextGaussian() * q,
                BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_WRITE,
                BufferProperties.MEM_HOST_WRITE_ONLY);
    }

    CLBuffer createMatrixBuffer(int rows, int cols, MatrixElementProvider provider,
            BufferProperties... properties) {
        ByteBuffer fb = BufferUtils.createByteBuffer(rows * cols * Float.BYTES);
        if (provider != null) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    fb.putFloat(provider.get(r, c));
                }
            }
            fb.flip();
        }
        return environment.mem(fb, properties);
    }

    private class SamplesContext implements AutoCloseable {

        private final List<CLEvent> events;

        public SamplesContext(int eventsCapacity) {
            events = new ArrayList<CLEvent>(eventsCapacity);
        }

        @Override
        public void close() {
            events.forEach(CLEvent::close);
        }

        public CLEvent propagate(CLTrainingBatch batch) {
            setBatchSize(batch.batchSize);
            CLEvent event = forward(batch, null);
            // enqueue max index resolution
            event = prepareMatrixRange(1, batch.batchSize, event);
            int arg = 0;
            indexerKernel.setArg(arg++, outputs());
            indexerKernel.setArg(arg++, activationBuffers[layers - 1]);
            indexerKernel.setArg(arg++, outputBuffer);
            environment.run(indexerKernel, range);

            // synchronize outputs memory once indexer complete
            range.reset();
            event = prepareEvent(event);
            environment.read(outputBuffer, outputs, false, 0l, range);
            return event;
        }

        private CLEvent forward(CLTrainingBatch batch, CLEvent event) {
            CLMemObject inputs = batch.getInputs();
            int inputsOffset = batch.offset;
            for (int layer = 0; layer < layers; layer++) {
                int neurons = neurons(layer);
                event = prepareMatrixRange(neurons, batch.batchSize, event);
                int arg = 0;
                forwardKernel.setArg(arg++, neurons);
                forwardKernel.setArg(arg++, features(layer));
                forwardKernel.setArg(arg++, weightBuffers[layer]);
                forwardKernel.setArg(arg++, biasBuffers[layer]);
                forwardKernel.setArg(arg++, inputsOffset);
                forwardKernel.setArg(arg++, inputs);
                forwardKernel.setArg(arg++, zBuffers[layer]);
                forwardKernel.setArg(arg++, activationBuffers[layer]);
                // enqueue kernel execution
                environment.run(forwardKernel, range);
                inputs = activationBuffers[layer];
                inputsOffset = 0;
            }
            return event;
        }

        public CLEvent costDerivative(CLTrainingBatch batch, CLEvent event) {
            int neurons = outputs();
            event = prepareMatrixRange(neurons, batch.batchSize, event);
            int arg = 0;
            costKernel.setArg(arg++, neurons);
            costKernel.setArg(arg++, activationBuffers[layers - 1]);
            costKernel.setArg(arg++, batch.offset);
            costKernel.setArg(arg++, batch.getExpectedOutputs());
            environment.run(costKernel, range);
            return event;
        }

        private CLEvent backward(int layer, int batchSize,
                CLMemObject inputs, int inputsOffset,
                CLEvent event) {
            int neurons = neurons(layer);
            int features = features(layer);

            // sigmoid_prime
            event = prepareMatrixRange(neurons, batchSize, event);
            int arg = 0;
            sigmoidPrimeKernel.setArg(arg++, neurons);
            sigmoidPrimeKernel.setArg(arg++, zBuffers[layer]);
            sigmoidPrimeKernel.setArg(arg++, activationBuffers[layer]);
            environment.run(sigmoidPrimeKernel, range);

            // bgrads = sum(activation[layer])
            // wgrads = activation . T(inputs)
            event = prepareMatrixRange(neurons, features, event);
            arg = 0;
            dotTransposeKernel.setArg(arg++, neurons);
            dotTransposeKernel.setArg(arg++, features);
            dotTransposeKernel.setArg(arg++, batchSize);
            dotTransposeKernel.setArg(arg++, activationBuffers[layer]);
            dotTransposeKernel.setArg(arg++, inputsOffset);
            dotTransposeKernel.setArg(arg++, inputs);
            dotTransposeKernel.setArg(arg++, wgrads[layer]);
            dotTransposeKernel.setArg(arg++, bgrads[layer]);
            environment.run(dotTransposeKernel, range);

            if (layer > 0) {
                // activations[layer-1] = T(weights[layer]) . activations[layer
                event = prepareMatrixRange(neurons(layer - 1), batchSize, event);
                arg = 0;
                transposeDotKernel.setArg(arg++, neurons);
                transposeDotKernel.setArg(arg++, features);
                transposeDotKernel.setArg(arg++, weightBuffers[layer]);
                transposeDotKernel.setArg(arg++, activationBuffers[layer]);
                transposeDotKernel.setArg(arg++, activationBuffers[layer - 1]);
                environment.run(transposeDotKernel, range);
            }
            return event;
        }

        public CLEvent prepareMatrixRange(int rows, int columns, CLEvent waitFor) {
            range.reset(columns, rows);
            // set the event to generate to sync next layers
            return prepareEvent(waitFor);
        }

        public CLEvent prepareEvent(CLEvent waitFor) {
            CLEvent event = newEvent();
            range.setEvent(event);
            if (waitFor != null)
                range.setEventWaitList(waitFor);
            else
                range.setEventWaitList();
            return event;
        }

        public CLEvent newEvent() {
            CLEvent event = new CLEvent();
            events.add(event);
            return event;
        }
    }

    public static void main(String[] args) {
        System.out.println((1.0 / (1.0 - Math.exp(-0))) * 0);
    }
}
