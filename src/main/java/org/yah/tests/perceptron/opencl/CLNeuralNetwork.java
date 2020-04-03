package org.yah.tests.perceptron.opencl;

import org.lwjgl.BufferUtils;
import org.yah.tests.perceptron.*;
import org.yah.tests.perceptron.opencl.CLSamplesSource.CLTrainingBatch;
import org.yah.tests.perceptron.opencl.CLSamplesSource.CLTrainingSamples;
import org.yah.tools.opencl.CLEnvironment;
import org.yah.tools.opencl.CLUtils;
import org.yah.tools.opencl.cmdqueue.CLCommandQueue.KernelNDRange;
import org.yah.tools.opencl.context.CLContext;
import org.yah.tools.opencl.kernel.CLKernel;
import org.yah.tools.opencl.mem.BufferProperties;
import org.yah.tools.opencl.mem.CLBuffer;
import org.yah.tools.opencl.mem.CLMemObject;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.util.Iterator;

public class CLNeuralNetwork implements NeuralNetwork, AutoCloseable {

    private static final int SUM_WORKGROUP_SIZE = 128;

    private static final int TYPE_SIZE = Double.BYTES;

    interface MatrixElementProvider {
        double get(int row, int col);
    }

    final CLEnvironment environment;

    private final int[] layerSizes;
    private final int layers;

    private final int totalNeurons, totalWeights;

    private final IntBuffer networkBuffer;

    private final CLKernel propagateKernel;
    private final CLKernel evaluate0Kernel;
    private final CLKernel evaluate1Kernel;
    private final CLKernel sumMatchedKernel;
    private final CLKernel trainingKernel;
    private final CLKernel sumGradsKernel;
    private final CLKernel updateNetworkKernel;

    private final long[] localWorkSize;
    private final long workGroupSize;

    private final KernelNDRange range;
    private final KernelNDRange sumRange;

    private CLBuffer weightsBuffer;
    private CLBuffer biasesBuffer;

    private CLBuffer outputsBuffer;
    private int outputsCapacity;
    private IntBuffer outputsIntBuffer;

    private CLBuffer matchedCountBuffer;
    private int matchedCountCapacity;

    private CLBuffer wgradsBuffer;
    private CLBuffer bgradsBuffer;
    private int trainingBuffersCapacity;

    private final IntBuffer matchedCountResult = BufferUtils.createIntBuffer(1);

    public CLNeuralNetwork(int... layerSizes) throws IOException {
        this(null, layerSizes);
    }

    public CLNeuralNetwork(CLContext clContext, int... layerSizes) throws IOException {
        this.layerSizes = layerSizes;
        this.layers = layerSizes.length - 1;
        this.environment = CLEnvironment.builder()
                .withContext(clContext)
                .withSourceResource("cl/neuralnetwork.cl")
                .withOptions("-DLAYERS=" + layers + " -DTYPE=double")
                .build();

        int maxFeatures = layerSizes[0];
        int maxNeurons = 0;
        int tn = 0, tw = 0;
        for (int layer = 0; layer < layers; layer++) {
            maxFeatures = Math.max(maxFeatures, features(layer));
            maxNeurons = Math.max(maxNeurons, neurons(layer));
            tn += neurons(layer);
            tw += neurons(layer) * features(layer);
        }
        totalNeurons = tn;
        totalWeights = tw;

        localWorkSize = new long[]{CLUtils.nextPowerOfTwo(maxNeurons), CLUtils.nextPowerOfTwo(maxFeatures)};
        workGroupSize = localWorkSize[0] * localWorkSize[1];
        range = environment.kernelRange();
        range.globalWorkSizes(localWorkSize[0], localWorkSize[1]);
        range.localWorkSizes(localWorkSize);
        range.validate();

        sumRange = environment.kernelRange();

        networkBuffer = createNetworkBuffer();
        weightsBuffer = createWeights();
        biasesBuffer = createBiases();

        propagateKernel = environment.kernel("propagate");
        evaluate0Kernel = environment.kernel("evaluate0");
        evaluate1Kernel = environment.kernel("evaluate1");
        sumMatchedKernel = environment.kernel("sum_matched");
        trainingKernel = environment.kernel("train");
        sumGradsKernel = environment.kernel("sum_grads");
        updateNetworkKernel = environment.kernel("update_network");
    }

    private CLBuffer createBiases() {
        // zero device memory (not always done if using size initialization)
        DoubleBuffer buffer = BufferUtils.createDoubleBuffer(totalNeurons);
        return environment.mem(buffer, BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_WRITE,
                BufferProperties.MEM_HOST_READ_ONLY);
    }

    private CLBuffer createWeights() {
        int size = 0;
        for (int layer = 0; layer < layers; layer++) {
            size += neurons(layer) * features(layer);
        }
        DoubleBuffer buffer = BufferUtils.createDoubleBuffer(size);
        for (int layer = 0; layer < layers; layer++) {
            int neurons = neurons(layer);
            int features = features(layer);
            double q = Math.sqrt(2.0 / features);
            for (int n = 0; n < neurons; n++) {
                for (int f = 0; f < features; f++) {
                    double w = RandomUtils.nextGaussian() * q;
                    buffer.put(w);
                }
            }
        }
        buffer.flip();
        return environment.mem(buffer, BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_WRITE,
                BufferProperties.MEM_HOST_READ_ONLY);
    }

    /**
     * <code>
     * typedef struct Network {
     * int inputs;
     * int layers[LAYERS];
     * int totalWeights;
     * int totalNeurons;
     * } Network;
     * </code>
     */
    private IntBuffer createNetworkBuffer() {
        IntBuffer res = BufferUtils.createIntBuffer(layerSizes.length + 2);
        return res.put(layerSizes).put(totalWeights).put(totalNeurons).flip();
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
    public void snapshot(int layer, DoubleBuffer buffer) {
        throw new UnsupportedOperationException();
    }

    @Override
    public SamplesSource createSampleSource() {
        return new CLSamplesSource(this);
    }

    @Override
    public void propagate(InputSamples samples, int[] outputs) {
        if (outputsIntBuffer == null || outputsIntBuffer.capacity() < outputs.length) {
            outputsIntBuffer = BufferUtils.createIntBuffer(outputs.length);
        }
        outputsIntBuffer.limit(outputs.length);
        propagate(samples, outputsIntBuffer);
        outputsIntBuffer.get(outputs);
    }

    @Override
    public void propagate(InputSamples samples, IntBuffer outputs) {
        CLTrainingSamples clsamples = (CLTrainingSamples) samples;
        Iterator<CLTrainingBatch> iterator = clsamples.iterator();
        long event = 0;
        while (iterator.hasNext()) {
            CLTrainingBatch batch = iterator.next();
            range.waitForEvent(event);
            event = forward(batch, null, outputs);

            range.waitForEvent(event);
            event = readOutputs(batch, outputs);
        }
        environment.finish();
        outputs.position(0);
    }

    @Override
    public double evaluate(TrainingSamples samples, int[] outputs) {
        double res;
        if (outputs != null) {
            if (outputsIntBuffer == null || outputsIntBuffer.capacity() < outputs.length)
                outputsIntBuffer = BufferUtils.createIntBuffer(outputs.length);
            outputsIntBuffer.limit(outputs.length);
            res = evaluate(samples, outputsIntBuffer);
            outputsIntBuffer.get(outputs);
        } else {
            res = evaluate(samples, (IntBuffer) null);
        }
        return res;
    }

    @Override
    public double evaluate(TrainingSamples samples, IntBuffer outputs) {
        int matched = 0;
        CLTrainingSamples clSamples = (CLTrainingSamples) samples;
        for (CLTrainingBatch batch : clSamples) {
            long forwardEvent = forward(batch, batch.getExpectedIndices(), outputs);

            long matchedSumEvent = sumMatched(batch, forwardEvent);
            range.waitForEvent(matchedSumEvent);
            range.requestEvent();
            matchedSumEvent = environment.read(matchedCountBuffer, matchedCountResult, false, 0, range);

            long outputsEvent = 0;
            if (outputs != null) {
                range.waitForEvent(forwardEvent);
                outputsEvent = readOutputs(batch, outputs);
            }

            range.waitForEvents(matchedSumEvent, outputsEvent);
            environment.waitForEvents(range);
            matched += matchedCountResult.get(0);

            range.waitForEvents();
        }
        environment.finish();
        if (outputs != null)
            outputs.position(0);

        return matched / (double) samples.size();
    }

    @Override
    public void train(TrainingSamples samples, double learningRate) {
        CLTrainingSamples clsamples = (CLTrainingSamples) samples;
        long event = 0;
        for (CLTrainingBatch batch : clsamples) {
            event = train(batch, event);
            event = sumGrads(batch, event);
            event = updateNetwork(event, learningRate / batch.batchSize);
        }
        environment.finish();
    }

    private long forward(CLTrainingBatch batch, CLMemObject expectedIndices, IntBuffer outputs) {
        CLKernel kernel;
        if (expectedIndices == null) {
            kernel = propagateKernel;
        } else if (outputs == null) {
            kernel = evaluate0Kernel;
        } else {
            kernel = evaluate1Kernel;
        }

        int index = 0;
        kernel.setArg(index++, networkBuffer);
        kernel.setArg(index++, weightsBuffer);
        kernel.setArg(index++, biasesBuffer);
        kernel.setArg(index++, batch.offset);
        kernel.setArg(index++, batch.getInputs());
        if (expectedIndices != null) {
            ensureMatchedCount(batch.batchSize);
            kernel.setArg(index++, expectedIndices);
            kernel.setArg(index++, matchedCountBuffer);
            if (outputs != null) {
                ensureOutputs(batch.batchSize);
                kernel.setArg(index++, outputsBuffer);
            }
        } else {
            ensureOutputs(batch.batchSize);
            kernel.setArg(index++, outputsBuffer);
        }
        kernel.setArgSize(index++, workGroupSize * TYPE_SIZE);
        kernel.setArgSize(index, outputs() * Integer.BYTES);

        range.globalWorkSizes(localWorkSize[0], localWorkSize[1] * batch.batchSize).requestEvent();
        return environment.run(kernel, range);
    }

    private long train(CLTrainingBatch batch, long event) {
        ensureTrainingBufferSize(batch.batchSize);
        int index = 0;
        trainingKernel.setArg(index++, networkBuffer);
        trainingKernel.setArg(index++, weightsBuffer);
        trainingKernel.setArg(index++, biasesBuffer);
        trainingKernel.setArg(index++, batch.offset);
        trainingKernel.setArg(index++, batch.getInputs());
        trainingKernel.setArg(index++, batch.getExpectedIndices());
        trainingKernel.setArg(index++, wgradsBuffer);
        trainingKernel.setArg(index++, bgradsBuffer);
        trainingKernel.setArgSize(index++, workGroupSize * TYPE_SIZE); // partial
        trainingKernel.setArgSize(index++, totalNeurons * TYPE_SIZE); // zs
        trainingKernel.setArgSize(index, totalNeurons * TYPE_SIZE); // activations

        range.globalWorkSizes(localWorkSize[0], localWorkSize[1] * batch.batchSize).waitForEvent(event).requestEvent();
        return environment.run(trainingKernel, range);
    }

    private long sumMatched(CLTrainingBatch batch, long forwardEvent) {
        long event = forwardEvent;
        int size = batch.batchSize;
        do {
            int groups = (int) Math.ceil(size / (double) SUM_WORKGROUP_SIZE);
            event = sumMatched(size, groups, event);
            size = groups;
        } while (size > 1);
        return event;
    }

    private long sumMatched(int size, int groups, long event) {
        sumRange.globalWorkSizes(groups * SUM_WORKGROUP_SIZE)
                .localWorkSizes(SUM_WORKGROUP_SIZE)
                .waitForEvent(event).requestEvent();

        sumMatchedKernel.setArg(0, size);
        sumMatchedKernel.setArg(1, matchedCountBuffer);
        sumMatchedKernel.setArgSize(2, SUM_WORKGROUP_SIZE * Integer.BYTES);
        return environment.run(sumMatchedKernel, sumRange);
    }

    private long sumGrads(CLTrainingBatch batch, long trainingEvent) {
        long event = trainingEvent;
        int size = batch.batchSize;
        do {
            int groupSize = SUM_WORKGROUP_SIZE;
            if (size < SUM_WORKGROUP_SIZE) {
                groupSize = CLUtils.nextPowerOfTwo(size);
            }
            int groups = (int) Math.ceil(size / (double) groupSize);
            event = sumGrads(size, groups, groupSize, event);
            size = groups;
        } while (size > 1);
        return event;
    }

    private long sumGrads(int size, int groups, int groupSize, long event) {
        sumRange.globalWorkSizes(groups * groupSize, totalWeights)
                .localWorkSizes(groupSize, 1)
                .waitForEvent(event)
                .requestEvent();

        int index = 0;
        sumGradsKernel.setArg(index++, networkBuffer);
        sumGradsKernel.setArg(index++, size);
        sumGradsKernel.setArg(index++, wgradsBuffer);
        sumGradsKernel.setArg(index++, bgradsBuffer);
        sumGradsKernel.setArgSize(index++, totalWeights * TYPE_SIZE);
        sumGradsKernel.setArgSize(index, totalNeurons * TYPE_SIZE);
        return environment.run(sumGradsKernel, sumRange);
    }

    private long updateNetwork(long event, double lr) {
        sumRange.globalWorkSizes(totalWeights)
                .localWorkSizes()
                .waitForEvent(event)
                .requestEvent();
        int index = 0;
        updateNetworkKernel.setArg(index++, networkBuffer);
        updateNetworkKernel.setArg(index++, lr);
        updateNetworkKernel.setArg(index++, weightsBuffer);
        updateNetworkKernel.setArg(index++, biasesBuffer);
        updateNetworkKernel.setArg(index++, wgradsBuffer);
        updateNetworkKernel.setArg(index, bgradsBuffer);

        return environment.run(updateNetworkKernel, sumRange);
    }

    private long readOutputs(CLTrainingBatch batch, IntBuffer outputs) {
        range.requestEvent();
        outputs.position(batch.offset).limit(batch.offset + batch.batchSize);
        return environment.read(outputsBuffer, outputs, false, 0, range);
    }

    private void ensureOutputs(int size) {
        if (outputsCapacity < size) {
            if (outputsBuffer != null)
                outputsBuffer.close();
            outputsBuffer = environment.mem(size * Integer.BYTES,
                    BufferProperties.MEM_ALLOC_HOST_PTR,
                    BufferProperties.MEM_WRITE_ONLY,
                    BufferProperties.MEM_HOST_READ_ONLY);
            outputsCapacity = size;
        }
    }

    private void ensureMatchedCount(int size) {
        size = Math.max(size, SUM_WORKGROUP_SIZE);
        if (matchedCountCapacity < size) {
            if (matchedCountBuffer != null)
                matchedCountBuffer.close();
            matchedCountBuffer = environment.mem(size * Integer.BYTES,
                    BufferProperties.MEM_ALLOC_HOST_PTR,
                    BufferProperties.MEM_READ_WRITE,
                    BufferProperties.MEM_HOST_READ_ONLY);
            matchedCountCapacity = size;
        }
    }

    private void ensureTrainingBufferSize(int size) {
        if (trainingBuffersCapacity < size) {
            if (wgradsBuffer != null)
                wgradsBuffer.close();
            if (bgradsBuffer != null)
                bgradsBuffer.close();
            wgradsBuffer = environment.mem(size * totalWeights * TYPE_SIZE,
                    BufferProperties.MEM_READ_WRITE,
                    BufferProperties.MEM_HOST_NO_ACCESS);
            bgradsBuffer = environment.mem(size * totalNeurons * TYPE_SIZE,
                    BufferProperties.MEM_READ_WRITE,
                    BufferProperties.MEM_HOST_NO_ACCESS);
            trainingBuffersCapacity = size;
        }
    }

    CLBuffer createMatrixBuffer(int rows, int cols, MatrixElementProvider provider,
                                BufferProperties... properties) {
        ByteBuffer fb = BufferUtils.createByteBuffer(rows * cols * TYPE_SIZE);
        if (provider != null) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    fb.putDouble(provider.get(r, c));
                }
            }
            fb.flip();
        }
        return environment.mem(fb, properties);
    }

}
