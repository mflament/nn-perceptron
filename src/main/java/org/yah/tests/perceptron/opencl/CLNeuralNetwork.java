package org.yah.tests.perceptron.opencl;

import org.lwjgl.BufferUtils;
import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.NeuralNetworkState;
import org.yah.tests.perceptron.base.AbstractBatchedNeuralNetwork;
import org.yah.tests.perceptron.base.DirectBufferOutputs;
import org.yah.tests.perceptron.base.SamplesSource;
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

public final class CLNeuralNetwork extends AbstractBatchedNeuralNetwork<CLTrainingBatch, DirectBufferOutputs>
        implements AutoCloseable {

    private static final int SUM_WORKGROUP_SIZE = 64;

    private static final int TYPE_SIZE = Double.BYTES;

    interface MatrixElementProvider {
        double get(int row, int col);
    }

    final CLEnvironment environment;

    private final IntBuffer networkBuffer;

    private final CLKernel propagateKernel;
    private final CLKernel sumMatchedKernel;

    private final CLKernel trainingKernel;
    private final CLKernel sumGradsKernel;
    private final CLKernel updateNetworkKernel;

    private final long[] localWorkSize;
    private final long workGroupSize;

    private final KernelNDRange range;
    private final KernelNDRange sumRange;

    /**
     * for each layer: weights (column major) + biases
     */
    private CLBuffer dataBuffer;

    private CLBuffer outputsBuffer;
    private int outputsCapacity;

    private CLBuffer matchedCountBuffer;
    private int matchedCountCapacity;

    private CLBuffer gradsBuffer;
    private int trainingBuffersCapacity;

    private final IntBuffer matchedCountResult = BufferUtils.createIntBuffer(1);

    public CLNeuralNetwork(NeuralNetworkState state) throws IOException {
        this(null, state);
    }

    public CLNeuralNetwork(CLContext clContext, NeuralNetworkState state) throws IOException {
        super(state);
        this.environment = CLEnvironment.builder()
                .withContext(clContext)
                .withSourceResource("cl/neuralnetwork.cl")
                .withOptions("-DLAYERS=" + layers() + " -DTYPE=double")
                .build();

        localWorkSize = new long[]{CLUtils.nextPowerOfTwo(maxNeurons()), CLUtils.nextPowerOfTwo(maxFeatures())};
        workGroupSize = localWorkSize[0] * localWorkSize[1];
        range = environment.kernelRange();
        range.globalWorkSizes(localWorkSize[0], localWorkSize[1]);
        range.localWorkSizes(localWorkSize);
        range.validate();

        sumRange = environment.kernelRange();

        networkBuffer = createNetworkBuffer();
        dataBuffer = createDataCLBuffer();

        propagateKernel = environment.kernel("propagate");
        sumMatchedKernel = environment.kernel("sum_matched");

        trainingKernel = environment.kernel("train");
        sumGradsKernel = environment.kernel("sum_grads");
        updateNetworkKernel = environment.kernel("update_network");
    }

    private CLBuffer createDataCLBuffer() {
        return environment.mem(createDataDoubleBufffer(), BufferProperties.MEM_COPY_HOST_PTR,
                BufferProperties.MEM_READ_WRITE);
    }

    private DoubleBuffer createDataDoubleBufffer() {
        DoubleBuffer buffer = BufferUtils.createDoubleBuffer(totalWeights() + totalNeurons());
        for (int layer = 0; layer < layers(); layer++) {
            visitWeights(layer, (l, neuron, feature) -> buffer.put(weight(l, neuron, feature)));
            visitBiases(layer, (l, neuron) -> buffer.put(bias(l, neuron)));
        }
        return buffer.flip();
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
        IntBuffer res = BufferUtils.createIntBuffer(layers() + 3);
        res.put(features());
        for (int layer = 0; layer < layers(); layer++) {
            res.put(neurons(layer));
        }
        res.put(totalWeights()).put(totalNeurons());
        return res.flip();
    }

    @Override
    public void close() {
        environment.close();
    }

    @Override
    protected SamplesSource<CLTrainingBatch> createSampleSource() {
        return new CLSamplesSource(this);
    }

    @Override
    public DirectBufferOutputs createOutpus(int samples) {
        return new DirectBufferOutputs(samples);
    }

    @Override
    protected void doPropagate(InputSamples samples, DirectBufferOutputs outputs) {
        super.doPropagate(samples, outputs);
        outputs.reset();
    }

    @Override
    protected double doEvaluate(InputSamples samples, DirectBufferOutputs outputs) {
        double res = super.doEvaluate(samples, outputs);
        if (outputs != null)
            outputs.reset();
        return res;
    }

    @Override
    protected void propagate(CLTrainingBatch batch, DirectBufferOutputs outputs) {
        range.dontWaitForEvents();
        long event = forward(batch, null, outputs.buffer());
        range.waitForEvent(event);
        readOutputs(batch, outputs.buffer());
    }

    @Override
    protected int evaluate(CLTrainingBatch batch, DirectBufferOutputs outputs) {
        IntBuffer outputsBuffer = outputs != null ? outputs.buffer() : null;
        range.dontWaitForEvents();
        long forwardEvent = forward(batch, batch.getExpectedIndices(), outputsBuffer);
        long matchedSumEvent = sumMatched(batch, forwardEvent);
        range.waitForEvent(matchedSumEvent);
        range.requestEvent();
        matchedSumEvent = environment.read(matchedCountBuffer, matchedCountResult, false, 0, range);
        if (outputs != null) {
            range.waitForEvent(forwardEvent);
            readOutputs(batch, outputsBuffer);
        }
        environment.waitForEvent(matchedSumEvent);
        return matchedCountResult.get(0);
    }

    @Override
    protected void train(CLTrainingBatch batch, double learningRate) {
        ensureTrainingBufferSize(batch.batchSize);
        int index = 0;
        trainingKernel.setArg(index++, networkBuffer);
        trainingKernel.setArg(index++, dataBuffer);
        trainingKernel.setArg(index++, batch.offset);
        trainingKernel.setArg(index++, batch.getInputs());
        trainingKernel.setArg(index++, batch.getExpectedIndices());
        trainingKernel.setArg(index++, gradsBuffer);
        trainingKernel.setArgSize(index++, workGroupSize * TYPE_SIZE); // partial
        trainingKernel.setArgSize(index++, totalNeurons() * TYPE_SIZE); // zs
        trainingKernel.setArgSize(index, totalNeurons() * TYPE_SIZE); // activations
        range.globalWorkSizes(localWorkSize[0], localWorkSize[1] * batch.batchSize)
                .dontWaitForEvents()
                .requestEvent();
        long event = environment.run(trainingKernel, range);
        environment.waitForEvent(event);

        event = sumGrads(batch, event);
        environment.waitForEvent(event);

        event = updateNetwork(event, learningRate / batch.batchSize);
        environment.waitForEvent(event);
    }

    @Override
    protected void updateState() {
        DoubleBuffer buffer = BufferUtils.createDoubleBuffer(totalWeights() + totalNeurons());
        environment.read(dataBuffer, buffer);
        for (int layer = 0; layer < layers(); layer++) {
            visitWeights(layer, (l, neuron, feature) -> weight(l, neuron, feature, buffer.get()));
            visitBiases(layer, (l, neuron) -> bias(l, neuron, buffer.get()));
        }
    }

    @Override
    protected void updateModel() {
        DoubleBuffer doubleBufffer = createDataDoubleBufffer();
        environment.write(dataBuffer, doubleBufffer);
    }

    private long forward(CLTrainingBatch batch, CLMemObject expectedIndices, IntBuffer outputs) {
        int index = 0;
        propagateKernel.setArg(index++, networkBuffer);
        propagateKernel.setArg(index++, dataBuffer);

        propagateKernel.setArg(index++, batch.offset);
        propagateKernel.setArg(index++, batch.getInputs());
        propagateKernel.setArg(index++, expectedIndices);

        if (outputs != null)
            ensureOutputs(batch.batchSize);
        propagateKernel.setArg(index++, outputsBuffer);
        if (expectedIndices != null)
            ensureMatchedCount(batch.batchSize);
        propagateKernel.setArg(index++, expectedIndices != null ? matchedCountBuffer : null);

        propagateKernel.setArgSize(index++, workGroupSize * TYPE_SIZE);
        propagateKernel.setArgSize(index, outputs() * Integer.BYTES);

        range.globalWorkSizes(localWorkSize[0], localWorkSize[1] * batch.batchSize).requestEvent();
        return environment.run(propagateKernel, range);
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
        sumRange.globalWorkSizes(groups * groupSize, totalWeights() + totalNeurons())
                .localWorkSizes(groupSize, 1)
                .waitForEvent(event)
                .requestEvent();

        int index = 0;
        sumGradsKernel.setArg(index++, networkBuffer);
        sumGradsKernel.setArg(index++, size);
        sumGradsKernel.setArg(index++, gradsBuffer);
        sumGradsKernel.setArgSize(index, groupSize * TYPE_SIZE);
        return environment.run(sumGradsKernel, sumRange);
    }

    private long updateNetwork(long event, double lr) {
        sumRange.globalWorkSizes(totalWeights() + totalNeurons())
                .localWorkSizes()
                .waitForEvent(event)
                .requestEvent();
        int index = 0;
        updateNetworkKernel.setArg(index++, networkBuffer);
        updateNetworkKernel.setArg(index++, lr);
        updateNetworkKernel.setArg(index++, dataBuffer);
        updateNetworkKernel.setArg(index, gradsBuffer);
        return environment.run(updateNetworkKernel, sumRange);
    }

    private void readOutputs(CLTrainingBatch batch, IntBuffer outputs) {
        outputs.position(batch.offset).limit(batch.offset + batch.batchSize);
        environment.read(outputsBuffer, outputs, true, 0, range);
    }

    private void ensureOutputs(int size) {
        if (outputsCapacity < size) {
            if (outputsBuffer != null)
                outputsBuffer.close();
            outputsBuffer = environment.mem(size * Integer.BYTES,
                    //BufferProperties.MEM_ALLOC_HOST_PTR,
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
            if (gradsBuffer != null)
                gradsBuffer.close();

            gradsBuffer = environment.mem(size * (totalWeights() + totalNeurons())  * TYPE_SIZE,
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
