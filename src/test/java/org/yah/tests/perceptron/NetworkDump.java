package org.yah.tests.perceptron;

import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.base.DefaultNetworkState;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;
import org.yah.tests.perceptron.opencl.CLNeuralNetwork;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.yah.tests.perceptron.SamplesProviders.newTrainingProvider;

public class NetworkDump {

    public static final double DELTA = 1E-6;

    public final NeuralNetworkState state;
    private final double[][] inputs;
    private final int[] expectedOutputs;
    private final int epochs;
    private final int batchSize;
    private final double learningRate;

    private final List<NeuralNetworkState> epochStates;

    private NetworkDump(NeuralNetworkState initialState,
                        double[][] inputs,
                        int[] expectedOutputs,
                        int epochs,
                        int batchSize,
                        double learningRate,
                        List<NeuralNetworkState> epochStates) {
        this.state = initialState;
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.epochStates = epochStates;
    }

    public void test(NeuralNetwork network) {
        assertState(0, state, network.getState());
        TrainingSamplesProvider provider = newTrainingProvider(inputs, false, expectedOutputs);
        TrainingSamples samples = network.createTraining(provider, batchSize);
        for (int epoch = 0; epoch < epochs; epoch++) {
            network.train(samples, learningRate);
            assertState(epoch, epochStates.get(epoch), network.getState());
        }
    }

    private static void assertState(int epoch, NeuralNetworkState expected, NeuralNetworkState actual) {
        assertEquals("layers", expected.layers(), actual.layers());
        assertEquals("features", expected.features(), actual.features());
        for (int layer = 0; layer < expected.layers(); layer++) {
            assertEquals("neurons[" + layer + "]", expected.neurons(layer), actual.neurons(layer));
        }
        expected.visitWeights((layer, neuron, feature) ->
                assertEquals(String.format("epoch[%d] weight[%d][%d][%d]", epoch, layer, neuron, feature),
                        expected.weight(layer, neuron, feature),
                        actual.weight(layer, neuron, feature), DELTA)
        );
        expected.visitBiases((layer, neuron) ->
                assertEquals(String.format("epoch[%d] bias[%d][%d]", epoch, layer, neuron),
                        expected.bias(layer, neuron),
                        actual.bias(layer, neuron), DELTA)
        );
    }

    public void save(Path target) throws IOException {
        try (DataOutputStream os = outputStream(target)) {
            os.writeInt(state.layers());
            os.writeInt(state.features());
            for (int i = 0; i < state.layers(); i++) {
                os.writeInt(state.neurons(i));
            }
            state.visitWeights((layer, neuron, feature) -> silentWrite(os, state.weight(layer, neuron, feature)));
            state.visitBiases((layer, neuron) -> silentWrite(os, state.bias(layer, neuron)));
            os.writeInt(inputs.length);
            for (double[] input : inputs) {
                for (double v : input) {
                    os.writeDouble(v);
                }
            }
            for (int expectedOutput : expectedOutputs) {
                os.writeInt(expectedOutput);
            }
            os.writeInt(epochs);
            os.writeInt(batchSize);
            os.writeDouble(learningRate);
            epochStates.forEach(s -> writeState(s, os));
        }
    }


    public static NetworkDump load(Path path) throws IOException {
        try (DataInputStream is = inputStream(path)) {
            int layers = is.readInt();
            int[] layerSizes = readInts(is, layers + 1);
            NeuralNetworkState state = new DefaultNetworkState(layerSizes);
            state.visitWeights((layer, neuron, feature) -> state.weight(layer, neuron, feature, silentRead(is)));
            state.visitBiases((layer, neuron) -> state.bias(layer, neuron, silentRead(is)));
            int samples = is.readInt();
            double[][] inputs = new double[samples][state.features()];
            for (int sample = 0; sample < samples; sample++) {
                for (int f = 0; f < state.features(); f++) {
                    inputs[sample][f] = is.readDouble();
                }
            }
            int[] expectedOutputs = new int[samples];
            for (int sample = 0; sample < samples; sample++) {
                expectedOutputs[sample] = is.readInt();
            }
            int epochs = is.readInt();
            int batchSize = is.readInt();
            double learningRate = is.readDouble();
            List<NeuralNetworkState> epochStates = new ArrayList<>(epochs);
            for (int e = 0; e < epochs; e++) {
                epochStates.add(readState(state, is));
            }
            return new NetworkDump(state, inputs, expectedOutputs, epochs, batchSize, learningRate, epochStates);
        }
    }

    private static void writeState(NeuralNetworkState state, DataOutputStream os) {
        state.visitWeights((layer, neuron, feature) -> silentWrite(os, state.weight(layer, neuron, feature)));
        state.visitBiases((layer, neuron) -> silentWrite(os, state.bias(layer, neuron)));
    }

    private static NeuralNetworkState readState(NeuralNetworkState state, DataInputStream is) {
        NeuralNetworkState epochState = new DefaultNetworkState(state);
        state.visitWeights((layer, neuron, feature) -> epochState.weight(layer, neuron, feature, silentRead(is)));
        state.visitBiases((layer, neuron) -> epochState.bias(layer, neuron, silentRead(is)));
        return epochState;
    }

    private static int[] readInts(DataInputStream is, int length) throws IOException {
        int[] res = new int[length];
        for (int i = 0; i < res.length; i++) {
            res[i] = is.readInt();
        }
        return res;
    }

    private static DataInputStream inputStream(Path path) throws FileNotFoundException {
        return new DataInputStream(new BufferedInputStream(new FileInputStream(path.toFile())));
    }

    private DataOutputStream outputStream(Path target) throws FileNotFoundException {
        return new DataOutputStream(new BufferedOutputStream(new FileOutputStream(target.toFile())));
    }

    public static NetworkDump create(NeuralNetwork network,
                                     TrainingSamplesProvider provider,
                                     int batchSize,
                                     int epochs,
                                     double learningRate) {
        NeuralNetworkState initialState = new DefaultNetworkState(network);
        double[][] inputs = new double[provider.samples()][network.features()];
        int[] expectedOutputs = new int[provider.samples()];
        for (int sample = 0; sample < provider.samples(); sample++) {
            for (int f = 0; f < network.features(); f++) {
                inputs[sample][f] = provider.input(sample, f);
            }
            expectedOutputs[sample] = provider.outputIndex(sample);
        }
        TrainingSamples samples = network.createTraining(provider, batchSize);
        List<NeuralNetworkState> epochStates = new ArrayList<>(epochs);
        for (int epoch = 0; epoch < epochs; epoch++) {
            network.train(samples, learningRate);
            epochStates.add(network.getState());
        }
        return new NetworkDump(initialState, inputs, expectedOutputs, epochs, batchSize, learningRate, epochStates);
    }

    private static void silentWrite(DataOutputStream os, double d) {
        try {
            os.writeDouble(d);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static double silentRead(DataInputStream is) {
        try {
            return is.readDouble();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static NetworkDump createDump(int[] layers, int batchSize, int epochs) throws IOException {
        if (layers[0] != 2) throw new IllegalArgumentException("network features must be 2");
        NeuralNetworkState state = new DefaultNetworkState(RandomUtils.newRandomSource(123456), layers);
        NetworkDump networkDump = NetworkDump.create(new MatrixNeuralNetwork<>(CMArrayMatrix::new, state),
                NeuralNetworkSandbox.NAND,
                batchSize, epochs, 0.1);
        StringBuilder sb = new StringBuilder();
        sb.append("dumps/matrix_n");
        for (int layer : layers) {
            sb.append(layer).append('_');
        }
        sb.append('b').append(batchSize).append(".dump");
        networkDump.save(Paths.get(sb.toString()));
        return networkDump;
    }

    public static void main(String[] args) throws IOException {
        //NetworkDump dump = createDump(new int[]{2, 2}, 0, 1);
//        NetworkDump dump = createDump(new int[]{2, 3, 2}, 0, 1);
        System.out.println("PID: " + ProcessHandle.current().pid());
        NetworkDump dump = NetworkDump.load(Paths.get("dumps/matrix_n2_3_2_b0.dump"));
        //NetworkDump dump = NetworkDump.load(Paths.get("dumps/matrix_n2_2_b0.dump"));
        dump.test(new CLNeuralNetwork(dump.state));
    }

}
