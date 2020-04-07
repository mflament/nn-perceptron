package org.yah.tests.perceptron.base;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.yah.tests.perceptron.AbstractNetworkStateTest;
import org.yah.tests.perceptron.NetworkDump;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.NeuralNetworkState;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleSupplier;

import static org.junit.Assert.assertEquals;

public abstract class AbstractNeuralNetworkTest extends AbstractNetworkStateTest {

    private List<AutoCloseable> resources;

    @Before
    public void setup() {
        super.setup();
        resources = new ArrayList<>();
    }

    @After
    public void close() {
        for (AutoCloseable resource : resources) {
            try {
                resource.close();
            } catch (Exception ignored) {
            }
        }
    }

    @Override
    protected final NeuralNetworkState newState(DoubleSupplier randomSource, int[] layers) {
        NeuralNetwork neuralNetwork = newNetwork(new DefaultNetworkState(randomSource, layers));
        if (neuralNetwork instanceof AutoCloseable)
            resources.add((AutoCloseable) neuralNetwork);
        return neuralNetwork;
    }

    @Override
    protected final NeuralNetworkState newState(NeuralNetworkState from) {
        NeuralNetwork neuralNetwork = newNetwork(from);
        if (neuralNetwork instanceof AutoCloseable)
            resources.add((AutoCloseable) neuralNetwork);
        return neuralNetwork;
    }

    @Test
    public void updateState() {
        NeuralNetwork neuralNetwork = newNetwork(newState(2, 3, 2));
        double[][][] expecteds = createExepectedWeights(neuralNetwork);
        updateModel(neuralNetwork);
        updateState(neuralNetwork);
        NeuralNetworkState state = neuralNetwork.getState();
        state.visitWeights((layer, neuron, feature) ->
                assertEquals(expecteds[layer][neuron][feature], state.weight(layer, neuron, feature), DELTA));
    }

    @Test
    public void testDump() throws IOException {
        NetworkDump dump = NetworkDump.load(Paths.get("dumps/test.dump"));
        dump.test(newNetwork(dump.state));
    }

    protected abstract NeuralNetwork newNetwork(NeuralNetworkState state);

    protected abstract void updateState(NeuralNetworkState network);

    protected abstract void updateModel(NeuralNetworkState network);

}
