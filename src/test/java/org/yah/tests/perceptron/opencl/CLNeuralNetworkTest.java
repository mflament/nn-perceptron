package org.yah.tests.perceptron.opencl;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.NeuralNetworkState;
import org.yah.tests.perceptron.base.AbstractNeuralNetworkTest;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork;
import org.yah.tools.opencl.context.CLContext;

import java.io.IOException;

/**
 * @author Yah
 */
public class CLNeuralNetworkTest extends AbstractNeuralNetworkTest {

    private static CLContext context;

    @BeforeClass
    public static void setupContext() {
        context = CLContext.createDefault((msg, data) -> System.out.println(msg));
    }

    @AfterClass
    public static void closeContext() {
        if (context != null)
            context.close();
    }

    @Override
    protected NeuralNetwork newNetwork(NeuralNetworkState state) {
        try {
            return new CLNeuralNetwork(context, state);
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    @Override
    protected void updateState(NeuralNetworkState network) {
        ((CLNeuralNetwork) network).updateState();
    }

    @Override
    protected void updateModel(NeuralNetworkState network) {
        ((CLNeuralNetwork) network).updateModel();
    }

}
