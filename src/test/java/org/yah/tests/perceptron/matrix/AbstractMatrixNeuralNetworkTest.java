package org.yah.tests.perceptron.matrix;

import org.junit.Before;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.NeuralNetworkState;
import org.yah.tests.perceptron.base.AbstractNeuralNetworkTest;
import org.yah.tests.perceptron.base.DefaultNetworkState;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork.MatrixFactory;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;
import org.yah.tests.perceptron.mt.MTNeuralNetwork;

import java.util.function.Consumer;
import java.util.function.DoubleSupplier;

/**
 * @author Yah
 */
public abstract class AbstractMatrixNeuralNetworkTest<M extends Matrix<M>> extends AbstractNeuralNetworkTest {

    protected MatrixFactory<M> matrixFactory;

    @Before
    public void setup() {
        super.setup();
        matrixFactory = createMatrixFactory();
    }

    @Override
    protected NeuralNetwork newNetwork(NeuralNetworkState state) {
        return new MatrixNeuralNetwork<>(matrixFactory, state);
    }

    @Override
    protected void updateState(NeuralNetworkState network) {
        ((MatrixNeuralNetwork<?>) network).updateState();
    }

    @Override
    protected void updateModel(NeuralNetworkState network) {
        ((MatrixNeuralNetwork<?>) network).updateModel();
    }

    protected abstract MatrixFactory<M> createMatrixFactory();

}
