package org.yah.tests.perceptron.base;

import org.junit.Test;
import org.yah.tests.perceptron.AbstractNetworkStateTest;
import org.yah.tests.perceptron.NeuralNetworkState;

import java.util.function.DoubleSupplier;

import static org.junit.Assert.*;

public class DefaultNetworkStateTest extends AbstractNetworkStateTest {

    @Override
    protected NeuralNetworkState newState(DoubleSupplier randomSource, int[] layers) {
        return new DefaultNetworkState(randomSource, layers);
    }

    @Override
    protected NeuralNetworkState newState(NeuralNetworkState from) {
        return new DefaultNetworkState(from);
    }
}