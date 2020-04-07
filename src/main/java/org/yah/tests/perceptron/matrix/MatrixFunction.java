package org.yah.tests.perceptron.matrix;

@FunctionalInterface
public
interface MatrixFunction {
    double apply(int row, int column, double value);
}
