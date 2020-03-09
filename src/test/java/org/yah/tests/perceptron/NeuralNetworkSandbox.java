package org.yah.tests.perceptron;

import static org.yah.tests.perceptron.NeuralNetwork.ArrayBatchSource;

import org.yah.tests.perceptron.NeuralNetwork.Batch;
import org.yah.tests.perceptron.NeuralNetwork.BatchSource;

public class NeuralNetworkSandbox {

    private static final BatchSource XOR_BATCHES = new ArrayBatchSource(
            new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },  new int[] { 0, 1, 1, 0 });

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 2, 2);
        Iterable<Batch> iterable = nn.batchIterable(XOR_BATCHES);
        float score = nn.evaluate(iterable.iterator());
        System.out.println(score);
        while (true) {
            nn.train(10000, iterable, 0.1f);
            score = nn.evaluate(iterable.iterator());
            System.out.println(score);
        }
    }
}
