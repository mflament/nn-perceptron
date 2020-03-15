/**
 * 
 */
package org.yah.tests.perceptron.jni;

import org.yah.tests.perceptron.InputSamples;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;

/**
 * @author Yah
 *
 */
public class NativeNeuralNetwork implements NeuralNetwork {

    /**
     * 
     */
    public NativeNeuralNetwork() {
        // TODO Auto-generated constructor stub
    }

    @Override
    public int layers() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public int features() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public int outputs() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public int features(int layer) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public int neurons(int layer) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public SamplesSource createSampleSource() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void propagate(InputSamples samples, int[] outputs) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public double evaluate(TrainingSamples samples, int[] outputs) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public void train(TrainingSamples samples, double learningRate) {
        // TODO Auto-generated method stub
        
    }

}
