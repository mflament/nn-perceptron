/**
 * 
 */
package org.yah.tests.perceptron.opencl;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.function.Consumer;

import org.junit.After;
import org.junit.Before;
import org.yah.tests.perceptron.AbstractNeuralNetworkTest;
import org.yah.tests.perceptron.SamplesProviders;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tools.opencl.context.CLContext;

/**
 * @author Yah
 *
 */
public class CLNeuralNetworkTest extends AbstractNeuralNetworkTest<CLNeuralNetwork> {

    private CLContext context;

    @Before
    public void setup() throws IOException {
        context = CLContext.createDefault((msg, data) -> System.out.println(msg));
    }

    @After
    public void release() {
        if (context != null)
            context.close();
    }

    @Override
    protected void withNetwork(Consumer<CLNeuralNetwork> consumer, int... layerSizes) {
        try (CLNeuralNetwork network = new CLNeuralNetwork(context, layerSizes)) {
            consumer.accept(network);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
    
    public static void main(String[] args) throws IOException {
        try (CLNeuralNetwork network = new CLNeuralNetwork(2,2,2)) {
            double[][] inputs = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
            int[] expectedIndices = new int[] { 0, 1, 1, 0 };
            TrainingSamplesProvider provider = SamplesProviders.newTrainingProvider(inputs, false,
                    expectedIndices);
            SamplesSource sampleSource = network.createSampleSource();
            TrainingSamples samples = sampleSource.createTraining(provider, 0);
            int[] outputs = new int[samples.size()]; 
            double score = network.evaluate(samples, outputs);
            System.out.println(Arrays.toString(outputs));
            System.out.println(score);
        }
    }
}
