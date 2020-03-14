/**
 * 
 */
package perceptronorg.yah.tests.perceptron;

import static org.yah.tests.perceptron.array.ArrayMatrix.transpose;

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.array.ArrayBatch;
import org.yah.tests.perceptron.array.ArrayMatrixNeuralNetwork;

/**
 * @author Yah
 *
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
public class TrainingBenchmark {

    private static final double[][] INPUTS = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    private static final int[] OUTPUTS = { 0, 1, 1, 0 };

    @org.openjdk.jmh.annotations.State(Scope.Benchmark)
    public static class ArrayNetworkInput {
        NeuralNetwork network;
        ArrayBatch batch;
        @Setup(Level.Trial)
        public void setup() {
            network = new ArrayMatrixNeuralNetwork(2, 2, 2);
            batch = new ArrayBatch(transpose(INPUTS), OUTPUTS, network.outputs());
        }
    }
    
    @Benchmark
    public void array_training(ArrayNetworkInput input) {
        input.network.train(input.batch, 0.1);
    }

}
