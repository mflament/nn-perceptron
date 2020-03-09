/**
 * 
 */
package perceptronorg.yah.tests.perceptron;

import static org.yah.tests.perceptron.Matrix.transpose;

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.NeuralNetwork.Batch;

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
    public static class TrainingInput {
        NeuralNetwork network;
        Batch batch;
        @Setup(Level.Trial)
        public void setup() {
            network = new NeuralNetwork(2, 2, 2);
            batch = new Batch(transpose(INPUTS), OUTPUTS, network.outputs());
        }
    }

    @Benchmark
    public void training(TrainingInput input) {
        input.network.train(input.batch, 0.1);
    }

}
