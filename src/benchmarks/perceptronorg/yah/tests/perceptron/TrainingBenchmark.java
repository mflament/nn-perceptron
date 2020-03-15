/**
 * 
 */
package perceptronorg.yah.tests.perceptron;

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.matrix.MatrixBatch;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;
import org.yah.tests.perceptron.matrix.array.RMArrayMatrix;
import org.yah.tests.perceptron.matrix.flat.CMFlatMatrix;

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
    public static class CMArrayNetworkInput {
        NeuralNetwork<MatrixBatch<CMArrayMatrix>> network;
        MatrixBatch<CMArrayMatrix> batch;

        @Setup(Level.Trial)
        public void setup() {
            network = new MatrixNeuralNetwork<>(CMArrayMatrix::new, 2, 2, 2);
            batch = network.createBatchSource().createBatch(INPUTS, OUTPUTS, true);
        }
    }

    @org.openjdk.jmh.annotations.State(Scope.Benchmark)
    public static class RMArrayNetworkInput {
        NeuralNetwork<MatrixBatch<RMArrayMatrix>> network;
        MatrixBatch<RMArrayMatrix> batch;

        @Setup(Level.Trial)
        public void setup() {
            network = new MatrixNeuralNetwork<>(RMArrayMatrix::new, 2, 2, 2);
            batch = network.createBatchSource().createBatch(INPUTS, OUTPUTS, true);
        }
    }
    
    @org.openjdk.jmh.annotations.State(Scope.Benchmark)
    public static class CMFlatNetworkInput {
        NeuralNetwork<MatrixBatch<CMFlatMatrix>> network;
        MatrixBatch<CMFlatMatrix> batch;

        @Setup(Level.Trial)
        public void setup() {
            network = new MatrixNeuralNetwork<>(CMFlatMatrix::new, 2, 2, 2);
            batch = network.createBatchSource().createBatch(INPUTS, OUTPUTS, true);
        }
    }

    @Benchmark
    public void cm_array_training(CMArrayNetworkInput input) {
        input.network.train(input.batch, 0.1);
    }

    @Benchmark
    public void rm_array_training(RMArrayNetworkInput input) {
        input.network.train(input.batch, 0.1);
    }

    @Benchmark
    public void cm_flat_training(CMFlatNetworkInput input) {
        input.network.train(input.batch, 0.1);
    }

}
