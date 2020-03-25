/**
 * 
 */
package perceptronorg.yah.tests.perceptron;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.infra.Blackhole;
import org.yah.tests.perceptron.mt.MTMatrix;

/**
 * @author Yah
 *
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
public class MatrixBenchmark {

    private static final int ROWS = 256;

    @org.openjdk.jmh.annotations.State(Scope.Benchmark)
    public static class MatrixInput {
        MTMatrix matrix;

        @Setup(Level.Trial)
        public void setup() {
            Random random = new Random();
            matrix = new MTMatrix(ROWS, 1);
            for (int r = 0; r < ROWS; r++) {
                matrix.set(r, 0, random.nextDouble());
            }
        }
    }

    @Benchmark
    public void maxRowIndex(MatrixInput input, Blackhole bh) {
        bh.consume(input.matrix.maxRowIndex(0));
    }

}
