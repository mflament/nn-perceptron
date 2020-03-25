package perceptronorg.yah.tests.perceptron;

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
public class NetworkBenchmark {

    @org.openjdk.jmh.annotations.State(Scope.Benchmark)
    public abstract static class NetworkInput {
        
    }
    
}
