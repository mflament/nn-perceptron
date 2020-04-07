package org.yah.tests.perceptron.opencl;

import org.yah.tests.perceptron.matrix.nd.NDMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Stream;

public class MatrixLogParser {

    private static class LogMatrix {
        private final String name;
        private final NDMatrix matrix;

        public LogMatrix(String name, int dimensions) {
            this.name = name;
            this.matrix = NDMatrix.withDimensions(dimensions);
        }

        public boolean match(String name, int dimensions) {
            return name.equals(this.name) && matrix.dimensions() == dimensions;
        }

        public int length(int index) {
            return matrix.length(index);
        }

        public void set(double v, int... indices) {
            matrix.set(v, indices);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            matrix.visit((indices, value) -> {
                int dim = indices.length - 1;
                if (indices[dim] == 0) {
                    if (matrix.dimensions() == 1)
                        sb.append(name).append(System.lineSeparator());
                    else if (indices[indices.length - 2] == 0) {
                        sb.append(name);
                        for (int i = 0; i < indices.length - 2; i++)
                            sb.append('[').append(indices[i]).append(']');
                        sb.append(System.lineSeparator());
                    }
                }
                sb.append(String.format("%7.3f ", value));
                if (indices[dim] == matrix.length(dim) - 1)
                    sb.append(System.lineSeparator());
            });
            return sb.toString();
        }
    }

    private final List<LogMatrix> matrices = new ArrayList<>();

    private void parse(Stream<String> lines) {
        lines.forEach(line -> {
            String[] parts = line.split(";");
            if (parts.length > 0) {
                String name = parts[0];
                int[] indices = new int[parts.length - 2];
                for (int i = 0; i < indices.length; i++) {
                    indices[i] = Integer.parseInt(parts[i + 1]);
                }
                double value = Double.parseDouble(parts[parts.length - 1]);
                LogMatrix matrix = findOrCreate(name, indices.length);
                matrix.set(value, indices);
            }
        });
    }

    private LogMatrix findOrCreate(String name, int dimensions) {
        return matrices.stream()
                .filter(m -> m.match(name, dimensions))
                .findFirst()
                .orElseGet(() -> newMatrix(name, dimensions));
    }

    private LogMatrix newMatrix(String name, int dimensions) {
        LogMatrix matrix = new LogMatrix(name, dimensions);
        this.matrices.add(matrix);
        return matrix;
    }

    private void printMatrices(Predicate<LogMatrix> selector) {
        matrices.stream().filter(selector).forEach(System.out::println);
    }

    private void printMatrices() {
        printMatrices(m -> true);
    }

    public static void main(String[] args) throws IOException {
        MatrixLogParser parser = new MatrixLogParser();
        try (BufferedReader reader = new BufferedReader(new FileReader("logs.txt"))) {
            parser.parse(reader.lines());
        }
        parser.printMatrices();
    }

}
