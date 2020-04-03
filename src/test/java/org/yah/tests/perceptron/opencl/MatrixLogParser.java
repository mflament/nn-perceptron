package org.yah.tests.perceptron.opencl;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class MatrixLogParser {

    private static final Pattern LINE_PATTERN = Pattern.compile("(\\w+)\\[(\\d+)\\]\\[(\\d+)\\]\\[(\\d+)\\](?:\\[(\\d+)\\])?\\s*=\\s*(.*)");

    private static class Matrix {
        private final double data[][] = new double[32][32];
        private int rows, columns;

        public double get(int row, int col) {
            return data[row][col];
        }

        public void set(int row, int col, double v) {
            data[row][col] = v;
            rows = Math.max(rows, row + 1);
            columns = Math.max(columns, col + 1);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    sb.append(String.format("%7.3f ", get(r, c)));
                }
                sb.append(System.lineSeparator());
            }
            return sb.toString();
        }

        public void add(Matrix mx) {
            rows = mx.rows;
            columns = mx.columns;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    data[r][c] += mx.get(r, c);
                }
            }
        }
    }

    private final Map<String, Matrix[][]> matrices = new LinkedHashMap<>();

    private void parse(Stream<String> lines) {
        lines.forEach(line -> {
            if (line.isBlank())
                return;
            Matcher matcher = LINE_PATTERN.matcher(line);
            if (matcher.matches()) {
                int index = 1;
                String name = matcher.group(index++);
                int sample = Integer.parseInt(matcher.group(index++));
                int layer = Integer.parseInt(matcher.group(index++));
                int row = Integer.parseInt(matcher.group(index++));
                int col = matcher.group(index) != null ? Integer.parseInt(matcher.group(index)) : 0;
                index++;
                double value = Double.parseDouble(matcher.group(index));
                Matrix[][] mxs = matrices.get(name);
                if (mxs == null)
                    matrices.put(name, mxs = new Matrix[2][4]);
                if (mxs[layer][sample] == null)
                    mxs[layer][sample] = new Matrix();
                mxs[layer][sample].set(row, col, value);
            } else
                throw new IllegalArgumentException("Invalid line " + line);
        });
    }

    private Matrix sum(String name, int layer) {
        Matrix[][] mxs = matrices.get(name);
        Matrix res = new Matrix();
        for (Matrix mx : mxs[layer]) {
            res.add(mx);
        }
        return res;
    }

    private void printMatrices(String name) {
        Matrix[][] mxs = matrices.get(name);
        for (int layer = 0; layer < 2; layer++) {
            for (int sample = 0; sample < 4; sample++) {
                if (mxs[layer][sample] != null) {
                    System.out.println(String.format("%s layer %d sample %d", name, layer, sample));
                    System.out.println(mxs[layer][sample]);
                }
            }
        }

    }
    private void printMatrices() {
        matrices.keySet().forEach(this::printMatrices);
    }

    public static void main(String[] args) throws IOException {
        MatrixLogParser parser = new MatrixLogParser();
        try (BufferedReader reader = new BufferedReader(new FileReader("logs.txt"))) {
            parser.parse(reader.lines());
        }
        System.out.print("bgrad 1\n" + parser.sum("bgrad", 1));
        System.out.print("bgrad 0\n" + parser.sum("bgrad", 0));

        System.out.print("wgrad 1\n" + parser.sum("wgrad", 1));
        System.out.print("wgrad 0\n" + parser.sum("wgrad", 0));

        parser.printMatrices("bgrad");

        parser.printMatrices("input");

    }

}
