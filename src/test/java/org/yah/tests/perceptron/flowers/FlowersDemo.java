package org.yah.tests.perceptron.flowers;

import java.util.BitSet;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.yah.tests.perceptron.AbstractGLDemo;
import org.yah.tests.perceptron.Batch;
import org.yah.tests.perceptron.Batch.BatchSource;
import org.yah.tests.perceptron.GLDemoLauncher;
import org.yah.tests.perceptron.JavaNeuralNetwork;
import org.yah.tests.perceptron.Labels;
import org.yah.tests.perceptron.Matrix;
import org.yah.tests.perceptron.NeuralNetwork;

import com.badlogic.gdx.Input;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.Pixmap;

public class FlowersDemo extends AbstractGLDemo {

    private static final int WIDTH = 1024;
    private static final int HEIGHT = 900;
    private static final double NOISE_SCALE = 3f;

    private static final int FLOWERS = WIDTH * HEIGHT;

    private static final int[] LAYERS = { 2, 18, 2 };
    private static final int SAMPLES = (int) (FLOWERS * 0.05);
    private static final int BATCH_SIZE = 64;
    private static final double LEARNING_RATE = 0.8f;

    private ExecutorService executor;

    private int[] flowers = new int[FLOWERS];

    private int[] FLOWER_COLORS = { 0xee0000ff, 0x00ee00ff };
    private int[] DARKER_FLOWER_COLORS = { 0xcc0000ff, 0x00cc00ff };
    private int[] SAMPLED_FLOWER_COLORS = { 0x000000ff, 0x000000ff };

    private double[][] allFlowers = new double[2][FLOWERS]; // inputs = LAYERS[0] x flowers
    private int[] outputs = new int[FLOWERS]; // outputs = LAYERS[-1] x flowers

    private FlowerBatchSource batchSource;

    private Random random;

    private NeuralNetwork network;
    private long lastLog;
    private int epoch;

    private boolean paused = true;
    private boolean destroyed;

    public FlowersDemo() {}

    @Override
    public void create() {
        super.create();
        executor = Executors.newSingleThreadExecutor();
        createBuffer(WIDTH, HEIGHT);
        random = JavaNeuralNetwork.RANDOM;
        network = new JavaNeuralNetwork(LAYERS);
        batchSource = new FlowerBatchSource(SAMPLES, BATCH_SIZE);
        executor.submit(this::createFlowers);
    }

    @Override
    public void configure(Lwjgl3ApplicationConfiguration config) {
        config.setWindowedMode(WIDTH, HEIGHT);
    }

    @Override
    public void dispose() {
        destroyed = true;
        executor.shutdownNow();
        super.dispose();
    }

    private void createFlowers() {
        long seed = JavaNeuralNetwork.seed();
        OpenSimplexNoise noise = new OpenSimplexNoise(seed < 0 ? System.currentTimeMillis() : seed);
        int flowerIndex = 0;
        for (int y = 0; y < HEIGHT; y++) {
            double dy = y / (double) HEIGHT;
            for (int x = 0; x < WIDTH; x++) {
                double dx = x / (double) WIDTH;
                double n = noise.eval(dx * NOISE_SCALE, dy * NOISE_SCALE);
                n = (n + 1) / 2.0;
                flowers[flowerIndex] = (int) (n + 0.5);
                allFlowers[0][flowerIndex] = dx;
                allFlowers[1][flowerIndex] = dy;
                flowerIndex++;
            }
        }

        evaluate();

        schedule(this::renderFlowers);
        lastLog = System.currentTimeMillis();
        executor.submit(this::train);
    }

    @Override
    public boolean keyDown(int keycode) {
        switch (keycode) {
        case Input.Keys.SPACE:
            paused = !paused;
            return true;
        default:
            return super.keyDown(keycode);
        }
    }

    private void evaluate() {
        long elapsed = System.currentTimeMillis() - lastLog;
        if (elapsed > 500) {
            synchronized (outputs) {
                network.propagate(allFlowers, outputs);
            }
            schedule(this::renderFlowers);

            double overallAccuracy = Labels.countMatched(flowers, outputs) / (double) FLOWERS;
            double es = elapsed / 1000.0;
            System.out.println(String.format("%.2f%%(%.2f%%) %.2f e/s %.2f Ms/s",
                    overallAccuracy * 100, network.accuracy(),
                    epoch / es, (epoch * SAMPLES / 1000000.0) / es));
            lastLog = System.currentTimeMillis();
            epoch = 0;
        }
    }

    private void train() {
        try {
            while (!destroyed) {
                while (paused)
                    Thread.sleep(100);
                network.train(Batch.iterator(network, batchSource), LEARNING_RATE);
                epoch++;
                evaluate();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void renderFlowers(Pixmap pixmap) {
        synchronized (outputs) {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    int flowerIndex = y * WIDTH + x;
                    int flower = flowers[flowerIndex];
                    int flowerColor;
                    if (batchSource.sampled.get(flowerIndex))
                        flowerColor = SAMPLED_FLOWER_COLORS[flower];
                    else if (outputs[flowerIndex] != flower)
                        flowerColor = DARKER_FLOWER_COLORS[flower];
                    else
                        flowerColor = FLOWER_COLORS[flower];
                    pixmap.drawPixel(x, y, flowerColor);
                }
            }
        }
    }

    private class FlowerBatchSource implements BatchSource {

        private final int[] samples;

        private final int size, batchSize;

        private final BitSet sampled;

        public FlowerBatchSource(int size, int batchSize) {
            this.samples = new int[size];
            this.size = size;
            this.batchSize = Math.min(size, batchSize);

            int count = 0;
            sampled = new BitSet(FLOWERS);
            while (count < size) {
                samples[count] = random.nextInt(FLOWERS);
                if (sampled.get(samples[count]))
                    continue;
                sampled.set(samples[count]);
                count++;
            }
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int batchSize() {
            return batchSize;
        }

        @Override
        public void load(int index, int size, Batch batch) {
            Matrix.zero(batch.expectedMatrix);
            for (int i = 0; i < size; i++) {
                int flowerIndex = samples[index + i];
                int x = flowerIndex % WIDTH;
                int y = flowerIndex / WIDTH;
                batch.inputs[0][i] = x / (double) WIDTH;
                batch.inputs[1][i] = y / (double) HEIGHT;
                int flower = flowers[flowerIndex];
                batch.expectedIndices[i] = flower;
                batch.expectedMatrix[flower][i] = 1;
            }
        }
    }

    public static void main(String[] args) {
        GLDemoLauncher.launch(new FlowersDemo());
    }
}
