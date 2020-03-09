package org.yah.tests.perceptron.flowers;

import java.util.BitSet;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.yah.tests.perceptron.AbstractGLDemo;
import org.yah.tests.perceptron.GLDemoLauncher;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.NeuralNetwork.Batch;
import org.yah.tests.perceptron.NeuralNetwork.Labels;

import com.badlogic.gdx.Input;
import com.badlogic.gdx.graphics.Pixmap;

public class FlowersDemo extends AbstractGLDemo {

    private static final int WIDTH = 640;
    private static final int HEIGHT = 480;
    private static final double NOISE_SCALE = 3f;

    private static final int FLOWERS = WIDTH * HEIGHT;

    private static final int[] LAYERS = { 2, 16, 2 };
    private static final int SAMPLES = FLOWERS / 60;
    private static final int BATCH_SIZE = 32;
    private static final float LEARNING_RATE = 0.5f;

    private ExecutorService executor;

    private int[] flowers = new int[FLOWERS];

    private int[] FLOWER_COLORS = { 0xee0000ff, 0x00ee00ff };
    private int[] DARKER_FLOWER_COLORS = { 0xcc0000ff, 0x00cc00ff };
    private int[] SAMPLED_FLOWER_COLORS = { 0x000000ff, 0x000000ff };

    private float[][] allFlowers = new float[2][FLOWERS]; // inputs = LAYERS[0] x flowers
    private float[][] outputs = new float[2][FLOWERS]; // outputs = LAYERS[-1] x flowers

    private final int[] evaluatedFlowers = new int[FLOWERS];

    private FlowerBatchSource batchSource;

    private Random random;

    private NeuralNetwork network;
    private long lastLog;
    private int epoch;

    private boolean paused;
    private boolean destroyed;

    public FlowersDemo() {}

    @Override
    public void create() {
        super.create();
        executor = Executors.newSingleThreadExecutor();
        createBuffer(WIDTH, HEIGHT);
        random = NeuralNetwork.RANDOM;
        network = new NeuralNetwork(LAYERS);
        batchSource = new FlowerBatchSource(SAMPLES, BATCH_SIZE);
        executor.submit(this::createFlowers);
    }

    @Override
    public void dispose() {
        destroyed = true;
        executor.shutdownNow();
        super.dispose();
    }

    private void createFlowers() {
        long seed = NeuralNetwork.seed();
        OpenSimplexNoise noise = new OpenSimplexNoise(seed < 0 ? System.currentTimeMillis() : seed);
        int flowerIndex = 0;
        for (int y = 0; y < HEIGHT; y++) {
            float dy = y / (float) HEIGHT;
            for (int x = 0; x < WIDTH; x++) {
                float dx = x / (float) WIDTH;
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
            network.propagate(allFlowers, outputs);
            synchronized (evaluatedFlowers) {
                for (int flowerIndex = 0; flowerIndex < FLOWERS; flowerIndex++) {
                    evaluatedFlowers[flowerIndex] = NeuralNetwork.maxRowIndex(outputs, flowerIndex);
                }
            }
            schedule(this::renderFlowers);

            float accuracy = Labels.countMatched(flowers, evaluatedFlowers) / (float) FLOWERS;
            double es = elapsed / 1000.0;
            System.out
                    .println(String.format("%.2f%% %.2f e/s %.2f Ms/s", accuracy * 100, epoch / es,
                            (epoch * SAMPLES / 1000000.0) / es));
            lastLog = System.currentTimeMillis();
            epoch = 0;
        }
    }

    private void train() {
        try {
            while (!destroyed) {
                while (paused)
                    Thread.sleep(100);
                network.train(network.batchIterator(batchSource), LEARNING_RATE);
                epoch++;
                evaluate();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void renderFlowers(Pixmap pixmap) {
        synchronized (evaluatedFlowers) {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    int flowerIndex = y * WIDTH + x;
                    int flower = flowers[flowerIndex];
                    int flowerColor;
                    if (batchSource.sampled.get(flowerIndex))
                        flowerColor = SAMPLED_FLOWER_COLORS[flower];
                    else if (evaluatedFlowers[flowerIndex] != flower)
                        flowerColor = DARKER_FLOWER_COLORS[flower];
                    else
                        flowerColor = FLOWER_COLORS[flower];
                    pixmap.drawPixel(x, y, flowerColor);
                }
            }
        }
    }

    private class FlowerBatchSource implements NeuralNetwork.BatchSource {

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
            for (int i = 0; i < size; i++) {
                int flowerIndex = samples[index + i];
                int x = flowerIndex % WIDTH;
                int y = flowerIndex / WIDTH;
                batch.inputs[0][i] = x / (float) WIDTH;
                batch.inputs[1][i] = y / (float) HEIGHT;
                int flower = flowers[flowerIndex];
                for (int output = 0; output < network.outputs(); output++) {
                    batch.expected[output][i] = output == flower ? 1 : 0;
                }
            }
        }
    }

    public static void main(String[] args) {
        GLDemoLauncher.launch(new FlowersDemo());
    }
}
