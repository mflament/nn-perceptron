package org.yah.tests.perceptron.flowers;

import java.nio.IntBuffer;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.lwjgl.BufferUtils;
import org.yah.tests.perceptron.AbstractGLDemo;
import org.yah.tests.perceptron.Batch;
import org.yah.tests.perceptron.BatchSource;
import org.yah.tests.perceptron.BatchSource.TrainingSet;
import org.yah.tests.perceptron.GLDemoLauncher;
import org.yah.tests.perceptron.Matrix;
import org.yah.tests.perceptron.MatrixNeuralNetwork;
import org.yah.tests.perceptron.MatrixNeuralNetwork.MatrixFactory;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.array.ArrayMatrix;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.Pixmap.Format;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.TextureData;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;

public class FlowersDemo<M extends Matrix<M>> extends AbstractGLDemo {

    private static final int WIDTH = 800;
    private static final int HEIGHT = 600;
    private static final double NOISE_SCALE = 3f;

    private static final int FLOWERS = WIDTH * HEIGHT;

    private static final int[] LAYERS = { 2, 16, 2 };
    private static final int SAMPLES = (int) (FLOWERS * 0.005);
    private static final int BATCH_SIZE = 64;
    private static final double LEARNING_RATE = 0.5f;

    private ExecutorService executor;

    private int[] FLOWER_COLORS = { 0xff0000ee, 0xff00ee00 };
    private int[] DARKER_FLOWER_COLORS = { 0xff0000aa, 0xff00aa00 };
    private int[] SAMPLED_FLOWER_COLORS = { 0xff000000, 0xff000000 };

    private int[] flowers = new int[FLOWERS];
    private int[] randomFlowers = new int[FLOWERS];
    private int[] outputs = new int[FLOWERS];

    private NeuralNetwork<M> network;
    private boolean paused = true;
    private boolean destroyed;
    private long lastLog;
    private long epoch;

    private Batch<M> allFlowersBatch;
    private TrainingSet<M> trainingSet;

    private Texture texture;
    private FlowersTextureData textureData;

    private final MatrixFactory<M> matrixFactory;

    public FlowersDemo(MatrixFactory<M> matrixFactory) {
        this.matrixFactory = matrixFactory;
    }

    @Override
    public void create() {
        super.create();
        executor = Executors.newSingleThreadExecutor();
        textureData = new FlowersTextureData();
        texture = new Texture(textureData);

        MatrixNeuralNetwork<M> network = new MatrixNeuralNetwork<>(matrixFactory, LAYERS);
        this.network = network;

        BatchSource<M> batchSource = network.createBatchSource();
        createFlowers();
        createFlowersBatch(batchSource);
        createTrainingBatches(batchSource);

        System.out.println("Flowers demo config:");
        System.out.println("  Network: " + network);
        System.out.println("  flowers: " + allFlowersBatch.size());
        System.out.println(String.format("  samples: %d (%d x %d), %s%%", trainingSet.samples(),
                trainingSet.batchCount(), trainingSet.batchSize(),
                trainingSet.samples() / (double) allFlowersBatch.size() * 100.0));
        executor.submit(this::trainingLoop);
    }

    private void createFlowersBatch(BatchSource<M> batchSource) {
        double[][] inputs = new double[FLOWERS][2];
        int flowerIndex = 0;
        for (int y = 0; y < HEIGHT; y++) {
            double dy = y / (double) HEIGHT;
            for (int x = 0; x < WIDTH; x++) {
                double[] flower = inputs[flowerIndex];
                flower[0] = x / (double) WIDTH;
                flower[1] = dy;
                flowerIndex++;
            }
        }
        allFlowersBatch = batchSource.createBatch(inputs, flowers, true);
    }

    private void createTrainingBatches(BatchSource<M> batchSource) {
        int flowerIndex;
        int samples = Math.min(FLOWERS, SAMPLES);
        double[][] inputs = new double[2][samples];
        int[] expecteds = new int[samples];
        randomizeFlowers();
        for (int sample = 0; sample < samples; sample++) {
            flowerIndex = randomFlowers[sample];
            int x = flowerIndex % WIDTH;
            int y = flowerIndex / WIDTH;
            inputs[0][sample] = x / (double) WIDTH;
            inputs[1][sample] = y / (double) HEIGHT;
            expecteds[sample] = flowers[flowerIndex];
        }
        trainingSet = batchSource.createBatches(inputs, expecteds, BATCH_SIZE);
    }

    private void randomizeFlowers() {
        Random random = MatrixNeuralNetwork.RANDOM;
        for (int i = 0; i < randomFlowers.length; i++)
            randomFlowers[i] = i;
        for (int i = 0; i < randomFlowers.length; i++)
            swap(randomFlowers, i, random.nextInt(randomFlowers.length));
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

    @Override
    protected void render(SpriteBatch spriteBatch) {
        spriteBatch.draw(texture, 0, 0, getWidth(), getHeight());
    }

    private void createFlowers() {
        long seed = MatrixNeuralNetwork.seed();
        OpenSimplexNoise noise = new OpenSimplexNoise(seed < 0 ? System.currentTimeMillis() : seed);
        int flowerIndex = 0;
        for (int y = 0; y < HEIGHT; y++) {
            double dy = y / (double) HEIGHT;
            for (int x = 0; x < WIDTH; x++) {
                double dx = x / (double) WIDTH;
                double n = noise.eval(dx * NOISE_SCALE, dy * NOISE_SCALE);
                n = (n + 1) / 2.0;
                flowers[flowerIndex++] = (int) (n + 0.5);
            }
        }
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

    private void trainingLoop() {
        try {
            double overallAccuracy = network.evaluate(allFlowersBatch, outputs);
            schedule(this::renderFlowers);
            System.out.println(String.format("%f%%", overallAccuracy * 100));

            lastLog = System.currentTimeMillis();
            while (!destroyed) {
                while (paused && !destroyed) {
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {}
                }
                if (destroyed)
                    return;

                network.train(trainingSet.iterator(), LEARNING_RATE);
                epoch++;

                long elapsed = System.currentTimeMillis() - lastLog;
                if (elapsed > 1000) {
                    synchronized (outputs) {
                        overallAccuracy = network.evaluate(allFlowersBatch, outputs);
                    }
                    schedule(this::renderFlowers);
                    long samples = epoch * SAMPLES;
                    double accuracy = network.evaluate(trainingSet.iterator());
                    System.out.println(String.format(
                            "training accuracy: %.2f%%; overall accuracy: %.2f%%; e/s: %.2f; s/ms: %.2f",
                            accuracy * 100, overallAccuracy * 100,
                            epoch / (elapsed / 1000.0),
                            samples / (double) elapsed));
                    lastLog = System.currentTimeMillis();
                    epoch = 0;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void renderFlowers() {
        synchronized (outputs) {
            for (int flowerIndex = 0; flowerIndex < FLOWERS; flowerIndex++) {
                int flower = flowers[flowerIndex];
                int flowerColor;
                if (outputs[flowerIndex] != flower)
                    flowerColor = DARKER_FLOWER_COLORS[flower];
                else
                    flowerColor = FLOWER_COLORS[flower];
                textureData.putPixel(flowerColor);
            }
            textureData.flip();
        }
        for (int i = 0; i < SAMPLES; i++) {
            int flowerIndex = randomFlowers[i];
            textureData.setPixel(flowerIndex, SAMPLED_FLOWER_COLORS[flowers[flowerIndex]]);
        }
        texture.load(textureData);
    }

    private static void swap(int[] array, int a, int b) {
        int buff = array[a];
        array[a] = array[b];
        array[b] = buff;
    }

    private static class FlowersTextureData implements TextureData {

        private static final Format FORMAT = Format.RGBA8888;
        private static final int GL_FORMAT = Format.toGlFormat(FORMAT);
        private static final int GL_TYPE = Format.toGlType(FORMAT);

        private final IntBuffer buffer;

        public FlowersTextureData() {
            buffer = BufferUtils.createIntBuffer(WIDTH * HEIGHT);
        }

        public void flip() {
            buffer.flip();
        }

        public void putPixel(int color) {
            buffer.put(color);
        }

        public void setPixel(int index, int color) {
            buffer.put(index, color);
        }

        @Override
        public TextureDataType getType() { return TextureDataType.Custom; }

        @Override
        public boolean isPrepared() { return true; }

        @Override
        public void prepare() {}

        @Override
        public Pixmap consumePixmap() {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean disposePixmap() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void consumeCustomData(int target) {
            Gdx.gl.glTexImage2D(target, 0, GL_FORMAT, WIDTH, HEIGHT, 0, GL_FORMAT, GL_TYPE, buffer);
        }

        @Override
        public int getWidth() { return WIDTH; }

        @Override
        public int getHeight() { return HEIGHT; }

        @Override
        public Format getFormat() { return FORMAT; }

        @Override
        public boolean useMipMaps() {
            return false;
        }

        @Override
        public boolean isManaged() { return false; }
    }

    public static void main(String[] args) {
        GLDemoLauncher.launch(new FlowersDemo<>(ArrayMatrix::new));
    }
}
