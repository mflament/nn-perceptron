package org.yah.tests.perceptron.flowers;

import java.nio.IntBuffer;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.lwjgl.BufferUtils;
import org.yah.tests.perceptron.AbstractGLDemo;
import org.yah.tests.perceptron.GLDemoLauncher;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.SamplesProviders.TrainingSamplesProvider;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.Pixmap.Format;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.TextureData;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;

public class FlowersDemo extends AbstractGLDemo implements TrainingSamplesProvider {

    private static final int WIDTH = 800;
    private static final int HEIGHT = 600;
    private static final double NOISE_SCALE = 3f;

    private static final int FLOWERS = WIDTH * HEIGHT;

    private static final int[] LAYERS = { 2, 16, 2 };
    private static final int SAMPLES = (int) (FLOWERS * 0.005);
    private static final int EVAL_BATCH_SIZE = 0;
    private static final int BATCH_SIZE = 64;
    private static final double LEARNING_RATE = 0.5f;

    private ExecutorService executor;

    private int[] FLOWER_COLORS = { 0xff0000ee, 0xff00ee00 };
    private int[] DARKER_FLOWER_COLORS = { 0xff0000aa, 0xff00aa00 };
    private int[] SAMPLED_FLOWER_COLORS = { 0xff000000, 0xffffffff };

    private int[] flowers = new int[FLOWERS];
    private int[] randomFlowers = new int[FLOWERS];
    private int[] outputs = new int[FLOWERS];

    private final NeuralNetwork network;
    private boolean paused = true;
    private boolean destroyed;
    private long lastLog;
    private long epoch;

    private TrainingSamples allFlowers;
    private TrainingSamples trainingFlowers;

    private Texture texture;
    private FlowersTextureData textureData;
    
    private CountDownLatch exitLatch;

    public FlowersDemo(NeuralNetwork network) {
        this.network = network;
    }

    @Override
    public void create() {
        super.create();
        executor = Executors.newSingleThreadExecutor();
        textureData = new FlowersTextureData();
        texture = new Texture(textureData);

        SamplesSource samplesSource = network.createSampleSource();
        createFlowers();
        createFlowersBatch(samplesSource);
        createTrainingBatches(samplesSource);

        System.out.println("Flowers demo config:");
        System.out.println("  Network: " + network);
        System.out.println("  flowers: " + allFlowers.size());
        System.out.println(String.format("  samples: %d (%d x %d), %s%%", trainingFlowers.size(),
                trainingFlowers.batchCount(), trainingFlowers.batchSize(),
                trainingFlowers.size() / (double) allFlowers.size() * 100.0));
        exitLatch = new CountDownLatch(1);
        executor.submit(this::trainingLoop);
    }

    private void createFlowersBatch(SamplesSource samplesSource) {
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
        allFlowers = samplesSource.createTraining(this, EVAL_BATCH_SIZE);
    }

    @Override
    public int samples() {
        return FLOWERS;
    }

    @Override
    public int features() {
        return 2;
    }

    @Override
    public double input(int sample, int feature) {
        return feature == 0 ? (sample % WIDTH) / (double) WIDTH
                : (sample / WIDTH) / (double) HEIGHT;
    }

    @Override
    public int outputIndex(int sample) {
        return flowers[sample];
    }

    private void createTrainingBatches(SamplesSource samplesSource) {
        int flowerIndex;
        int samples = Math.min(FLOWERS, SAMPLES);
        double[][] inputs = new double[samples][2];
        int[] expecteds = new int[samples];
        randomizeFlowers();
        for (int sample = 0; sample < samples; sample++) {
            flowerIndex = randomFlowers[sample];
            int x = flowerIndex % WIDTH;
            int y = flowerIndex / WIDTH;
            inputs[sample][0] = x / (double) WIDTH;
            inputs[sample][1] = y / (double) HEIGHT;
            expecteds[sample] = flowers[flowerIndex];
        }
        TrainingSamplesProvider provider = new TrainingSamplesProvider() {
            @Override
            public int samples() {
                return samples;
            }

            @Override
            public double input(int sample, int feature) {
                int flowerIndex = randomFlowers[sample];
                return FlowersDemo.this.input(flowerIndex, feature);
            }

            @Override
            public int features() {
                return 2;
            }

            @Override
            public int outputIndex(int sample) {
                int flowerIndex = randomFlowers[sample];
                return flowers[flowerIndex];
            }
        };
        trainingFlowers = samplesSource.createTraining(provider, BATCH_SIZE);
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
        try {
            exitLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        executor.shutdownNow();
        closeQuietly(allFlowers);
        closeQuietly(trainingFlowers);
        closeQuietly(network);
        super.dispose();
    }

    private void closeQuietly(Object o) {
        if (o instanceof AutoCloseable) {
            try {
                ((AutoCloseable) o).close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
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
            schedule(this::renderFlowers);
            
            double overallAccuracy = network.evaluate(allFlowers, outputs);
            double trainingAccuracy = network.evaluate(trainingFlowers, null);
            System.out.println(String.format(
                    "training accuracy: %.2f%%; overall accuracy: %.2f%%", trainingAccuracy * 100, overallAccuracy * 100));
            
            lastLog = System.currentTimeMillis();
            while (!destroyed) {
                while (paused && !destroyed) {
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {}
                }
                if (destroyed)
                    return;

                network.train(trainingFlowers, LEARNING_RATE);
                epoch++;

                long elapsed = System.currentTimeMillis() - lastLog;
                if (elapsed > 1000) {
                    synchronized (outputs) {
                        overallAccuracy = network.evaluate(allFlowers, outputs);
                    }
                    schedule(this::renderFlowers);
                    long samples = epoch * SAMPLES;
                    trainingAccuracy = network.evaluate(trainingFlowers, null);
                    System.out.println(String.format(
                            "training accuracy: %.2f%%; overall accuracy: %.2f%%; e/s: %.2f; Ms/ms: %.2f",
                            trainingAccuracy * 100, overallAccuracy * 100,
                            epoch / (elapsed / 1000.0),
                            samples / (double) elapsed));
                    lastLog = System.currentTimeMillis();
                    epoch = 0;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            exitLatch.countDown();
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
            textureData.setPixel(flowerIndex, SAMPLED_FLOWER_COLORS[flowers[flowerIndex] == outputs[flowerIndex] ? 1 : 0]);
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
        MatrixNeuralNetwork<CMArrayMatrix> network = new MatrixNeuralNetwork<>(CMArrayMatrix::new,
                LAYERS);
//        NativeNeuralNetwork network = new NativeNeuralNetwork(LAYERS);
        GLDemoLauncher.launch(new FlowersDemo(network));
    }
}
