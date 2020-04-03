package org.yah.tests.perceptron.flowers;

import static org.lwjgl.glfw.GLFW.glfwPollEvents;
import static org.lwjgl.opengl.GL11.GL_TRIANGLES;
import static org.lwjgl.opengl.GL11.glDrawArrays;
import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL13.GL_TEXTURE0;
import static org.lwjgl.opengl.GL13.GL_TEXTURE1;
import static org.lwjgl.opengl.GL13.GL_TEXTURE2;
import static org.lwjgl.opengl.GL13.glActiveTexture;
import static org.lwjgl.opengl.GL20.glUniform1i;
import static org.lwjgl.opengl.GL20.glUniform4fv;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import org.lwjgl.BufferUtils;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.opengl.GL;
import org.lwjgl.system.MemoryUtil;
import org.yah.games.opengl.Color4f;
import org.yah.games.opengl.shader.Program;
import org.yah.games.opengl.shader.Shader;
import org.yah.games.opengl.texture.Texture2D;
import org.yah.games.opengl.texture.TextureDataType;
import org.yah.games.opengl.texture.TextureFormat;
import org.yah.games.opengl.texture.TextureInternalFormat;
import org.yah.games.opengl.texture.TextureMagFilter;
import org.yah.games.opengl.texture.TextureMinFilter;
import org.yah.games.opengl.texture.TextureWrap;
import org.yah.games.opengl.vao.ComponentType;
import org.yah.games.opengl.vao.VAO;
import org.yah.games.opengl.vbo.BufferAccess;
import org.yah.games.opengl.vbo.BufferAccess.Frequency;
import org.yah.games.opengl.vbo.BufferAccess.Nature;
import org.yah.games.opengl.vbo.VBO;
import org.yah.games.opengl.window.GLWindow;
import org.yah.tests.perceptron.NeuralNetwork;
import org.yah.tests.perceptron.SamplesSource;
import org.yah.tests.perceptron.TrainingSamples;
import org.yah.tests.perceptron.jni.NativeNeuralNetwork;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;
import org.yah.tests.perceptron.mt.MTNeuralNetwork;
import org.yah.tests.perceptron.opencl.CLNeuralNetwork;

public class FlowersDemo {

    private static final int WIDTH = 800;
    private static final int HEIGHT = 600;
    private static final double NOISE_SCALE = 3f;

    private static final int FLOWERS = WIDTH * HEIGHT;

    private static final int[] LAYERS = { 2, 16, 2 };
    private static final int SAMPLES = (int) (FLOWERS * 0.005);
    private static final int EVAL_BATCH_SIZE = 0;
    private static final int TRAINING_BATCH_SIZE = 256;
    private static final double LEARNING_RATE = 0.5f;

    private static final float[] QUAD_VERTICES = { -1, -1, 0, 1, //
            -1, 1, 0, 0, //
            1, 1, 1, 0, //
            -1, -1, 0, 1, //
            1, 1, 1, 0, //
            1, -1, 1, 1 };

    private static final Color4f[] FLOWER_COLORS = { new Color4f(0.9f, 0, 0, 1), new Color4f(0, 0.9f, 0, 1) };

    private List<Runnable> tasks = new ArrayList<>();

    private final ExecutorService executor;
    private final Object monitor = new Object();

    private final NeuralNetwork network;
    private final TrainingSamples allFlowers;
    private final TrainingSamples trainingFlowers;

    private final GLWindow window;
    private boolean paused = true;
    private boolean destroyed;

    private final IntBuffer outputsBuffer;

    private final Program renderProgram;
    private final VBO vbo;
    private final VAO vao;
    private final Texture2D flowersTexture;
    private final Texture2D outputsTexture;
    private final Texture2D samplesTexture;

    private OutputStream snapshotOutputStream;
    private InputStream snapshotInputStream;

    public FlowersDemo(NeuralNetwork network) {
        this.network = network;
        executor = Executors.newSingleThreadExecutor();

        SamplesSource samplesSource = network.createSampleSource();
        AllFlowersProvider flowersProvider = new AllFlowersProvider(WIDTH, HEIGHT, NOISE_SCALE);
        allFlowers = samplesSource.createTraining(flowersProvider, EVAL_BATCH_SIZE);
        TrainingFlowersProvider trainingProvider = new TrainingFlowersProvider(flowersProvider, SAMPLES);
        trainingFlowers = samplesSource.createTraining(trainingProvider, TRAINING_BATCH_SIZE);

        printInfo();

        window = GLWindow.builder()
                .withDebug()
                .withTitle("Flowers Demo - " + network.getClass().getSimpleName())
                .withSize(WIDTH, HEIGHT)
                .withKeyPressHandler(this::keyPressed)
                .build();
        window.centerOnScreen();
        GL.createCapabilities();

        renderProgram = Program.builder()
                .with(Shader.vertexShader("glsl/flowers.vs.glsl"))
                .with(Shader.fragmentShader("glsl/flowers.fs.glsl"))
                .build();

        vbo = VBO.builder().withData(QUAD_VERTICES, BufferAccess.from(Frequency.STATIC, Nature.DRAW)).build();
        vao = VAO.builder(renderProgram, vbo)
                .withAttribute("position", 2, ComponentType.FLOAT, false, ComponentType.FLOAT.sizeOf(4), 0)
                .withAttribute("aInputs",
                        2,
                        ComponentType.FLOAT,
                        false,
                        ComponentType.FLOAT.sizeOf(4),
                        ComponentType.FLOAT.sizeOf(2))
                .build();
        vao.bind();

        renderProgram.use();
        glUniform4fv(renderProgram.findUniformLocation("flowerColors"), colorBuffer());
        glUniform1i(renderProgram.findUniformLocation("expectedFlowers"), 0);
        glUniform1i(renderProgram.findUniformLocation("actualFlowers"), 1);
        glUniform1i(renderProgram.findUniformLocation("sampledFlowers"), 2);

        int samples = flowersProvider.samples();
        int[] outputs = new int[samples];
        for (int i = 0; i < samples; i++) {
            outputs[i] = flowersProvider.outputIndex(i);
        }

        outputsBuffer = BufferUtils.createIntBuffer(samples);
        outputsBuffer.put(outputs).flip();

        flowersTexture = Texture2D.builder(WIDTH, HEIGHT)
                .withInternalFormat(TextureInternalFormat.R32UI)
                .minFilter(TextureMinFilter.NEAREST)
                .magFilter(TextureMagFilter.NEAREST)
                .wrapS(TextureWrap.REPEAT)
                .withData(0, TextureFormat.RED_INTEGER, TextureDataType.UNSIGNED_INT,
                        MemoryUtil.memByteBuffer(outputsBuffer))
                .build();

        outputsTexture = Texture2D.builder(WIDTH, HEIGHT)
                .withInternalFormat(TextureInternalFormat.R32UI)
                .minFilter(TextureMinFilter.NEAREST)
                .magFilter(TextureMagFilter.NEAREST)
                .wrapS(TextureWrap.REPEAT)
                .withData(0, TextureFormat.RED_INTEGER, TextureDataType.UNSIGNED_INT, null)
                .build();

        ByteBuffer buffer = BufferUtils.createByteBuffer(samples);
        for (int i = 0; i < trainingProvider.samples(); i++) {
            double dx = trainingProvider.input(i, 0);
            double dy = trainingProvider.input(i, 1);
            int index = (int) ((dy * HEIGHT) * WIDTH + (dx * WIDTH));
            buffer.put(index, (byte) 1);
        }

        buffer.flip();
        samplesTexture = Texture2D.builder(WIDTH, HEIGHT)
                .withInternalFormat(TextureInternalFormat.R8UI)
                .minFilter(TextureMinFilter.NEAREST)
                .magFilter(TextureMagFilter.NEAREST)
                .wrapS(TextureWrap.REPEAT)
                .withData(0, TextureFormat.RED_INTEGER, TextureDataType.UNSIGNED_BYTE, buffer)
                .build();
    }

    private static FloatBuffer colorBuffer() {
        FloatBuffer fb = BufferUtils.createFloatBuffer(FLOWER_COLORS.length * 4);
        for (Color4f flowerColor : FLOWER_COLORS) {
            fb.put(flowerColor.r);
            fb.put(flowerColor.g);
            fb.put(flowerColor.b);
            fb.put(flowerColor.a);
        }
        return fb.flip();
    }

    private void printInfo() {
        System.out.println("Flowers demo config:");
        System.out.println("  Network: " + network + " (" + network.getClass().getSimpleName() + ")");
        System.out.println("  flowers: " + allFlowers.size());
        System.out.println(String.format("  samples: %d (%d x %d), %s%%", trainingFlowers.size(),
                trainingFlowers.batchCount(), trainingFlowers.batchSize(),
                trainingFlowers.size() / (double) allFlowers.size() * 100.0));
    }

    public void start() {
        window.show();
        glViewport(0, 0, WIDTH, HEIGHT);

        List<Runnable> nextTasks = new ArrayList<>();
        executor.submit(new TrainingLoop());
        while (!window.isCloseRequested()) {
            glfwPollEvents();
            synchronized (this) {
                List<Runnable> swap = tasks;
                tasks = nextTasks;
                nextTasks = swap;
            }
            for (Runnable runnable : nextTasks) {
                runnable.run();
            }
            nextTasks.clear();

            render();
        }
        dispose();
    }

    private void render() {
        renderProgram.use();
        vao.bind();

        glActiveTexture(GL_TEXTURE0);
        flowersTexture.bind();

        glActiveTexture(GL_TEXTURE1);
        outputsTexture.bind();

        glActiveTexture(GL_TEXTURE2);
        samplesTexture.bind();

        glDrawArrays(GL_TRIANGLES, 0, 6);
        window.swapBuffers();
    }

    private synchronized void schedule(Runnable runnable) {
        tasks.add(runnable);
    }

    private void keyPressed(int key, int scancode, int mods) {
        switch (key) {
        case GLFW.GLFW_KEY_SPACE:
            notifyTrainingLoop(() -> paused = !paused);
            break;
        case GLFW.GLFW_KEY_ESCAPE:
            window.requestClose();
            break;
        }
    }

    private void notifyTrainingLoop(Runnable action) {
        synchronized (monitor) {
            action.run();
            monitor.notify();
        }
    }

    private void dispose() {
        notifyTrainingLoop(() -> destroyed = true);
        executor.shutdown();

        vao.delete();
        vbo.delete();
        renderProgram.delete();
        flowersTexture.delete();
        GLFW.glfwTerminate();

        try {
            executor.awaitTermination(1, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        closeQuietly(allFlowers);
        closeQuietly(trainingFlowers);

        closeQuietly(snapshotOutputStream);
        closeQuietly(snapshotInputStream);
    }

    private class TrainingLoop implements Runnable {
        private double overallAccuracy;
        private final AtomicBoolean updateComplete = new AtomicBoolean();

        @Override
        public void run() {
            try {
                updateOutputs();

                double trainingAccuracy = network.evaluate(trainingFlowers);
                System.out.println(String.format(
                        "training accuracy: %.2f%%; overall accuracy: %.2f%%",
                        trainingAccuracy * 100, overallAccuracy * 100));

                long lastLog = System.currentTimeMillis();
                int lastEepochs = 0;
                long epochs = 0;
                long trainingTime = 0, evaluationTime, trainingEvaluationTime, start;
                while (true) {
                    synchronized (monitor) {
                        while (paused && !destroyed) {
                            monitor.wait();
                            lastLog = System.currentTimeMillis();
                        }
                        if (destroyed)
                            break;
                    }

                    start = System.nanoTime();
                    network.train(trainingFlowers, LEARNING_RATE);
                    //snapshot(epochs);
                    //checkSnapshot(epochs);

                    trainingTime += System.nanoTime() - start;
                    lastEepochs++;
                    epochs++;

                    long elapsed = System.currentTimeMillis() - lastLog;
                    if (elapsed > 1000) {
                        outputsBuffer.position(0);

                        start = System.nanoTime();
                        overallAccuracy = network.evaluate(allFlowers, outputsBuffer);
                        evaluationTime = System.nanoTime() - start;

                        start = System.nanoTime();
                        trainingAccuracy = network.evaluate(trainingFlowers);
                        trainingEvaluationTime = System.nanoTime() - start;

                        long samples = lastEepochs * SAMPLES;
                        double eps = lastEepochs / (elapsed / 1000.0);
                        double sms = samples / (double) elapsed;
                        long avgtt = TimeUnit.NANOSECONDS.toMillis(trainingTime) / lastEepochs;
                        long et = TimeUnit.NANOSECONDS.toMillis(evaluationTime);
                        long tet = TimeUnit.NANOSECONDS.toMillis(trainingEvaluationTime);
                        System.out.println(String.format(
                                "%-8d: training accuracy: %.2f%%; overall accuracy: %.2f%%; %.2f e/s; %.2f s/ms; training: %dms; eval: %dms; training eval:%dms",
                                epochs,
                                trainingAccuracy * 100, overallAccuracy * 100,
                                eps, sms,
                                avgtt, et, tet));

                        updateOutputs();
                        lastLog = System.currentTimeMillis();
                        lastEepochs = 0;
                        trainingTime = 0;
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        /** @noinspection unused*/
        private void checkSnapshot(long epoch) {
            // byte[] expected;
            ByteBuffer buffer;
            try {
                if (snapshotInputStream == null) { snapshotInputStream = new FileInputStream("snapshot.bin"); }
                buffer = ByteBuffer.wrap(snapshotInputStream.readAllBytes());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }

            DoubleBuffer actualBuffer = network.snapshot().asDoubleBuffer();

            DoubleBuffer doubleBuffer = buffer.asDoubleBuffer();
            for (int l = 0; l < network.layers(); l++) {
                int neurons = network.neurons(l);
                int features = network.features(l);
                for (int n = 0; n < neurons; n++) {
                    for (int f = 0; f < features; f++) {
                        double expected = doubleBuffer.get();
                        double actual = actualBuffer.get();
                        if (expected != actual)
                            throw new IllegalStateException(
                                    String.format("mimatch at layer %d,weight(%d, %d)", l, n, f));
                    }
                }
                for (int n = 0; n < neurons; n++) {
                    double expected = doubleBuffer.get();
                    double actual = actualBuffer.get();
                    if (expected != actual)
                        throw new IllegalStateException(
                                String.format("mimatch at layer %d, bias(%d)", l, n));
                }

            }
//            if (expected.length < 16) {
//                System.out.println("End of snapshot");
//                return;
//            }
//
//            byte[] actual = networkSnapshot();
//            for (int i = 0; i < actual.length; i++) {
//                if (actual[i] != expected[i])
//                    throw new IllegalStateException("snapshot mismatch " + epoch);
//            }
        }

        boolean snapshoted = false;

        private void snapshot() {
            if (snapshoted)
                return;
            ByteBuffer snapshot = network.snapshot();
            // byte[] digest = networkSnapshot();
            try {
                if (snapshotOutputStream == null) { snapshotOutputStream = new FileOutputStream("snapshot.bin"); }
                snapshotOutputStream.write(snapshot.array());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            snapshoted = true;
        }

        /** @noinspection unused*/
        private byte[] networkSnapshot() {
            ByteBuffer snapshot = network.snapshot();
            MessageDigest md;
            try {
                md = MessageDigest.getInstance("MD5");
            } catch (NoSuchAlgorithmException e) {
                throw new RuntimeException(e);
            }
            md.update(snapshot);
            return md.digest();
        }

        private void updateOutputs() throws InterruptedException {
            outputsBuffer.position(0);
            overallAccuracy = network.evaluate(allFlowers, outputsBuffer);

            synchronized (monitor) {
                updateComplete.set(false);
                schedule(this::copyOutputs);
                while (!destroyed && !updateComplete.get())
                    monitor.wait();
            }
        }

        private void copyOutputs() {
            outputsTexture.bind();
            outputsBuffer.position(0);
            outputsTexture.updateData(TextureFormat.RED_INTEGER, TextureDataType.UNSIGNED_INT,
                    MemoryUtil.memByteBuffer(outputsBuffer));
            notifyTrainingLoop(() -> updateComplete.set(true));
        }
    }

    public static void main(String[] args) throws IOException {
        NeuralNetwork network = createNetwork(args.length > 0 ? args[0] : "matrix");
        try {
            new FlowersDemo(network).start();
        } finally {
            closeQuietly(network);
        }
    }

    private static void closeQuietly(Object o) {
        if (o instanceof AutoCloseable) {
            try {
                ((AutoCloseable) o).close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private static NeuralNetwork createNetwork(String arg) throws IOException {
        NeuralNetwork network;
        switch (arg) {
        case "native":
            network = new NativeNeuralNetwork(LAYERS);
            break;
        case "cl":
            network = new CLNeuralNetwork(LAYERS);
            break;
        case "mt":
            network = new MTNeuralNetwork(LAYERS);
            break;
        default:
            network = new MatrixNeuralNetwork<>(CMArrayMatrix::new, LAYERS);
        }
        return network;
    }
}
