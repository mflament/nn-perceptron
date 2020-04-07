package org.yah.tests.perceptron.flowers;

import org.lwjgl.BufferUtils;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.opengl.GL;
import org.lwjgl.system.MemoryUtil;
import org.yah.games.opengl.Color4f;
import org.yah.games.opengl.shader.Program;
import org.yah.games.opengl.shader.Shader;
import org.yah.games.opengl.texture.*;
import org.yah.games.opengl.vao.ComponentType;
import org.yah.games.opengl.vao.VAO;
import org.yah.games.opengl.vbo.BufferAccess;
import org.yah.games.opengl.vbo.BufferAccess.Frequency;
import org.yah.games.opengl.vbo.BufferAccess.Nature;
import org.yah.games.opengl.vbo.VBO;
import org.yah.games.opengl.window.GLWindow;
import org.yah.tests.perceptron.*;
import org.yah.tests.perceptron.base.DefaultNetworkState;
import org.yah.tests.perceptron.base.DirectBufferOutputs;
import org.yah.tests.perceptron.jni.NativeNeuralNetwork;
import org.yah.tests.perceptron.matrix.MatrixNeuralNetwork;
import org.yah.tests.perceptron.matrix.array.CMArrayMatrix;
import org.yah.tests.perceptron.mt.MTNeuralNetwork;
import org.yah.tests.perceptron.opencl.CLNeuralNetwork;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.lwjgl.glfw.GLFW.glfwPollEvents;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL13.*;
import static org.lwjgl.opengl.GL20.glUniform1i;
import static org.lwjgl.opengl.GL20.glUniform4fv;

public class FlowersDemo {

    private static final int WIDTH = 800;
    private static final int HEIGHT = 600;
    private static final double NOISE_SCALE = 3f;

    private static final int FLOWERS = WIDTH * HEIGHT;

    private static final int[] LAYERS = {2, 16, 2};
    private static final int SAMPLES = (int) (FLOWERS * 0.005);
    private static final int EVAL_BATCH_SIZE = 0;
    private static final int TRAINING_BATCH_SIZE = 256;
    private static final double LEARNING_RATE = 0.5f;

    private static final float[] QUAD_VERTICES = {-1, -1, 0, 1, //
            -1, 1, 0, 0, //
            1, 1, 1, 0, //
            -1, -1, 0, 1, //
            1, 1, 1, 0, //
            1, -1, 1, 1};

    private static final Color4f[] FLOWER_COLORS = {new Color4f(0.9f, 0, 0, 1), new Color4f(0, 0.9f, 0, 1)};

    private List<Runnable> tasks = new ArrayList<>();

    private final ExecutorService executor;
    private final Object monitor = new Object();

    private final NeuralNetwork network;
    private final TrainingSamples allFlowers;
    private final TrainingSamples trainingFlowers;

    private final GLWindow window;
    private boolean paused = true;
    private boolean destroyed;

    private final NetworkOutputs networkOutputs;

    private final Program renderProgram;
    private final VBO vbo;
    private final VAO vao;
    private final Texture2D flowersTexture;
    private final Texture2D outputsTexture;
    private final Texture2D samplesTexture;

    public FlowersDemo(NeuralNetwork network) {
        this.network = network;
        executor = Executors.newSingleThreadExecutor();

        AllFlowersProvider flowersProvider = new AllFlowersProvider(WIDTH, HEIGHT, NOISE_SCALE);
        allFlowers = network.createTraining(flowersProvider, EVAL_BATCH_SIZE);
        networkOutputs = network.createOutpus(allFlowers.size());

        TrainingFlowersProvider trainingProvider = new TrainingFlowersProvider(flowersProvider, SAMPLES);
        trainingFlowers = network.createTraining(trainingProvider, TRAINING_BATCH_SIZE);

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
        ByteBuffer outputsBufer = BufferUtils.createByteBuffer(samples * Integer.BYTES);
        for (int i = 0; i < samples; i++) {
            outputsBufer.putInt(flowersProvider.outputIndex(i));
        }
        outputsBufer.flip();

        flowersTexture = Texture2D.builder(WIDTH, HEIGHT)
                .withInternalFormat(TextureInternalFormat.R32UI)
                .minFilter(TextureMinFilter.NEAREST)
                .magFilter(TextureMagFilter.NEAREST)
                .wrapS(TextureWrap.REPEAT)
                .withData(0, TextureFormat.RED_INTEGER, TextureDataType.UNSIGNED_INT, outputsBufer)
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
                        networkOutputs.reset();
                        start = System.nanoTime();
                        overallAccuracy = network.evaluate(allFlowers, networkOutputs);
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

        private void updateOutputs() throws InterruptedException {
            synchronized (monitor) {
                updateComplete.set(false);
                schedule(this::copyOutputs);
                while (!destroyed && !updateComplete.get())
                    monitor.wait();
            }
        }

        private IntBuffer outputsBuffer;

        private void copyOutputs() {
            outputsTexture.bind();
            networkOutputs.reset();
            if (networkOutputs instanceof DirectBufferOutputs) {
                outputsBuffer = ((DirectBufferOutputs) networkOutputs).buffer();
            } else {
                if (outputsBuffer == null)
                    outputsBuffer = BufferUtils.createIntBuffer(allFlowers.size());
                networkOutputs.copy(outputsBuffer);
            }
            outputsTexture.updateData(TextureFormat.RED_INTEGER, TextureDataType.UNSIGNED_INT, MemoryUtil.memByteBuffer(outputsBuffer));
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
        NeuralNetworkState state = new DefaultNetworkState(RandomUtils.newRandomSource(), LAYERS);
        switch (arg) {
            case "native":
                network = new NativeNeuralNetwork(state);
                break;
            case "cl":
                network = new CLNeuralNetwork(state);
                break;
            case "mt":
                network = new MTNeuralNetwork(state);
                break;
            default:
                network = new MatrixNeuralNetwork<>(CMArrayMatrix::new, state);
        }
        return network;
    }
}
