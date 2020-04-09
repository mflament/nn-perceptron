package org.yah.tests.perceptron.mt;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * @author Yah
 *
 */
public class ChunkExecutor implements AutoCloseable {

    private static final int MIN_CHUNK_SIZE = 5000;
    
    private final ExecutorService executor;

    @FunctionalInterface
    public interface ChunkHandler {
        
        default void start(int chunksCount) {}

        void handle(int chunkIndex, int offset, int size);

        default void complete(int chunksCount) {}
    }

    private static class Chunk implements Runnable {
        private final int index;
        private int offset, size;
        private ChunkHandler handler;

        public Chunk(int index) {
            this.index = index;
        }

        public void prepare(int offset, int size, ChunkHandler handler) {
            this.offset = offset;
            this.size = size;
            this.handler = handler;
        }

        @Override
        public void run() {
            handler.handle(index, offset, size);
        }

        public void clear() {
            handler = null;
            offset = size = 0;
        }
    }

    private final Chunk[] chunks;

    public ChunkExecutor(int concurrency) {
        this.executor = Executors.newFixedThreadPool(concurrency - 1);
        chunks = new Chunk[concurrency];
        for (int i = 0; i < chunks.length; i++) {
            chunks[i] = new Chunk(i);
        }
    }

    @Override
    public void close() {
        executor.shutdown();
    }

    public void distribute(int count, ChunkHandler handler) {
        int concurrency = chunks.length;
        int chunkSize = count / concurrency;
        
        Chunk chunk;
        if (chunkSize < MIN_CHUNK_SIZE) {
            handler.start(1);
            chunk = chunks[0];
            chunk.prepare(0, count, handler);
            chunk.run();
            handler.complete(1);
            return;
        }
        
        int remaining = count % concurrency;
        int offset = 0;
        @SuppressWarnings("unchecked")
        Future<Chunk>[] futures = new Future[concurrency - 1];
        int chunkIndex;
        handler.start(concurrency);
        
        for (chunkIndex = 0; chunkIndex < concurrency - 1; chunkIndex++) {
            int size = chunkSize;
            if (remaining > 0) {
                size++;
                remaining--;
            }

            chunk = chunks[chunkIndex];
            chunk.prepare(offset, size, handler);
            futures[chunkIndex] = executor.submit(chunk, chunk);
            offset += size;
        }

        chunk = chunks[chunkIndex];
        chunk.prepare(offset, chunkSize, handler);
        chunk.run();

        remaining = futures.length;
        while (remaining > 0) {
            for (int i = 0; i < futures.length; i++) {
                if (futures[i] != null && futures[i].isDone()) {
                    chunk = safeGet(futures[i]);
                    chunk.clear();
                    futures[i] = null;
                    remaining--;
                }
            }
        }
        handler.complete(concurrency);
    }

    private Chunk safeGet(Future<Chunk> future) {
        try {
            return future.get();
        } catch (InterruptedException e) {
            throw new IllegalStateException("Should never happen");
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        }
    }
}
