/**
 * 
 */
package org.yah.tests.perceptron;

import java.nio.IntBuffer;
import java.util.LinkedList;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.InputAdapter;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Matrix4;

/**
 * @author Yah
 *
 */
public abstract class AbstractGLDemo extends InputAdapter implements GLDemo {

    protected SpriteBatch spriteBatch;
    private final LinkedList<Runnable> scheduledTasks = new LinkedList<Runnable>();
    private int width, height;

    @Override
    public void create() {
        spriteBatch = new SpriteBatch();
        width = Gdx.graphics.getWidth();
        height = Gdx.graphics.getHeight();
        Gdx.input.setInputProcessor(this);

    }

    protected final void schedule(Runnable task) {
        synchronized (scheduledTasks) {
            scheduledTasks.addLast(task);
        }
    }

    @Override
    public String getTitle() { return getClass().getSimpleName(); }
    
    public int getWidth() { return width; }
    
    public int getHeight() { return height; }

    @Override
    public void dispose() {
        spriteBatch.dispose();
    }

    @Override
    public void render() {
        int pendings;
        synchronized (scheduledTasks) {
            pendings = scheduledTasks.size();
        }

        for (int i = 0; i < pendings; i++) {
            Runnable task;
            synchronized (scheduledTasks) {
                task = scheduledTasks.pop();
            }
            task.run();
        }

        spriteBatch.begin();
        render(spriteBatch);
        spriteBatch.end();
    }

    protected abstract void render(SpriteBatch spriteBatch);

    @Override
    public void resize(int width, int height) {
        this.width = width;
        this.height = height;
        spriteBatch.setProjectionMatrix(new Matrix4().setToOrtho2D(0, 0,
                Gdx.graphics.getWidth(), Gdx.graphics.getHeight()));
    }

    @Override
    public boolean keyDown(int keycode) {
        if (keycode == Input.Keys.ESCAPE) {
            Gdx.app.exit();
            return true;
        }
        return super.keyDown(keycode);
    }

    @Override
    public void configure(Lwjgl3ApplicationConfiguration config) {}

    protected static final IntBuffer toIntBuffer(Pixmap pixmap) {
        return pixmap.getPixels().asIntBuffer();
    }
}
