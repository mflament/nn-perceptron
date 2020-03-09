/**
 * 
 */
package org.yah.tests.perceptron;

import java.nio.IntBuffer;
import java.util.function.Consumer;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.InputAdapter;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.Pixmap.Format;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Matrix4;

/**
 * @author Yah
 *
 */
public class AbstractGLDemo extends InputAdapter implements GLDemo {

    private static final Format TEXTURE_FORMAT = Format.RGBA8888;

    protected SpriteBatch spriteBatch;

    private Pixmap buffer;

    private Texture texture;

    private Consumer<Pixmap> nextUpdater;
    private int width, height;

    @Override
    public void create() {
        spriteBatch = new SpriteBatch();
        width = Gdx.graphics.getWidth();
        height = Gdx.graphics.getHeight();
        Gdx.input.setInputProcessor(this);
        
    }

    protected final synchronized void schedule(Consumer<Pixmap> nextUpdater) {
        this.nextUpdater = nextUpdater;
    }

    @Override
    public String getTitle() { return getClass().getSimpleName(); }

    @Override
    public void dispose() {
        spriteBatch.dispose();
    }

    @Override
    public void render() {
        if (texture == null) return;

        Consumer<Pixmap> updater = popUpdater();
        if (updater != null) {
            updater.accept(buffer);
            texture.draw(buffer, 0, 0);
        }
        spriteBatch.begin();
        render(spriteBatch);
        spriteBatch.end();
    }

    protected void render(SpriteBatch spriteBatch) {
        spriteBatch.draw(texture, 0, 0, width, height);
    }

    private synchronized Consumer<Pixmap> popUpdater() {
        Consumer<Pixmap> res = nextUpdater;
        nextUpdater = null;
        return res;
    }

    @Override
    public void resize(int width, int height) {
        this.width = width;
        this.height = height;
        spriteBatch.setProjectionMatrix(new Matrix4().setToOrtho2D(0, 0,
                Gdx.graphics.getWidth(), Gdx.graphics.getHeight()));
    }

    protected final void createBuffer(int width, int height) {
        if (buffer != null)
            buffer.dispose();
        buffer = new Pixmap(width, height, TEXTURE_FORMAT);
        if (texture != null)
            texture.dispose();
        texture = new Texture(buffer);
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
