package org.yah.tests.perceptron;

import com.badlogic.gdx.ApplicationListener;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;

public interface GLDemo extends ApplicationListener {

	String getTitle();

	@Override
	default void create() {}

	@Override
	default void resize(int width, int height) {}

	@Override
	void render();

	@Override
	default void pause() {}

	@Override
	default void resume() {}

	@Override
	default void dispose() {}

	void configure(Lwjgl3ApplicationConfiguration config);
	
}
