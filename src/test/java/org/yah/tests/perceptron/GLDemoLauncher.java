package org.yah.tests.perceptron;

import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;

public final class GLDemoLauncher {

	private GLDemoLauncher() {}

	public static void launch(GLDemo demo) {
		Lwjgl3ApplicationConfiguration config = new Lwjgl3ApplicationConfiguration();
		config.useOpenGL3(true, 3, 3);
		config.setTitle(demo.getTitle());
		demo.configure(config);
		new Lwjgl3Application(demo, config);
	}
}
