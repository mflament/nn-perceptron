#version 330 core

in vec2 position;
in vec2 aInputs;
out vec2 inputs;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
  inputs = aInputs;
}
