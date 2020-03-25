#version 330 core

uniform vec4 flowerColors[2];
uniform isampler2D expectedFlowers;
uniform isampler2D actualFlowers;
uniform isampler2D sampledFlowers;

#define DIM 0.7

in vec2 inputs;

out vec4 outColor;

void main() {
  int sampled = texture(sampledFlowers, inputs, 0).r;
  int expected = texture(expectedFlowers, inputs, 0).r;
  int actual = texture(actualFlowers, inputs, 0).r;
  if (sampled == 1) {
    outColor = vec4(1);
  } else {
    outColor = flowerColors[expected];
  }
  if (actual != expected)
    outColor *= DIM;
}
