#version 410 core
in  vec2 texCoord;
out vec3 color;

uniform sampler2D tex;

void main(){
  // vec3 scale = vec3(0.5f, 0.0f, -0.5f); // positive and negative view
  color = vec3(texture(tex, texCoord).r * 0.5f);
}