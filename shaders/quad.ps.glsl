#version 410 core
in  vec2 texCoord;
out vec3 color;

uniform sampler2D tex;

void main(){
  color = vec3(texture(tex, texCoord).r);
}