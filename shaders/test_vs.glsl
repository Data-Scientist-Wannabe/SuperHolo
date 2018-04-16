#version 410 core

in     vec2 pos;
in     vec2 uv;

out    vec2 texCoord;

void main()
{
    gl_Position = vec4(pos.xy, 0.0f, 1.0f);
    texCoord = uv;
}