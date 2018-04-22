#include <glad/glad.h>
#include "fs_quad.h"

static const GLfloat g_vertex_buffer_data[] = {
/*	position x, y	texcoord u,v */
   -1.0f, -1.0f,	0.0f, 1.0f, 
   -1.0f,  1.0f,	0.0f, 0.0f,
    1.0f, -1.0f,	1.0f, 1.0f,
	1.0f,  1.0f,	1.0f, 0.0f,
};

GLuint fsQuad_vBuffer;

void fsQuadInitialize()
{
    // Generate 1 buffer, put the resulting identifier in vertexbuffer
    glGenBuffers(1, &fsQuad_vBuffer);
    // The following commands will talk about our 'vertexbuffer' buffer
    glBindBuffer(GL_ARRAY_BUFFER, fsQuad_vBuffer);
    // Give our vertices to OpenGL.
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
}

void fsQuadDestroy()
{
    glDeleteBuffers(1, &fsQuad_vBuffer);
}

extern void fsQuadDraw()
{
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, fsQuad_vBuffer);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0 );
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    
    // Draw the triangle !
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray(0);
}