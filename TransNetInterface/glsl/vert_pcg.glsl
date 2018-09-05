// Uniforms
// ------------------------------------
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform  float u_size;

// Attributes
// ------------------------------------
attribute vec3 a_position;
attribute vec4 a_color;

// Varying
// ------------------------------------
varying vec4 v_color;

void main()
{
    v_color = a_color;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    gl_PointSize = u_size;
}