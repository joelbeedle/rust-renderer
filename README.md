# rust-renderer

Fun project I did to help me to learn rust.

Essentially a renderer that can take a 3D model and render it, using only `minifb` for the window and frame buffer creation.

It can take `.obj` and `.off` files and render them, with multiple types of lighting available.

Also has support for Bézier patch conversions into triangles at varying resolutions.

## Features

- Back-face culling
- Painters algorithm
- Flat Shading
- Gouraud Shading
- Phong Shading

## TODO

- [ ] Add depth buffer
- [ ] Formalise functions to make it easier to use
- [ ] Allow lighting selections
- [ ] Simplify example binary
- [ ] Textures?
