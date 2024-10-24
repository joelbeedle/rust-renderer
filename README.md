# rust-renderer

Fun project I did to help me to learn rust.

Essentially a renderer that can take a 3D model and render it, using only `minifb` for the window and frame buffer creation.

It can take `.obj` and `.off` files and render them, with multiple types of lighting available.

Also has support for Bézier patch conversions into triangles at varying resolutions.

## Features

- [Back-face culling](https://en.wikipedia.org/wiki/Back-face_culling)
- [Painters algorithm](https://en.wikipedia.org/wiki/Painter%27s_algorithm)
- [Flat Shading](https://en.wikipedia.org/w/index.php?title=Shading&section=13#Flat_shading)
- [Gouraud Shading](https://en.wikipedia.org/wiki/Gouraud_shading)
- [Phong Shading](https://en.wikipedia.org/wiki/Phong_shading)

<img width="560" alt="image" src="https://github.com/user-attachments/assets/e9407655-4236-4ebd-ae29-416a989cf2f5">

## TODO

- [ ] Add depth buffer
- [ ] Formalise functions to make it easier to use
- [ ] Allow lighting selections
- [ ] Simplify example binary
- [ ] Textures?
