# rust-renderer

Rendering project that helped me to learn rust.

In essence, it's a renderer that can take any 3D model and render it. It doesn't use GPU acceleration, only `minifb` for the window and a frame buffer.

All rendering, camera, projection is done manually, writing pixels to the framebuffer (essentially an array of pixels). This means it's slow, but allowed me to focus on the things that were interesting in the project.

It can take `.obj` and `.off` files and render them, with multiple types of lighting available.

Also has support for Bézier patch conversions into triangles at varying resolutions.

## Features

- [Back-face culling](https://en.wikipedia.org/wiki/Back-face_culling)
- [Painters algorithm](https://en.wikipedia.org/wiki/Painter%27s_algorithm)
- [Flat Shading](https://en.wikipedia.org/w/index.php?title=Shading&section=13#Flat_shading)
- [Gouraud Shading](https://en.wikipedia.org/wiki/Gouraud_shading)
- [Phong Shading](https://en.wikipedia.org/wiki/Phong_shading)

<img width="560" alt="image" src="https://github.com/user-attachments/assets/e9407655-4236-4ebd-ae29-416a989cf2f5">
