extern crate minifb_renderer;
use minifb::{Key, Window, WindowOptions};

use minifb_renderer::*;

fn main() {
  let mut window = Window::new(
    "Test",
    WIDTH,
    HEIGHT,
    WindowOptions {
      ..WindowOptions::default()
    },
  )
  .unwrap_or_else(|e| {
    panic!("{}", e);
  });

  let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

  let mut camera = Camera::new(
    Point3D {
      x: -10.0,
      y: 0.0,
      z: -10.0,
    },
    90.0,
  );

  let cube = Cube::new(Point3D {
    x: 0.0,
    y: 0.0,
    z: 0.0,
  });
  let cube2 = Cube::new(Point3D {
    x: 2.0,
    y: 0.0,
    z: 0.0,
  });
  let cube3 = Cube::new(Point3D {
    x: 4.0,
    y: 0.0,
    z: 0.0,
  });
  let cube4 = Cube::new(Point3D {
    x: 0.0,
    y: 2.0,
    z: 0.0,
  });
  let cubes: Vec<Cube> = vec![cube, cube2, cube3, cube4];
  // let mut triangles = all_cube_faces(&cubes);
  let mut triangles = vec![];
  let data = io::read_data("models/teapot_data.txt").unwrap();
  let off_data = io::read_off_file("models/m114.off").unwrap();
  let obj_data = io::read_obj_file("models/Dixie_V2.obj").unwrap();
  // let mut off_indices: Vec<Point3D> = off_data.0;
  //let mut off_triangles: Vec<Triangle3D> = off_data.1;
  let mut off_triangles: Vec<Triangle3D> = obj_data;
  let indices: Vec<Point3D> = data.0;
  let bezier_patches: Vec<BezierPatch> = data.1;
  let mut tri: Vec<Vec<Triangle3D>> = Vec::new();

  for patch in bezier_patches {
    let mut itopoint: Vec<Point3D> = Vec::new();
    for point in patch.indices {
      itopoint.push(*indices.get(point - 1).unwrap());
    }
    let temp = io::tessellate_bezier_patch(&itopoint, 12);
    tri.push(temp);
  }
  for t in tri {
    for a in t {
      triangles.push(a);
    }
  }

  const MOVE_SPEED: f32 = 0.01;

  const ROTATE_SPEED: f32 = 0.01;

  let mut show_wireframe: bool = false;
  let mut show_faces: bool = true;

  while window.is_open() && !window.is_key_down(Key::Escape) {
    // Adjust x and y based on user input
    if window.is_key_down(Key::W) {
      // Move forward along the camera's forward vector
      camera.position.x += camera.forward.x * MOVE_SPEED;
      camera.position.y += camera.forward.y * MOVE_SPEED;
      camera.position.z += camera.forward.z * MOVE_SPEED;
    }
    if window.is_key_down(Key::S) {
      // Move backward along the camera's forward vector
      camera.position.x -= camera.forward.x * MOVE_SPEED;
      camera.position.y -= camera.forward.y * MOVE_SPEED;
      camera.position.z -= camera.forward.z * MOVE_SPEED;
    }
    if window.is_key_down(Key::A) {
      // Move left along the camera's right vector
      camera.position.x -= camera.right.x * MOVE_SPEED;
      camera.position.y -= camera.right.y * MOVE_SPEED;
      camera.position.z -= camera.right.z * MOVE_SPEED;
    }
    if window.is_key_down(Key::D) {
      // Move right along the camera's right vector
      camera.position.x += camera.right.x * MOVE_SPEED;
      camera.position.y += camera.right.y * MOVE_SPEED;
      camera.position.z += camera.right.z * MOVE_SPEED;
    }
    if window.is_key_down(Key::Space) {
      camera.position.x += camera.up.x * MOVE_SPEED;
      camera.position.y += camera.up.y * MOVE_SPEED;
      camera.position.z += camera.up.z * MOVE_SPEED;
    }
    if window.is_key_down(Key::LeftShift) {
      camera.position.x -= camera.up.x * MOVE_SPEED;
      camera.position.y -= camera.up.y * MOVE_SPEED;
      camera.position.z -= camera.up.z * MOVE_SPEED;
    }
    if window.is_key_down(Key::Right) {
      camera.yaw -= ROTATE_SPEED;
    }
    if window.is_key_down(Key::Left) {
      camera.yaw += ROTATE_SPEED;
    }
    if window.is_key_down(Key::Up) {
      camera.pitch += ROTATE_SPEED;
    }
    if window.is_key_down(Key::Down) {
      camera.pitch -= ROTATE_SPEED;
    }
    if window.is_key_released(Key::T) {
      show_wireframe = !show_wireframe;
    }
    if window.is_key_released(Key::F) {
      show_faces = !show_faces;
    }
    if window.is_key_released(Key::Semicolon) {
      dbg!(camera.position);
    }
    buffer.iter_mut().for_each(|p| *p = 0);
    off_triangles.sort_by(|a, b| {
      let depth_a = graphics::average_depth(a, &camera);
      let depth_b = graphics::average_depth(b, &camera);
      depth_b
        .partial_cmp(&depth_a)
        .unwrap_or(std::cmp::Ordering::Equal)
    });

    for triangle in &mut off_triangles {
      if show_faces {
        graphics::fill_triangle3d(&mut buffer, WIDTH, HEIGHT, triangle, &camera, 0xFF0000);
        // Fill triangle with red
      }
      if show_wireframe {
        graphics::draw_triangle3d(&mut buffer, WIDTH, HEIGHT, triangle, &camera, 0xffffff);
      }
    }

    camera.update_vectors();
    window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
  }
}
