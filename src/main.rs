use minifb::{Key, Window, WindowOptions, Scale};

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;

#[derive(Copy, Clone, Debug)]
struct Point3D {
  x: f32,
  y: f32,
  z: f32,
}

impl Point3D {
  fn subtract(&self, other: &Point3D) -> Point3D {
    Point3D {
      x: self.x - other.x,
      y: self.y - other.y,
      z: self.z - other.z,
    }
  }

  fn cross(&self, other: &Point3D) -> Point3D {
    Point3D {
      x: self.y * other.z - self.z * other.y,
      y: self.z * other.x - self.x * other.z,
      z: self.x * other.y - self.y * other.x,
    }
  }

  fn normalize(&self) -> Point3D {
    let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
    Point3D {
      x: self.x / len,
      y: self.y / len,
      z: self.z / len,
    }
  }

  fn multiply_scalar(&self, scalar: f32) -> Point3D {
      Point3D {
          x: self.x * scalar,
          y: self.y * scalar,
          z: self.z * scalar,
      }
  }

  fn dot(&self, other: &Point3D) -> f32 {
    self.x * other.x + self.y * other.y + self.z * other.z
  }

  fn magnitude(&self) -> f32 {
    (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
  }
}

#[derive(Debug)]
struct Camera {
  position: Point3D,
  forward: Point3D, // Direction camera is facing
  right: Point3D,   // Right vector
  up: Point3D,      // Up vector
  fov: f32,
  pitch: f32,
  yaw: f32,
}

#[derive(Copy, Clone, Debug)]
struct Vertex3D {
  position: Point3D,
  normal: Point3D,
  color: u32,
}

#[derive(Debug)]
struct Triangle3D {
  v1: Vertex3D,
  v2: Vertex3D,
  v3: Vertex3D,
  normal: Point3D,
}

impl Triangle3D {
  fn new(v1: Vertex3D, v2: Vertex3D, v3: Vertex3D) -> Triangle3D {
    let edge1 = v2.position.subtract(&v1.position);
    let edge2 = v3.position.subtract(&v1.position);
    let normal = edge2.cross(&edge1).normalize();
    Triangle3D { v1, v2, v3, normal }
  }
}

#[derive(Debug)]
struct BezierPatch {
  indices: [usize; 16]
}

struct Cube {
  vertices: [Point3D; 8],
}

impl Camera {
  fn project(&self, point: &Point3D, width: usize, height: usize) -> Point3D {
    let translated_point = Point3D {
      x: point.x - self.position.x,
      y: point.y - self.position.y,
      z: point.z - self.position.z,
    };

    // Apply rotations
    let rotated_point = apply_matrix(translated_point, rotate_y(self.yaw));
    let rotated_point = apply_matrix(rotated_point, rotate_x(self.pitch));

    let aspect_ratio = width as f32 / height as f32;
    let scale = (self.fov.to_radians() / 2.0).tan();
    let screen_x = (rotated_point.x / (rotated_point.z * scale * aspect_ratio)) * width as f32 / 2.0 + width as f32 / 2.0;
    let screen_y = -(rotated_point.y / (rotated_point.z * scale)) * height as f32 / 2.0 + height as f32 / 2.0;

    Point3D { x: screen_x, y: screen_y, z: rotated_point.z }
  }

    fn new(position: Point3D, fov: f32) -> Self {
      Camera {
        position,
        forward: Point3D { x: 0.0, y: 0.0, z: -1.0 },
        right: Point3D { x: 0.0, y: 0.0, z: 0.0 },
        up: Point3D { x: 0.0, y: 1.0, z: 0.0 },
        fov,
        yaw: 0.0,
        pitch: 0.0,
      }
    }

  fn update_vectors(&mut self) {
      // Calculate the new forward vector
    // Calculate the new forward vector based on yaw (rotation around the y-axis)
    self.forward = Point3D {
        x: self.pitch.to_radians().cos() * self.yaw.to_radians().cos(),
        y: self.pitch.to_radians().sin(),
        z: self.pitch.to_radians().cos() * self.yaw.to_radians().sin() 
    };

    // Normalize the forward vector
    let forward_len = (self.forward.x.powi(2) + self.forward.y.powi(2) + self.forward.z.powi(2)).sqrt();
    self.forward = Point3D {
        x: self.forward.x / forward_len,
        y: self.forward.y / forward_len,
        z: self.forward.z / forward_len,
    };

    // Calculate the right vector as the cross product of the world up vector and forward vector
    let world_up = Point3D { x: 0.0, y: 1.0, z: 0.0 };
    self.right = cross_product(&self.forward, &world_up);

    // Normalize the right vector
    let right_len = (self.right.x.powi(2) + self.right.y.powi(2) + self.right.z.powi(2)).sqrt();
    self.right = Point3D {
        x: self.right.x / right_len,
        y: self.right.y / right_len,
        z: self.right.z / right_len
    };

    // The up vector should be recalculated even if not used directly
    self.up = cross_product(&self.right, &self.forward);
  }
}

impl Cube {
  fn new(rel_pos: Point3D) -> Self {
    Cube {
      vertices: [
        Point3D { x: -1.0 + rel_pos.x, y:  1.0 + rel_pos.y, z: -1.0 + rel_pos.z },  // Front top left
        Point3D { x:  1.0 + rel_pos.x, y:  1.0 + rel_pos.y, z: -1.0 + rel_pos.z },  // Front top right
        Point3D { x: -1.0 + rel_pos.x, y: -1.0 + rel_pos.y, z: -1.0 + rel_pos.z },  // Front bottom left
        Point3D { x:  1.0 + rel_pos.x, y: -1.0 + rel_pos.y, z: -1.0 + rel_pos.z },  // Front bottom right
        Point3D { x: -1.0 + rel_pos.x, y:  1.0 + rel_pos.y, z:  1.0 + rel_pos.z },  // Back top left
        Point3D { x:  1.0 + rel_pos.x, y:  1.0 + rel_pos.y, z:  1.0 + rel_pos.z },  // Back top right
        Point3D { x: -1.0 + rel_pos.x, y: -1.0 + rel_pos.y, z:  1.0 + rel_pos.z },  // Back bottom left
        Point3D { x:  1.0 + rel_pos.x, y: -1.0 + rel_pos.y, z:  1.0 + rel_pos.z },  // Back bottom right
      ]
    }
  }
}

fn main() {
  let mut window = Window::new(
    "Test",
    WIDTH,
    HEIGHT,
    WindowOptions {
      ..WindowOptions::default()
    },
  ).unwrap_or_else(|e| {
    panic!("{}", e);
  });

  let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
  
  let mut camera = Camera::new(Point3D { x: -10.0, y: 0.0, z: -10.0 }, 90.0);

  let cube = Cube::new(Point3D {x: 0.0, y: 0.0, z: 0.0});
  let cube2 = Cube::new(Point3D {x: 2.0, y: 0.0, z: 0.0});
  let cube3 = Cube::new(Point3D {x: 4.0, y: 0.0, z: 0.0});
  let cube4 = Cube::new(Point3D {x: 0.0, y: 2.0, z: 0.0});
  let cubes: Vec<Cube> = vec![cube, cube2, cube3, cube4];
  // let mut triangles = all_cube_faces(&cubes);
  let mut triangles = vec![];
  let data = read_data("teapot_data.txt").unwrap();
  let off_data = read_off_file("m114.off").unwrap();
  let obj_data = read_obj_file("Mesh_Cat.obj").unwrap(); 
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
    let temp = tessellate_bezier_patch(&itopoint, 6);
    tri.push(temp);
  }
  for t in tri {
    for a in t {
      triangles.push(a);
    }
  }

  const MOVE_SPEED: f32 = 0.05;

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
    // dbg!(camera.position);
    buffer.iter_mut().for_each(|p| *p = 0);
    off_triangles.sort_by(|a, b| {
        let depth_a = average_depth(a, &camera);
        let depth_b = average_depth(b, &camera);
        depth_b.partial_cmp(&depth_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    for triangle in &mut off_triangles {
      if show_faces {
        fill_triangle3d(&mut buffer, WIDTH, HEIGHT, triangle, &camera, 0xFF0000); // Fill triangle with red
      }
      if show_wireframe { 
        draw_triangle3d(&mut buffer, WIDTH, HEIGHT, triangle, &camera, 0xffffff);
      }
    }

    camera.update_vectors();
    window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
  }
}


fn rgb_to_color(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}


fn color_to_rgb_f32(color: u32) -> (f32, f32, f32) {
    (
        ((color >> 16) & 0xFF) as f32 / 255.0,
        ((color >> 8) & 0xFF) as f32 / 255.0,
        (color & 0xFF) as f32 / 255.0,
    )
}

fn color_to_rgb(color: u32) -> (u8, u8, u8) {
    (
        ((color >> 16) & 0xFF) as u8,
        ((color >> 8) & 0xFF) as u8,
        (color & 0xFF) as u8,
    )
}

fn compute_lighting_all(vertex: &mut Vertex3D, light_position: &Point3D, camera_position: &Point3D) -> u32 {
  let normal = vertex.normal;
  let point_position = &vertex.position;
  let (r_a, g_a, b_a) = color_to_rgb_f32(0xff_00_00);
  let ambient_intensity = 0.2;
  let diffuse_intensity = 0.7;
  let specular_intensity = 0.9;
  let shininess = 128.0; // Shininess coefficient for specular highlights

  let light_direction = vector_sub(light_position, point_position).normalize();
  let view_direction = vector_sub(camera_position, point_position).normalize();
  let reflect_direction = reflect_vector(&(light_direction), &normal);

  let ambient = ambient_intensity;

  let diffuse = diffuse_intensity * normal.dot(&light_direction).max(0.0);

  let specular = if diffuse > 0.0 {
      specular_intensity * view_direction.dot(&reflect_direction).max(0.0).powf(shininess)
  } else {
      0.0
  };

  let light_value = (ambient + diffuse + specular).min(1.0);
  let r = ((r_a * light_value) * 255.0) as u8;
  let g = ((g_a * light_value) * 255.0) as u8;
  let b = ((b_a * light_value) * 255.0) as u8;
  let brightness = (light_value * 255.0) as u8;
  rgb_to_color(r, g, b)
}

fn compute_lighting_linatten(vertex: &mut Vertex3D, light_position: &Point3D) -> u32 {
  let normal: Point3D = vertex.normal;
  let point_position: Point3D = vertex.position;
  let (r_a, g_a, b_a) = color_to_rgb_f32(0xff_00_00);
  // let light_direction = light_position.subtract(&point_position).normalize();
  let light_direction = vector_sub(light_position, &point_position);
  let distance = light_direction.magnitude();
  let attenuation = 1.0 / (1.0 + 0.01 * distance); // Linear attenuation

  let dot_product = normal.dot(&light_direction.normalize()).max(0.05);
  let light_intensity = dot_product * attenuation;
  let brightness = (light_intensity * 255.0) as u8;
  let ambient_intensity = 0.7;
  let diffuse_intensity = 0.9;
  let r = ((r_a * (ambient_intensity + light_intensity * diffuse_intensity)) * 255.0) as u8;
  let g = ((g_a * (ambient_intensity + light_intensity * diffuse_intensity)) * 255.0) as u8;
  let b = ((b_a * (ambient_intensity + light_intensity * diffuse_intensity)) * 255.0) as u8;

  rgb_to_color(r, g, b)
}

fn reflect_vector(incident: &Point3D, normal: &Point3D) -> Point3D {
    // Compute dot product of incident vector and normal
    let dot_product = incident.dot(normal);

    // R = I - 2*(I·N)*N
    let reflected = incident.subtract(&normal.multiply_scalar(2.0 * dot_product));
    reflected
}

fn compute_lighting(normal: &mut Point3D, light_dir: &Point3D) -> u32 {
  let light_intensity = normal.dot(light_dir).max(0.05).min(0.5); // Clamp to [0,1]
  let brightness = (light_intensity * 255.0) as u8;
  rgb_to_color(brightness, brightness, brightness)
}

fn interpolate_color(color1: u32, color2: u32) -> u32 {
    let (r1, g1, b1) = color_to_rgb(color1);
    let (r2, g2, b2) = color_to_rgb(color2);

    let r = ((r1 as f32 + r2 as f32) / 2.0) as u8;
    let g = ((g1 as f32 + g2 as f32) / 2.0) as u8;
    let b = ((b1 as f32 + b2 as f32) / 2.0) as u8;

    rgb_to_color(r, g, b)
}

fn average_depth(triangle: &Triangle3D, camera: &Camera) -> f32 {
  let p1 = camera.project(&triangle.v1.position, 1, 1);
  let p2 = camera.project(&triangle.v2.position, 1, 1);
  let p3 = camera.project(&triangle.v3.position, 1, 1);
  (p1.z + p2.z + p3.z) / 3.0
}

fn is_on_screen(point: &Point3D, width: usize, height: usize) -> bool {
    point.x >= 0.0 && point.x < width as f32 && point.y >= 0.0 && point.y < height as f32
}

fn draw_triangle3d(buffer: &mut Vec<u32>, width: usize, height: usize, triangle: &Triangle3D, camera: &Camera, color: u32) {
    let p1 = camera.project(&triangle.v1.position, width, height);
    let p2 = camera.project(&triangle.v2.position, width, height);
    let p3 = camera.project(&triangle.v3.position, width, height);

    // Check if all vertices are off-screen
    if !is_on_screen(&p1, width, height) && !is_on_screen(&p2, width, height) && !is_on_screen(&p3, width, height) {
        return; // Skip drawing this triangle entirely if all points are off-screen
    }

    let centroid = Point3D {
        x: (triangle.v1.position.x + triangle.v2.position.x + triangle.v3.position.x) / 3.0,
        y: (triangle.v1.position.y + triangle.v2.position.y + triangle.v3.position.y) / 3.0,
        z: (triangle.v1.position.z + triangle.v2.position.z + triangle.v3.position.z) / 3.0,
    };

    let view_direction = vector_sub(&centroid, &camera.position);

    if dot_product(&triangle.normal, &view_direction) > 0.0 {  // Reverse condition to > 0.0 if normals are outward
      if p1.z > 0.1 && p2.z > 0.1 && p3.z > 0.1 {
        draw_line(buffer, width, &p1, &p2, color);
        draw_line(buffer, width, &p2, &p3, color);
        draw_line(buffer, width, &p3, &p1, color);
      }
    }
}

fn draw_line(buffer: &mut [u32], width: usize, start: &Point3D, end: &Point3D, color: u32) {
  let mut x0 = start.x as isize;
  let mut y0 = start.y as isize;
  let x1 = end.x as isize;
  let y1 = end.y as isize;

  let dx = (x1 - x0).abs();
  let sx = if x0 < x1 { 1 } else { -1 };
  let dy = -(y1 - y0).abs();
  let sy = if y0 < y1 { 1 } else { -1 };
  let mut err = dx + dy;

  while x0 != x1 || y0 != y1 {
    if x0 >= 0 && y0 >= 0 && (x0 as usize) < width && (y0 as usize) < buffer.len() / width {
      buffer[(x0 as usize) + (y0 as usize) * width] = color;
    }
    let e2 = 2 * err;
    if e2 >= dy {
      if x0 == x1 {
          break;
      }
      err += dy;
      x0 += sx;
    }
    if e2 <= dx {
      if y0 == y1 {
          break;
      }
      err += dx;
      y0 += sy;
    }
  }
}

fn fill_triangle(p1: &Point3D, p2: &Point3D, p3: &Point3D, color: u32, buffer: &mut [u32]) {
  let v1 = (p1.x as isize , p1.y as isize);
  let v2 = (p2.x as isize, p2.y as isize);
  let v3 = (p3.x as isize, p3.y as isize);
  let mut vertices = [v1, v2, v3];

  // Sort vertices by y-coordinate (ascending)
  vertices.sort_by_key(|k| k.1);

  // Lambda to interpolate x-coordinates on edges
  let interpolate_edge = |(x0, y0): (isize, isize), (x1, y1): (isize, isize), y: isize| -> isize {
    if y1 == y0 {
      x0
    } else {
      x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    }
  };

  // Scan-line fill between vertices
  let (v1, v2, v3) = (vertices[0], vertices[1], vertices[2]);

  // Interpolate between vertex 1 and vertex 2, and vertex 1 and vertex 3 simultaneously
  for y in v1.1..=v3.1 {
    let x_start = if y < v2.1 {
      interpolate_edge(v1, v2, y)
    } else {
      interpolate_edge(v2, v3, y)
    };
    let x_end = interpolate_edge(v1, v3, y);

    let (start, end) = if x_start < x_end { (x_start, x_end) } else { (x_end, x_start) };
    for x in start..=end {
      if x >= 0 && (x as usize) < WIDTH && y >= 0 && (y as usize) < HEIGHT {
        buffer[(y as usize) * WIDTH + (x as usize)] = color;
      }
      if x_start == x_end {break}
    }
  }
}

/// Calculate the barycentric coordinates of point p with respect to triangle (a, b, c)
fn barycentric_coords(p: &Point3D, a: &Point3D, b: &Point3D, c: &Point3D) -> (f32, f32, f32) {
    let v0 = (b.x - a.x, b.y - a.y);
    let v1 = (c.x - a.x, c.y - a.y);
    let v2 = (p.x - a.x, p.y - a.y);
    let d00 = v0.0 * v0.0 + v0.1 * v0.1;
    let d01 = v0.0 * v1.0 + v0.1 * v1.1;
    let d11 = v1.0 * v1.0 + v1.1 * v1.1;
    let d20 = v2.0 * v0.0 + v2.1 * v0.1;
    let d21 = v2.0 * v1.0 + v2.1 * v1.1;
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, v, w)
}

/// Interpolate color at a specific point inside the triangle
fn interpolate_color_at(u: f32, v: f32, w: f32, color1: u32, color2: u32, color3: u32) -> u32 {
    let (r1, g1, b1) = color_to_rgb(color1);
    let (r2, g2, b2) = color_to_rgb(color2);
    let (r3, g3, b3) = color_to_rgb(color3);

    let r = (u * r1 as f32 + v * r2 as f32 + w * r3 as f32) as u8;
    let g = (u * g1 as f32 + v * g2 as f32 + w * g3 as f32) as u8;
    let b = (u * b1 as f32 + v * b2 as f32 + w * b3 as f32) as u8;

    rgb_to_color(r, g, b)
}


fn rasterize_filled_triangle(
    buffer: &mut Vec<u32>, width: usize, height: usize,
    p1: &Point3D, p2: &Point3D, p3: &Point3D,
    color1: u32, color2: u32, color3: u32
) {
    let mut bbox_min = Point3D { x: width as f32 - 1.0, y: height as f32 - 1.0, z: 0.0 };
    let mut bbox_max = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let clamp = Point3D { x: (width as f32 - 1.0), y: (height as f32 - 1.0) , z: 0.0 };

    // Compute bounding box
    for p in [p1, p2, p3].iter() {
      let zero: f32 = 0.0;
        bbox_min.x = zero.max(bbox_min.x.min(p.x));
        bbox_min.y = zero.max(bbox_min.y.min(p.y));
        bbox_max.x = clamp.x.min(bbox_max.x.max(p.x));
        bbox_max.y = clamp.y.min(bbox_max.y.max(p.y));
    }

    // Clamping to screen dimensions
    bbox_min.x = bbox_min.x.max(0.0).floor();
    bbox_min.y = bbox_min.y.max(0.0).floor();
    bbox_max.x = bbox_max.x.min(clamp.x).ceil();
    bbox_max.y = bbox_max.y.min(clamp.y).ceil();

    // Scan through bounding box and fill pixels within the triangle
    for y in bbox_min.y as i32..=bbox_max.y as i32 {
        for x in bbox_min.x as i32..=bbox_max.x as i32 {
            let p = Point3D { x: x as f32, y: y as f32, z: 0.0 };
            let (u, v, w) = barycentric_coords(&p, p1, p2, p3);
            if u >= 0.0 && v >= 0.0 && w >= 0.0 { // The point is inside the triangle
                let idx = (y * width as i32 + x) as usize;
                buffer[idx] = interpolate_color_at(u, v, w, color1, color2, color3);
            }
        }
    }
}

fn fill_triangle3d(buffer: &mut Vec<u32>, width: usize, height: usize, triangle: &mut Triangle3D, camera: &Camera, color: u32) {
    let p1 = camera.project(&triangle.v1.position, width, height);
    let p2 = camera.project(&triangle.v2.position, width, height);
    let p3 = camera.project(&triangle.v3.position, width, height);

    // Check if all vertices are off-screen
    if !is_on_screen(&p1, width, height) && !is_on_screen(&p2, width, height) && !is_on_screen(&p3, width, height) {
        return; // Skip drawing this triangle entirely if all points are off-screen
    }
    let centroid = Point3D {
        x: (triangle.v1.position.x + triangle.v2.position.x + triangle.v3.position.x) / 3.0,
        y: (triangle.v1.position.y + triangle.v2.position.y + triangle.v3.position.y) / 3.0,
        z: (triangle.v1.position.z + triangle.v2.position.z + triangle.v3.position.z) / 3.0,
    };

    let light_position = Point3D { x: 10.0, y: 10.0, z: -3.0 };

    triangle.v1.color = compute_lighting_all(&mut triangle.v1, &light_position, &camera.position);
    triangle.v2.color = compute_lighting_all(&mut triangle.v2, &light_position, &camera.position);
    triangle.v3.color = compute_lighting_all(&mut triangle.v3, &light_position, &camera.position);

    let view_direction = vector_sub(&centroid, &camera.position);
    // let c = compute_lighting(&mut triangle.normal, &light_position);

    if dot_product(&triangle.normal, &view_direction) > 0.0 {  // Reverse condition to > 0.0 if normals are outward
      if p1.z > 0.1 && p2.z > 0.1 && p3.z > 0.1 {
        //fill_triangle(&p1, &p2, &p3, c as u32, buffer);
        rasterize_filled_triangle(buffer, width, height, &p1, &p2, &p3, triangle.v1.color, triangle.v2.color, triangle.v3.color);
      }
    }
}

fn rotate_x(angle: f32) -> [[f32; 3]; 3] {
  let cos_a = angle.cos();
  let sin_a = angle.sin();
  [
    [1.0, 0.0, 0.0],
    [0.0, cos_a, -sin_a],
    [0.0, sin_a, cos_a],
  ]
}

fn rotate_y(angle: f32) -> [[f32; 3]; 3] {
  let cos_a = angle.cos();
  let sin_a = angle.sin();
  [
    [cos_a, 0.0, sin_a],
    [0.0, 1.0, 0.0],
    [-sin_a, 0.0, cos_a],
  ]
}

fn apply_matrix(point: Point3D, matrix: [[f32; 3]; 3]) -> Point3D {
  Point3D {
    x: point.x * matrix[0][0] + point.y * matrix[0][1] + point.z * matrix[0][2],
    y: point.x * matrix[1][0] + point.y * matrix[1][1] + point.z * matrix[1][2],
    z: point.x * matrix[2][0] + point.y * matrix[2][1] + point.z * matrix[2][2],
  }
}
 
// Vector subtraction
fn vector_sub(a: &Point3D, b: &Point3D) -> Point3D {
    Point3D { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }
}

// Cross product
fn cross_product(a: &Point3D, b: &Point3D) -> Point3D {
    Point3D {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

// Dot product
fn dot_product(a: &Point3D, b: &Point3D) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn read_data<P: AsRef<Path>>(path: P) -> io::Result<(Vec<Point3D>, Vec<BezierPatch>)> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut vertices = Vec::new();
    let mut patches = Vec::new();
    let mut read_vertices = false;
    let mut read_patches = false;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
          continue;
            // This line is a comment or empty, skip it.
        } else if line.contains("Vertex Coordinates") {
            read_vertices = true;
            read_patches = false;
            continue;
        } else if line.contains("Bezier Patches") {
            read_vertices = false;
            read_patches = true;
            continue;
        }

        if read_vertices {
            if let Some(vertex) = parse_vertex(&line) {
                vertices.push(vertex);
            }
        } else if read_patches {
            if let Some(patch) = parse_patch(&line) {
                patches.push(patch);
            }
        }
    }

    Ok((vertices, patches))
}

fn read_obj_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<Triangle3D>> {
  let file = File::open(path)?;
  let reader = io::BufReader::new(file);

  let mut vertices = Vec::new();
  let mut normals = Vec::new();
  let mut triangles = Vec::new();

  for line in reader.lines() {
    let line = line?;
    let mut parts = line.split_whitespace();
    match parts.next() {
      Some("v") => {
        let x = parts.next().unwrap().parse::<f32>().unwrap();
        let y = parts.next().unwrap().parse::<f32>().unwrap();
        let z = parts.next().unwrap().parse::<f32>().unwrap();
        vertices.push(Point3D { x, y, z });
      },
      Some("vn") => {
        let x = parts.next().unwrap().parse::<f32>().unwrap();
        let y = parts.next().unwrap().parse::<f32>().unwrap();
        let z = parts.next().unwrap().parse::<f32>().unwrap();
        normals.push(Point3D {x, y, z});
      },
      Some("f") => {

        let indices: Vec<(usize, Option<usize>, Option<usize>)> = parts.map(|part| {
          let mut indices = part.split('/');
          let vertex_idx = indices.next().unwrap().parse::<usize>().unwrap() - 1;
          let tex_idx = indices.next().and_then(|t| if t.is_empty() { None } else { t.parse::<usize>().ok() }).map(|t| t - 1);
          let norm_idx = indices.next().and_then(|n| n.parse::<usize>().ok()).map(|n| n - 1);

          (vertex_idx, tex_idx, norm_idx)
        }).collect();
        // let idx: Vec<usize> = parts.map(|part| part.split('/').next().unwrap().parse::<usize>().unwrap() - 1).collect();
        match indices.len() {
          3 => {
            // Create a triangle from vertices
            let vertex1: Vertex3D = Vertex3D {position: vertices[indices[0].0].clone(), normal: normals[indices[0].2.unwrap()].clone(), color: 0xff_00_00};
            let vertex2: Vertex3D = Vertex3D {position: vertices[indices[1].0].clone(), normal: normals[indices[1].2.unwrap()].clone(), color: 0xff_00_00};
            let vertex3: Vertex3D = Vertex3D {position: vertices[indices[2].0].clone(), normal: normals[indices[2].2.unwrap()].clone(), color: 0xff_00_00};
            let triangle = Triangle3D::new(vertex1, vertex2, vertex3);
            triangles.push(triangle);
          },
          4 => {
            // Create two triangles from a quadrilateral face
            let vertex1: Vertex3D = Vertex3D {position: vertices[indices[0].0].clone(), normal: normals[indices[0].2.unwrap()].clone(), color: 0xff_00_00};
            let vertex2: Vertex3D = Vertex3D {position: vertices[indices[1].0].clone(), normal: normals[indices[1].2.unwrap()].clone(), color: 0xff_00_00};
            let vertex3: Vertex3D = Vertex3D {position: vertices[indices[2].0].clone(), normal: normals[indices[2].2.unwrap()].clone(), color: 0xff_00_00};
            let vertex4: Vertex3D = Vertex3D {position: vertices[indices[3].0].clone(), normal: normals[indices[3].2.unwrap()].clone(), color: 0xff_00_00};
            let triangle1 = Triangle3D::new(vertex1.clone(), vertex2.clone(), vertex3.clone());
            let triangle2 = Triangle3D::new(vertex3.clone(), vertex4.clone(), vertex1.clone());
            triangles.push(triangle1);
            triangles.push(triangle2);
          },
          _ => {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported face vertex count"));
          }
        }
      }
      _ => {}
    }
  }

  Ok(triangles)
}

fn read_off_file<P: AsRef<Path>>(path: P) -> io::Result<(Vec<Point3D>, Vec<Triangle3D> )> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();

    // Read the first line to confirm it's an OFF file
    let first_line = lines.next().unwrap()?;
    if first_line.trim() != "OFF" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not an OFF file"));
    }

    // Read the second line to get counts
    let second_line = lines.next().unwrap()?;
    let counts: Vec<usize> = second_line.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let vertex_count = counts[0];
    let face_count = counts[1];

    // Read vertices
    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let line = lines.next().unwrap()?;
        let coords: Vec<f32> = line.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        vertices.push(Point3D { x: coords[0], y: coords[1], z: coords[2] });
    }

    // Read faces (triangles)
    let mut triangles : Vec<Triangle3D> = Vec::with_capacity(face_count);
    for _ in 0..face_count {
        let line = lines.next().unwrap()?;
        let indices: Vec<usize> = line.split_whitespace()
            .skip(1) // Skip the first number which tells how many vertices in the face
            .map(|s| s.parse().unwrap())
            .collect();
        if indices.len() == 3 {
          let p1 = *vertices.get(indices[0]).unwrap();
          let p2 = *vertices.get(indices[1]).unwrap();
          let p3 = *vertices.get(indices[2]).unwrap();
          let edge1 = p2.subtract(&p1);
          let edge2 = p3.subtract(&p1);
          let normal = edge2.cross(&edge1).normalize();
          let v1 = Vertex3D {position: p1, normal, color: 0xff_00_00 };
          let v2 = Vertex3D {position: p2, normal, color: 0xff_00_00 };
          let v3 = Vertex3D {position: p3, normal, color: 0xff_00_00 };
          let triangle = Triangle3D::new(v1, v2, v3);
            triangles.push(triangle);
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Face is not a triangle"));
        }
    }

    Ok((vertices, triangles))
}

fn parse_vertex(line: &str) -> Option<Point3D> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() >= 4 {
        let x = tokens[1].parse().ok()?;
        let y = tokens[2].parse().ok()?;
        let z = tokens[3].parse().ok()?;
        Some(Point3D { x, y, z })
    } else {
        None
    }
}

fn parse_patch(line: &str) -> Option<BezierPatch> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() == 17 {
        let indices: Result<Vec<usize>, _> = tokens[1..].iter().map(|s| s.parse()).collect();
        if let Ok(indices) = indices {
            if indices.len() == 16 {
                let array: [usize; 16] = indices.try_into().unwrap(); // Safe because we checked length
                return Some(BezierPatch { indices: array });
            }
        }
    }
    None
}

fn evaluate_bezier_patch(control_points: &Vec<Point3D>, u: f32, v: f32) -> Point3D {
    assert!(control_points.len() == 16, "There must be exactly 16 control points.");

    let mut point = Point3D { x: 0.0, y: 0.0, z: 0.0 };

    for (i, &control_point) in control_points.iter().enumerate() {
        let ui = i / 4;  // Integer division to get row
        let vi = i % 4;  // Modulus to get column
        let bernstein_u = bernstein_polynomial(3, ui as u32, u);
        let bernstein_v = bernstein_polynomial(3, vi as u32, v);

        point.x += control_point.x * bernstein_u * bernstein_v;
        point.y += control_point.y * bernstein_u * bernstein_v;
        point.z += control_point.z * bernstein_u * bernstein_v;
    }

    point
}

fn bernstein_polynomial(n: u32, i: u32, t: f32) -> f32 {
    let binomial_coefficient = binomial(n, i) as f32;
    let t_i = t.powi(i as i32);
    let one_minus_t_n_minus_i = (1.0 - t).powi((n - i) as i32);

    binomial_coefficient * t_i * one_minus_t_n_minus_i
}

fn binomial(n: u32, k: u32) -> u32 {
    if k > n {
        return 0;
    }
    let mut result = 1;
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }
    result
}

fn tessellate_bezier_patch(control_points: &Vec<Point3D>, resolution: usize) -> Vec<Triangle3D> {
    let mut triangles = Vec::new();
    for i in 0..resolution {
        for j in 0..resolution {
            let u = i as f32 / resolution as f32;
            let v = j as f32 / resolution as f32;
            let step = 1.0 / resolution as f32;
            let p1 = evaluate_bezier_patch(control_points, u, v);
            let p2 = evaluate_bezier_patch(control_points, u + step, v);
            let p3 = evaluate_bezier_patch(control_points, u, v + step);
            let p4 = evaluate_bezier_patch(control_points, u + step, v + step);
            let edge1 = p2.subtract(&p1);
            let edge2 = p3.subtract(&p1);
            let normal = edge2.cross(&edge1).normalize();
            let v1 = Vertex3D {position: p1, normal: normal, color: 0xff_00_00};
            let v2 = Vertex3D {position: p2, normal: normal, color: 0xff_00_00};
            let v3 = Vertex3D {position: p3, normal: normal, color: 0xff_00_00};
            let v4 = Vertex3D {position: p4, normal: normal, color: 0xff_00_00};
            let triangle1 = Triangle3D::new(v1, v2, v3);
            let triangle2 = Triangle3D::new(v3, v2, v4);
            triangles.push(triangle1);
            triangles.push(triangle2);
        }
    }
    triangles
}
