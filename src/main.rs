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

#[derive(Debug)]
struct Triangle3D {
  p1: Point3D,
  p2: Point3D,
  p3: Point3D,
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
  
  let mut camera = Camera::new(Point3D { x: 0.0, y: 0.0, z: -5.0 }, 90.0);

  let cube = Cube::new(Point3D {x: 0.0, y: 0.0, z: 0.0});
  let cube2 = Cube::new(Point3D {x: 2.0, y: 0.0, z: 0.0});
  let cube3 = Cube::new(Point3D {x: 4.0, y: 0.0, z: 0.0});
  let cube4 = Cube::new(Point3D {x: 0.0, y: 2.0, z: 0.0});
  let cubes: Vec<Cube> = vec![cube, cube2, cube3, cube4];
  // let mut triangles = all_cube_faces(&cubes);
  let mut triangles = vec![];
  let data = read_data("teapot_data.txt").unwrap();
  let indices: Vec<Point3D> = data.0;
  let bezier_patches: Vec<BezierPatch> = data.1;
  let mut tri: Vec<Vec<Triangle3D>> = Vec::new(); 
  for (i, val) in indices.iter().enumerate() {
    dbg!(&i);
    dbg!(&val);
  }
  for patch in bezier_patches {
    dbg!(&patch);
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
    // dbg!(camera.position);
    buffer.iter_mut().for_each(|p| *p = 0);
    triangles.sort_by(|a, b| {
        let depth_a = average_depth(a, &camera);
        let depth_b = average_depth(b, &camera);
        depth_b.partial_cmp(&depth_a).unwrap_or(std::cmp::Ordering::Equal)
    });
    for triangle in &triangles {
      fill_triangle3d(&mut buffer, WIDTH, HEIGHT, triangle, &camera, 0xFF0000); // Fill triangle with red
      // draw_triangle3d(&mut buffer, WIDTH, HEIGHT, triangle, &camera, 0xffffff);
    }

    camera.update_vectors();
    window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
  }
}

fn average_depth(triangle: &Triangle3D, camera: &Camera) -> f32 {
  let p1 = camera.project(&triangle.p1, 1, 1);
  let p2 = camera.project(&triangle.p2, 1, 1);
  let p3 = camera.project(&triangle.p3, 1, 1);
  (p1.z + p2.z + p3.z) / 3.0
}

fn all_cube_faces(cubes: &[Cube]) -> Vec<Triangle3D> {
    let mut all_triangles = Vec::new();

    for cube in cubes {
        let triangles = cube_faces(cube);
        all_triangles.extend(triangles);
    }

    all_triangles
}

fn cube_faces(cube: &Cube) -> Vec<Triangle3D> {
  vec![
    // Front face
    Triangle3D { p1: cube.vertices[0], p2: cube.vertices[1], p3: cube.vertices[2] },
    Triangle3D { p1: cube.vertices[1], p2: cube.vertices[3], p3: cube.vertices[2] },
    // Right face
    Triangle3D { p1: cube.vertices[1], p2: cube.vertices[5], p3: cube.vertices[3] },
    Triangle3D { p1: cube.vertices[5], p2: cube.vertices[7], p3: cube.vertices[3] },
    // Back face
    Triangle3D { p1: cube.vertices[5], p2: cube.vertices[4], p3: cube.vertices[7] },
    Triangle3D { p1: cube.vertices[4], p2: cube.vertices[6], p3: cube.vertices[7] },
    // Left face
    Triangle3D { p1: cube.vertices[4], p2: cube.vertices[0], p3: cube.vertices[6] },
    Triangle3D { p1: cube.vertices[0], p2: cube.vertices[2], p3: cube.vertices[6] },
    // Top face
    Triangle3D { p1: cube.vertices[4], p2: cube.vertices[5], p3: cube.vertices[0] },
    Triangle3D { p1: cube.vertices[5], p2: cube.vertices[1], p3: cube.vertices[0] },
    // Bottom face
    Triangle3D { p1: cube.vertices[2], p2: cube.vertices[3], p3: cube.vertices[6] },
    Triangle3D { p1: cube.vertices[3], p2: cube.vertices[7], p3: cube.vertices[6] },
  ]
}

fn draw_triangle3d(buffer: &mut Vec<u32>, width: usize, height: usize, triangle: &Triangle3D, camera: &Camera, color: u32) {
    let p1 = camera.project(&triangle.p1, width, height);
    let p2 = camera.project(&triangle.p2, width, height);
    let p3 = camera.project(&triangle.p3, width, height);


    let edge1 = vector_sub(&triangle.p1, &triangle.p2);
    let edge2 = vector_sub(&triangle.p1, &triangle.p3);
    let normal = cross_product(&edge1, &edge2);

    let centroid = Point3D {
        x: (triangle.p1.x + triangle.p2.x + triangle.p3.x) / 3.0,
        y: (triangle.p1.y + triangle.p2.y + triangle.p3.y) / 3.0,
        z: (triangle.p1.z + triangle.p2.z + triangle.p3.z) / 3.0,
    };

    let view_direction = vector_sub(&centroid, &camera.position);

    if dot_product(&normal, &view_direction) > 0.0 {  // Reverse condition to > 0.0 if normals are outward
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

fn fill_triangle3d(buffer: &mut Vec<u32>, width: usize, height: usize, triangle: &Triangle3D, camera: &Camera, color: u32) {
    let p1 = camera.project(&triangle.p1, width, height);
    let p2 = camera.project(&triangle.p2, width, height);
    let p3 = camera.project(&triangle.p3, width, height);

    let edge1 = vector_sub(&triangle.p1, &triangle.p2);
    let edge2 = vector_sub(&triangle.p1, &triangle.p3);
    let normal = cross_product(&edge1, &edge2);

    let centroid = Point3D {
        x: (triangle.p1.x + triangle.p2.x + triangle.p3.x) / 3.0,
        y: (triangle.p1.y + triangle.p2.y + triangle.p3.y) / 3.0,
        z: (triangle.p1.z + triangle.p2.z + triangle.p3.z) / 3.0,
    };

    let view_direction = vector_sub(&centroid, &camera.position);

    if dot_product(&normal, &view_direction) > 0.0 {  // Reverse condition to > 0.0 if normals are outward
      if p1.z > 0.1 && p2.z > 0.1 && p3.z > 0.1 {
        fill_triangle(&p1, &p2, &p3, color, buffer);
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
            triangles.push(Triangle3D { p1, p2, p3 });
            triangles.push(Triangle3D { p2, p3: p4, p1: p3 });
        }
    }
    triangles
}
