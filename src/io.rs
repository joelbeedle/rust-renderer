use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use crate::geometry::*;

pub fn read_data<P: AsRef<Path>>(path: P) -> io::Result<(Vec<Point3D>, Vec<BezierPatch>)> {
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

pub fn read_obj_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<Triangle3D>> {
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
      }
      Some("vn") => {
        let x = parts.next().unwrap().parse::<f32>().unwrap();
        let y = parts.next().unwrap().parse::<f32>().unwrap();
        let z = parts.next().unwrap().parse::<f32>().unwrap();
        normals.push(Point3D { x, y, z });
      }
      Some("f") => {
        let mut indices: Vec<(usize, Option<usize>, Option<usize>)> = parts
          .map(|part| {
            let mut indices = part.split('/');
            let vertex_idx = indices.next().unwrap().parse::<usize>().unwrap() - 1;
            let tex_idx = indices
              .next()
              .and_then(|t| {
                if t.is_empty() {
                  None
                } else {
                  t.parse::<usize>().ok()
                }
              })
              .map(|t| t - 1);
            let norm_idx = indices
              .next()
              .and_then(|n| {
                if n.is_empty() {
                  None
                } else {
                  n.parse::<usize>().ok()
                }
              })
              .map(|n| n - 1);

            (vertex_idx, tex_idx, norm_idx)
          })
          .collect();

        if indices.iter().any(|(_, _, n)| n.is_none()) {
          let normal = compute_normal(
            &vertices[indices[0].0],
            &vertices[indices[1].0],
            &vertices[indices[2].0],
          );
          normals.push(normal);
          let normal_index = normals.len() - 1; // Index of the newly added normal

          // Update all indices to use this normal
          indices
            .iter_mut()
            .for_each(|&mut (_, _, ref mut n)| *n = Some(normal_index));
        }

        // let idx: Vec<usize> = parts.map(|part| part.split('/').next().unwrap().parse::<usize>().unwrap() - 1).collect();
        match indices.len() {
          3 => {
            // Triangles
            triangles.push(create_triangle(&vertices, &normals, &indices));
          }
          4 => {
            // Quadrilaterals
            triangles.push(create_triangle(
              &vertices,
              &normals,
              &[indices[0], indices[1], indices[2]],
            ));
            triangles.push(create_triangle(
              &vertices,
              &normals,
              &[indices[2], indices[3], indices[0]],
            ));
          }
          5 => {
            // Pentagons
            triangles.push(create_triangle(
              &vertices,
              &normals,
              &[indices[0], indices[1], indices[2]],
            ));
            triangles.push(create_triangle(
              &vertices,
              &normals,
              &[indices[0], indices[2], indices[3]],
            ));
            triangles.push(create_triangle(
              &vertices,
              &normals,
              &[indices[0], indices[3], indices[4]],
            ));
          }
          6 => {
            // Hexagons: Triangulate from the first vertex
            triangles.push(create_triangle(
              &vertices,
              &normals,
              &[indices[0], indices[1], indices[2]],
            ));
            triangles.push(create_triangle(
              &vertices,
              &normals,
              &[indices[0], indices[2], indices[3]],
            ));
            triangles.push(create_triangle(
              &vertices,
              &normals,
              &[indices[0], indices[3], indices[4]],
            ));
            triangles.push(create_triangle(
              &vertices,
              &normals,
              &[indices[0], indices[4], indices[5]],
            ));
          }
          _ => {
            return Err(io::Error::new(
              io::ErrorKind::InvalidData,
              format!("Unsupported face vertex count: {}", indices.len()),
            ));
          }
        }
      }
      _ => {}
    }
  }

  Ok(triangles)
}

pub fn read_off_file<P: AsRef<Path>>(path: P) -> io::Result<(Vec<Point3D>, Vec<Triangle3D>)> {
  let file = File::open(path)?;
  let reader = io::BufReader::new(file);
  let mut lines = reader.lines();

  // Read the first line to confirm it's an OFF file
  let first_line = lines.next().unwrap()?;
  if first_line.trim() != "OFF" {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      "Not an OFF file",
    ));
  }

  // Read the second line to get counts
  let second_line = lines.next().unwrap()?;
  let counts: Vec<usize> = second_line
    .split_whitespace()
    .map(|s| s.parse().unwrap())
    .collect();
  let vertex_count = counts[0];
  let face_count = counts[1];

  // Read vertices
  let mut vertices = Vec::with_capacity(vertex_count);
  for _ in 0..vertex_count {
    let line = lines.next().unwrap()?;
    let coords: Vec<f32> = line
      .split_whitespace()
      .map(|s| s.parse().unwrap())
      .collect();
    vertices.push(Point3D {
      x: coords[0],
      y: coords[1],
      z: coords[2],
    });
  }

  // Read faces (triangles)
  let mut triangles: Vec<Triangle3D> = Vec::with_capacity(face_count);
  for _ in 0..face_count {
    let line = lines.next().unwrap()?;
    let indices: Vec<usize> = line
      .split_whitespace()
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
      let v1 = Vertex3D {
        position: p1,
        normal,
        color: 0xff_00_00,
      };
      let v2 = Vertex3D {
        position: p2,
        normal,
        color: 0xff_00_00,
      };
      let v3 = Vertex3D {
        position: p3,
        normal,
        color: 0xff_00_00,
      };
      let triangle = Triangle3D::new(v1, v2, v3);
      triangles.push(triangle);
    } else {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "Face is not a triangle",
      ));
    }
  }

  Ok((vertices, triangles))
}

pub fn parse_vertex(line: &str) -> Option<Point3D> {
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

pub fn parse_patch(line: &str) -> Option<BezierPatch> {
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
  assert!(
    control_points.len() == 16,
    "There must be exactly 16 control points."
  );

  let mut point = Point3D {
    x: 0.0,
    y: 0.0,
    z: 0.0,
  };

  for (i, &control_point) in control_points.iter().enumerate() {
    let ui = i / 4; // Integer division to get row
    let vi = i % 4; // Modulus to get column
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

pub fn tessellate_bezier_patch(
  control_points: &Vec<Point3D>,
  resolution: usize,
) -> Vec<Triangle3D> {
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
      let v1 = Vertex3D {
        position: p1,
        normal: normal,
        color: 0xff_00_00,
      };
      let v2 = Vertex3D {
        position: p2,
        normal: normal,
        color: 0xff_00_00,
      };
      let v3 = Vertex3D {
        position: p3,
        normal: normal,
        color: 0xff_00_00,
      };
      let v4 = Vertex3D {
        position: p4,
        normal: normal,
        color: 0xff_00_00,
      };
      let triangle1 = Triangle3D::new(v1, v2, v3);
      let triangle2 = Triangle3D::new(v3, v2, v4);
      triangles.push(triangle1);
      triangles.push(triangle2);
    }
  }
  triangles
}
