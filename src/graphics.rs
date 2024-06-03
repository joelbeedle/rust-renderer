use crate::geometry::*;
use crate::math::barycentric_coords;
use crate::Camera;
use crate::{HEIGHT, WIDTH};

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

fn compute_lighting_all(
  vertex: &mut Vertex3D,
  light_position: &Point3D,
  camera_position: &Point3D,
) -> u32 {
  let normal = vertex.normal;
  let point_position = &vertex.position;
  let (r_a, g_a, b_a) = color_to_rgb_f32(0xff_00_00);
  let ambient_intensity = 0.2;
  let diffuse_intensity = 0.7;
  let specular_intensity = 0.9;
  let shininess = 128.0; // Shininess coefficient for specular highlights

  let light_direction = point_position.subtract(&light_position).normalize();
  let view_direction = vector_sub(camera_position, point_position).normalize();
  let reflect_direction = crate::math::reflect_vector(&(light_direction), &normal);

  let ambient = ambient_intensity;

  let diffuse = diffuse_intensity * normal.dot(&light_direction).max(0.0);

  let specular = if diffuse > 0.0 {
    specular_intensity
      * view_direction
        .dot(&reflect_direction)
        .max(0.0)
        .powf(shininess)
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
  let light_direction = point_position.subtract(&light_position).normalize();
  let distance = light_direction.magnitude();
  let attenuation = 1.0 / (1.0 + 0.01 * distance); // Linear attenuation

  let dot_product = normal.dot(&light_direction).max(0.05);
  let light_intensity = dot_product * attenuation;
  let brightness = (light_intensity * 255.0) as u8;
  let ambient_intensity = 0.7;
  let diffuse_intensity = 0.9;
  let r = ((r_a * (ambient_intensity + light_intensity * diffuse_intensity)) * 255.0) as u8;
  let g = ((g_a * (ambient_intensity + light_intensity * diffuse_intensity)) * 255.0) as u8;
  let b = ((b_a * (ambient_intensity + light_intensity * diffuse_intensity)) * 255.0) as u8;

  rgb_to_color(r, g, b)
}

fn compute_lighting_flat_linatten(
  normal: &mut Point3D,
  light_position: &Point3D,
  point_position: &Point3D,
) -> u32 {
  let (r_a, g_a, b_a) = color_to_rgb_f32(0xd1c7c0);
  let light_direction = point_position.subtract(&light_position).normalize();
  //let light_direction = light_position.subtract(&point_position).normalize();
  let distance = light_direction.magnitude();
  let attenuation = 1.0 / (1.0 + 0.01 * distance); // Linear attenuation

  let dot_product = normal.dot(&light_direction).max(0.05);
  let light_intensity = dot_product * attenuation;
  let brightness = (light_intensity * 255.0) as u8;
  let ambient_intensity = 0.1;
  let diffuse_intensity = 0.5;
  let r = ((r_a * (ambient_intensity + light_intensity * diffuse_intensity)) * 255.0) as u8;
  let g = ((g_a * (ambient_intensity + light_intensity * diffuse_intensity)) * 255.0) as u8;
  let b = ((b_a * (ambient_intensity + light_intensity * diffuse_intensity)) * 255.0) as u8;

  rgb_to_color(r, g, b)
}

fn compute_lighting(normal: &mut Point3D, light_dir: &Point3D, point_position: &Point3D) -> u32 {
  let light_direction = vector_sub(light_dir, &point_position);
  let light_intensity = normal.dot(&light_direction.normalize()).max(0.05).min(1.0); // Clamp to [0,1]
  let brightness = (light_intensity * 5.0 * 255.0) as u8;
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

pub fn average_depth(triangle: &Triangle3D, camera: &Camera) -> f32 {
  let p1 = camera.project(&triangle.v1.position, 1, 1);
  let p2 = camera.project(&triangle.v2.position, 1, 1);
  let p3 = camera.project(&triangle.v3.position, 1, 1);
  (p1.z + p2.z + p3.z) / 3.0
}

fn is_on_screen(point: &Point3D, width: usize, height: usize) -> bool {
  point.x >= 0.0 && point.x < width as f32 && point.y >= 0.0 && point.y < height as f32
}

pub fn draw_triangle3d(
  buffer: &mut Vec<u32>,
  width: usize,
  height: usize,
  triangle: &Triangle3D,
  camera: &Camera,
  color: u32,
) {
  let p1 = camera.project(&triangle.v1.position, width, height);
  let p2 = camera.project(&triangle.v2.position, width, height);
  let p3 = camera.project(&triangle.v3.position, width, height);

  // Check if all vertices are off-screen
  if !is_on_screen(&p1, width, height)
    && !is_on_screen(&p2, width, height)
    && !is_on_screen(&p3, width, height)
  {
    return; // Skip drawing this triangle entirely if all points are off-screen
  }

  let centroid = Point3D {
    x: (triangle.v1.position.x + triangle.v2.position.x + triangle.v3.position.x) / 3.0,
    y: (triangle.v1.position.y + triangle.v2.position.y + triangle.v3.position.y) / 3.0,
    z: (triangle.v1.position.z + triangle.v2.position.z + triangle.v3.position.z) / 3.0,
  };

  let view_direction = vector_sub(&centroid, &camera.position);

  if dot_product(&triangle.normal, &view_direction) > 0.0 {
    // Reverse condition to > 0.0 if normals are outward
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
  let v1 = (p1.x as isize, p1.y as isize);
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

    let (start, end) = if x_start < x_end {
      (x_start, x_end)
    } else {
      (x_end, x_start)
    };
    for x in start..=end {
      if x >= 0 && (x as usize) < WIDTH && y >= 0 && (y as usize) < HEIGHT {
        buffer[(y as usize) * WIDTH + (x as usize)] = color;
      }
      if x_start == x_end {
        break;
      }
    }
  }
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
  buffer: &mut Vec<u32>,
  width: usize,
  height: usize,
  p1: &Point3D,
  p2: &Point3D,
  p3: &Point3D,
  color1: u32,
  color2: u32,
  color3: u32,
) {
  let mut bbox_min = Point3D {
    x: width as f32 - 1.0,
    y: height as f32 - 1.0,
    z: 0.0,
  };
  let mut bbox_max = Point3D {
    x: 0.0,
    y: 0.0,
    z: 0.0,
  };
  let clamp = Point3D {
    x: (width as f32 - 1.0),
    y: (height as f32 - 1.0),
    z: 0.0,
  };

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
      let p = Point3D {
        x: x as f32,
        y: y as f32,
        z: 0.0,
      };
      let (u, v, w) = barycentric_coords(&p, p1, p2, p3);
      if u >= 0.0 && v >= 0.0 && w >= 0.0 {
        // The point is inside the triangle
        let idx = (y * width as i32 + x) as usize;
        buffer[idx] = interpolate_color_at(u, v, w, color1, color2, color3);
      }
    }
  }
}

pub fn fill_triangle3d(
  buffer: &mut Vec<u32>,
  width: usize,
  height: usize,
  triangle: &mut Triangle3D,
  camera: &Camera,
  color: u32,
) {
  let p1 = camera.project(&triangle.v1.position, width, height);
  let p2 = camera.project(&triangle.v2.position, width, height);
  let p3 = camera.project(&triangle.v3.position, width, height);

  // Check if all vertices are off-screen
  if !is_on_screen(&p1, width, height)
    && !is_on_screen(&p2, width, height)
    && !is_on_screen(&p3, width, height)
  {
    return; // Skip drawing this triangle entirely if all points are off-screen
  }
  let centroid = Point3D {
    x: (triangle.v1.position.x + triangle.v2.position.x + triangle.v3.position.x) / 3.0,
    y: (triangle.v1.position.y + triangle.v2.position.y + triangle.v3.position.y) / 3.0,
    z: (triangle.v1.position.z + triangle.v2.position.z + triangle.v3.position.z) / 3.0,
  };

  let light_position = Point3D {
    x: -23.0,
    y: 10.0,
    z: 40.0,
  };

  let light_pos2 = Point3D {
    x: 11.0,
    y: 16.0,
    z: 58.0,
  };

  let view_direction = vector_sub(&centroid, &camera.position);
  if dot_product(&triangle.normal, &view_direction) > 0.0 {
    // triangle.v1.color =
    //    compute_lighting_all(&mut triangle.v1, &light_position, &camera.position);
    //triangle.v2.color =
    //    compute_lighting_all(&mut triangle.v2, &light_position, &camera.position);
    //triangle.v3.color =
    //    compute_lighting_all(&mut triangle.v3, &light_position, &camera.position);

    let c = compute_lighting_flat_linatten(&mut triangle.normal, &camera.position, &centroid);

    if p1.z > 0.1 && p2.z > 0.1 && p3.z > 0.1 {
      fill_triangle(&p1, &p2, &p3, c as u32, buffer);
      //rasterize_filled_triangle(
      //    buffer,
      //    width,
      //    height,
      //    &p1,
      //    &p2,
      //    &p3,
      //    triangle.v1.color,
      //    triangle.v2.color,
      //    triangle.v3.color,
      //);
    }
  }
}
