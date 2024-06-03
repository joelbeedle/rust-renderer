use crate::geometry::*;

pub fn reflect_vector(incident: &Point3D, normal: &Point3D) -> Point3D {
  // Compute dot product of incident vector and normal
  let dot_product = incident.dot(normal);

  // R = I - 2*(I·N)*N
  let reflected = incident.subtract(&normal.multiply_scalar(2.0 * dot_product));
  reflected
}

pub fn rotate_x(angle: f32) -> [[f32; 3]; 3] {
  let cos_a = angle.cos();
  let sin_a = angle.sin();
  [[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]]
}

pub fn rotate_y(angle: f32) -> [[f32; 3]; 3] {
  let cos_a = angle.cos();
  let sin_a = angle.sin();
  [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]]
}

pub fn apply_matrix(point: Point3D, matrix: [[f32; 3]; 3]) -> Point3D {
  Point3D {
    x: point.x * matrix[0][0] + point.y * matrix[0][1] + point.z * matrix[0][2],
    y: point.x * matrix[1][0] + point.y * matrix[1][1] + point.z * matrix[1][2],
    z: point.x * matrix[2][0] + point.y * matrix[2][1] + point.z * matrix[2][2],
  }
}

/// Calculate the barycentric coordinates of point p with respect to triangle (a, b, c)
pub fn barycentric_coords(p: &Point3D, a: &Point3D, b: &Point3D, c: &Point3D) -> (f32, f32, f32) {
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
