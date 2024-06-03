#[derive(Copy, Clone, Debug)]
pub struct Point3D {
  pub x: f32,
  pub y: f32,
  pub z: f32,
}

impl Point3D {
  pub fn subtract(&self, other: &Point3D) -> Point3D {
    Point3D {
      x: self.x - other.x,
      y: self.y - other.y,
      z: self.z - other.z,
    }
  }

  pub fn cross(&self, other: &Point3D) -> Point3D {
    Point3D {
      x: self.y * other.z - self.z * other.y,
      y: self.z * other.x - self.x * other.z,
      z: self.x * other.y - self.y * other.x,
    }
  }

  pub fn normalize(&self) -> Point3D {
    let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
    Point3D {
      x: self.x / len,
      y: self.y / len,
      z: self.z / len,
    }
  }

  pub fn multiply_scalar(&self, scalar: f32) -> Point3D {
    Point3D {
      x: self.x * scalar,
      y: self.y * scalar,
      z: self.z * scalar,
    }
  }

  pub fn dot(&self, other: &Point3D) -> f32 {
    self.x * other.x + self.y * other.y + self.z * other.z
  }

  pub fn magnitude(&self) -> f32 {
    (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
  }
}

#[derive(Copy, Clone, Debug)]
pub struct Vertex3D {
  pub position: Point3D,
  pub normal: Point3D,
  pub color: u32,
}

#[derive(Debug)]
pub struct Triangle3D {
  pub v1: Vertex3D,
  pub v2: Vertex3D,
  pub v3: Vertex3D,
  pub normal: Point3D,
}

impl Triangle3D {
  pub fn new(v1: Vertex3D, v2: Vertex3D, v3: Vertex3D) -> Triangle3D {
    let edge1 = v2.position.subtract(&v1.position);
    let edge2 = v3.position.subtract(&v1.position);
    let normal = edge2.cross(&edge1).normalize();
    Triangle3D { v1, v2, v3, normal }
  }
}

pub fn create_triangle(
  vertices: &Vec<Point3D>,
  normals: &Vec<Point3D>,
  indices: &[(usize, Option<usize>, Option<usize>)],
) -> Triangle3D {
  Triangle3D::new(
    Vertex3D {
      position: vertices[indices[0].0].clone(),
      normal: normals[indices[0].2.unwrap()].clone(),
      color: 0xff_00_00,
    },
    Vertex3D {
      position: vertices[indices[1].0].clone(),
      normal: normals[indices[1].2.unwrap()].clone(),
      color: 0xff_00_00,
    },
    Vertex3D {
      position: vertices[indices[2].0].clone(),
      normal: normals[indices[2].2.unwrap()].clone(),
      color: 0xff_00_00,
    },
  )
}

#[derive(Debug)]
pub struct BezierPatch {
  pub indices: [usize; 16],
}

pub struct Cube {
  pub vertices: [Point3D; 8],
}

impl Cube {
  pub fn new(rel_pos: Point3D) -> Self {
    Cube {
      vertices: [
        Point3D {
          x: -1.0 + rel_pos.x,
          y: 1.0 + rel_pos.y,
          z: -1.0 + rel_pos.z,
        }, // Front top left
        Point3D {
          x: 1.0 + rel_pos.x,
          y: 1.0 + rel_pos.y,
          z: -1.0 + rel_pos.z,
        }, // Front top right
        Point3D {
          x: -1.0 + rel_pos.x,
          y: -1.0 + rel_pos.y,
          z: -1.0 + rel_pos.z,
        }, // Front bottom left
        Point3D {
          x: 1.0 + rel_pos.x,
          y: -1.0 + rel_pos.y,
          z: -1.0 + rel_pos.z,
        }, // Front bottom right
        Point3D {
          x: -1.0 + rel_pos.x,
          y: 1.0 + rel_pos.y,
          z: 1.0 + rel_pos.z,
        }, // Back top left
        Point3D {
          x: 1.0 + rel_pos.x,
          y: 1.0 + rel_pos.y,
          z: 1.0 + rel_pos.z,
        }, // Back top right
        Point3D {
          x: -1.0 + rel_pos.x,
          y: -1.0 + rel_pos.y,
          z: 1.0 + rel_pos.z,
        }, // Back bottom left
        Point3D {
          x: 1.0 + rel_pos.x,
          y: -1.0 + rel_pos.y,
          z: 1.0 + rel_pos.z,
        }, // Back bottom right
      ],
    }
  }
}

// Vector subtraction
pub fn vector_sub(a: &Point3D, b: &Point3D) -> Point3D {
  Point3D {
    x: a.x - b.x,
    y: a.y - b.y,
    z: a.z - b.z,
  }
}

// Cross product
pub fn cross_product(a: &Point3D, b: &Point3D) -> Point3D {
  Point3D {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  }
}

// Dot product
pub fn dot_product(a: &Point3D, b: &Point3D) -> f32 {
  a.x * b.x + a.y * b.y + a.z * b.z
}

pub fn compute_normal(p1: &Point3D, p2: &Point3D, p3: &Point3D) -> Point3D {
  let u = Point3D {
    x: p2.x - p1.x,
    y: p2.y - p1.y,
    z: p2.z - p1.z,
  };
  let v = Point3D {
    x: p3.x - p1.x,
    y: p3.y - p1.y,
    z: p3.z - p1.z,
  };
  u.cross(&v).normalize()
}
