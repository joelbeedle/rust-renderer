use crate::geometry::*;
use crate::math::{apply_matrix, rotate_x, rotate_y};
#[derive(Debug)]
pub struct Camera {
  pub position: Point3D,
  pub forward: Point3D, // Direction camera is facing
  pub right: Point3D,   // Right vector
  pub up: Point3D,      // Up vector
  pub fov: f32,
  pub pitch: f32,
  pub yaw: f32,
}

impl Camera {
  pub fn project(&self, point: &Point3D, width: usize, height: usize) -> Point3D {
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
    let screen_x = (rotated_point.x / (rotated_point.z * scale * aspect_ratio)) * width as f32
      / 2.0
      + width as f32 / 2.0;
    let screen_y =
      -(rotated_point.y / (rotated_point.z * scale)) * height as f32 / 2.0 + height as f32 / 2.0;

    Point3D {
      x: screen_x,
      y: screen_y,
      z: rotated_point.z,
    }
  }

  pub fn new(position: Point3D, fov: f32) -> Self {
    Camera {
      position,
      forward: Point3D {
        x: 0.0,
        y: 0.0,
        z: -1.0,
      },
      right: Point3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
      },
      up: Point3D {
        x: 0.0,
        y: 1.0,
        z: 0.0,
      },
      fov,
      yaw: 0.0,
      pitch: 0.0,
    }
  }

  pub fn update_vectors(&mut self) {
    // Calculate the new forward vector
    // Calculate the new forward vector based on yaw (rotation around the y-axis)
    self.forward = Point3D {
      x: self.pitch.to_radians().cos() * self.yaw.to_radians().cos(),
      y: self.pitch.to_radians().sin(),
      z: self.pitch.to_radians().cos() * self.yaw.to_radians().sin(),
    };

    // Normalize the forward vector
    let forward_len =
      (self.forward.x.powi(2) + self.forward.y.powi(2) + self.forward.z.powi(2)).sqrt();
    self.forward = Point3D {
      x: self.forward.x / forward_len,
      y: self.forward.y / forward_len,
      z: self.forward.z / forward_len,
    };

    // Calculate the right vector as the cross product of the world up vector and forward vector
    let world_up = Point3D {
      x: 0.0,
      y: 1.0,
      z: 0.0,
    };
    self.right = cross_product(&self.forward, &world_up);

    // Normalize the right vector
    let right_len = (self.right.x.powi(2) + self.right.y.powi(2) + self.right.z.powi(2)).sqrt();
    self.right = Point3D {
      x: self.right.x / right_len,
      y: self.right.y / right_len,
      z: self.right.z / right_len,
    };

    // The up vector should be recalculated even if not used directly
    self.up = cross_product(&self.right, &self.forward);
  }
}
