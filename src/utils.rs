use glam::{Vec2, vec2};

use std::f32::consts::PI;

use rand::prelude::*;
use rand::distr::Uniform;

pub fn generate_perturbed_grid(num_per_side: usize) -> Vec<Vec2> {
    (0..num_per_side).flat_map(|x| {
        (0..num_per_side).map(move |y| { 
            //generate 0 < tx < 1, 0 < ty < 1 based on x, y
            let (tx, ty) = (x as f32 / ((num_per_side) as f32), y as f32 / ((num_per_side) as f32));
            return vec2((tx - 0.5) * 6.0 + 3.0 / num_per_side as f32, (ty - 0.5) * 6.0 + 3.0 / num_per_side as f32) 
                + vec2(
                    ((tx * 238.4).cos() * (ty * 328.8483).sin() * 2.0 * PI).cos(), 
                    ((tx * 238.4).cos() * (ty * 328.8483).sin() * 2.0 * PI).sin()
                ) * 1.0 / num_per_side as f32;
        })
    }).collect()
}

pub fn generate_points_white_noise(num_points: usize) -> Vec<Vec2> {
    let mut sites: Vec<glam::Vec2> = vec![];

    let mut rng = rand::rng();
    let distr = Uniform::new(-3.0f32, 3.0f32).unwrap();

    for _i in 0..num_points {
        sites.push(vec2(rng.sample(&distr), rng.sample(&distr)))
    }

    sites
}