use epson::voronoi_2d::*;
use epson::utils::{generate_perturbed_grid, generate_points_white_noise};

use glam::vec2;

use std::f32::consts::PI;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

pub fn test_perturbed_grids(c: &mut Criterion) {
    for i in 5..9 {
        test_perturbed_grid(c, (2usize).pow(i));
    }
}

fn test_perturbed_grid(c: &mut Criterion, num_per_side: usize) {
    let sites = generate_perturbed_grid(num_per_side);

    c.bench_with_input(BenchmarkId::new("Voronoi from perturbed grid", num_per_side), &sites, move |b, s| {
        b.iter(|| voronoi(s, vec2(11.0, 11.0)));
    });
}

criterion_group!(benches, test_perturbed_grids);
criterion_main!(benches);

