use epson::voronoi_pga3d::voronoi_3d;
use glam::vec3;

fn main() {
    let sites = vec![vec3(0.001, 0.0, 1.0), vec3(0.0, 0.001, -1.0), vec3(0.001, 1.0, 0.0), vec3(0.001, -1.0, 0.0), vec3(1.0, 0.0, 0.001), vec3(-1.0, 0.0, 0.0)];

    let v = voronoi_3d(&sites, vec3(2.0, 2.0, 2.0));

    println!("{:?}", v.cells[0].neighbor_indices);
    println!("{:?}", v.cells[1].neighbor_indices);
    println!("{:?}", v.cells[2].neighbor_indices);
    println!("{:?}", v.cells[3].neighbor_indices);
    println!("{:?}", v.cells[4].neighbor_indices);
    println!("{:?}", v.cells[5].neighbor_indices);
}