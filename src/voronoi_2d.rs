use glam::{Vec2, vec2, mat2, swizzles::Vec2Swizzles };
use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;

const EPSILON: f32 = 0.00001;
use std::collections::{VecDeque, HashSet};

use rayon::prelude::*;

type Accelerator = ImmutableKdTree<f32, usize, 2, 64>;

pub struct CellInfo {
    pub wedges: Vec<(Vec2, Vec2)>
}

pub struct Cell {
    pub site: Vec2,
    pub neighbor_indices: Vec<usize>,

    pub info: Option<CellInfo>
}

impl Cell {
    pub fn get_wedges(&self) -> Option<&Vec<(Vec2, Vec2)>> {
        self.info.as_ref().map(|i| &i.wedges)
    }
}

pub struct Voronoi {
    pub cells: Vec<Cell>,
    bounds: Vec2,

    kdtree: Accelerator
}

pub fn voronoi(sites: &Vec<Vec2>, bounds: Vec2) -> Voronoi {
    let points: Vec<_> = sites.iter().map(|p| [p.x, p.y]).collect();
    let kdtree: Accelerator = Accelerator::from(&points[..]);
    drop(points);
    
    let mut result = Voronoi {
        cells: sites.iter().map(|p| Cell { site: *p, neighbor_indices: vec![], info: None }).collect(),
        bounds,
        kdtree,
    };

    result.calculate_all_cell_neighbors_parallel();

    result
}

struct RayTraceResult {
    intersection: Vec2,
    normal: Vec2,
    cell_idx: Option<usize>
}

struct Projector {
    r1: Vec2,
    r2: Vec2
}

impl Voronoi {
    fn calculate_all_cell_neighbors(&mut self) {
        for i in 0..self.cells.len() { 
            let (neighbors, wedges) = self.calculate_cell_neighbors(i); 
            self.cells[i].neighbor_indices = neighbors;
            self.cells[i].info = Some(CellInfo { wedges });
        }
    }

    fn calculate_all_cell_neighbors_parallel(&mut self) {
        let results: Vec<_> = (0..self.cells.len()).into_par_iter().map(|i| {
            self.calculate_cell_neighbors(i)
        }).collect();
        for (i, r) in results.into_iter().enumerate() {
            self.cells[i].neighbor_indices = r.0;
            self.cells[i].info = Some(CellInfo { wedges: r.1 });
        }
    }

    fn calculate_cell_neighbors(&self, cell_idx: usize) -> (Vec<usize>, Vec<(Vec2, Vec2)>){
        let mut neighbors: HashSet<usize> = HashSet::new();
        let mut wedges: Vec<(Vec2, Vec2)> = vec![];
        let cell = &self.cells[cell_idx];

        //initialize projector queue with 3 projectors (with rays out along the vertices of an equilateral triangle)
        let mut projectors = VecDeque::from([
            Projector { r1: vec2(0.0, -1.0), r2: vec2(-0.86603, 0.5) },
            Projector { r1: vec2(-0.86603, 0.5), r2: vec2(0.86603, 0.5) },
            Projector { r1: vec2(0.86603, 0.5), r2: vec2(0.0, -1.0) }
        ]);
        // DEBUG: 4 initial projectors, one for each quadrant
        // let mut projectors = VecDeque::from([
        //     Projector { r1: vec2(0.0, 1.0), r2: vec2(1.0, 0.0) },
        //     Projector { r1: vec2(0.0, 1.0), r2: vec2(-1.0, 0.0) },
        //     Projector { r1: vec2(0.0, -1.0), r2: vec2(1.0, 0.0) },
        //     Projector { r1: vec2(0.0, -1.0), r2: vec2(-1.0, 0.0) },
        // ]);

        while !projectors.is_empty() {
            let projector = projectors.pop_front().unwrap();
            // println!("Processing projector {:} : {:}", projector.r1, projector.r2);

            let r1_hit = self.nearest_cell_along_ray(cell.site, projector.r1);
            let r2_hit = self.nearest_cell_along_ray(cell.site, projector.r2);
            
            let line_intersection_system = mat2(r1_hit.normal, r2_hit.normal).transpose();    
            
            if line_intersection_system.determinant().abs() < EPSILON {
                //the lines do not intersect, so they are parallel
                if r1_hit.cell_idx == r2_hit.cell_idx && r1_hit.normal.dot(r2_hit.normal) > 0.0 {
                    //both rays hit the same line, so the cell on the other side must be a neighbor
                    if let Some(neighbor_idx) = r1_hit.cell_idx {
                        neighbors.insert(neighbor_idx);
                        // println!("Found neighbor {:}", neighbor_idx);
                    }
                    wedges.push((r1_hit.intersection, r2_hit.intersection));
                    continue;
                } else {
                    //both rays hit different cells
                    //we want to split this projector in two by a ray pointing in the same direction as the lines
                    let line_direction =  {
                        let t = r1_hit.normal.perp();
                        //the line has two valid tangent vectors, choose the one which is in the same direction as the projector
                        if t.dot(projector.r1) > 0.0 { t } else { -t }
                    };
                    
                    // println!("Splitting projector into {:} : {:} : {:}", projector.r1, line_direction, projector.r2);
                    projectors.push_back(Projector { r1: projector.r1, r2: line_direction });
                    projectors.push_back(Projector { r1: line_direction, r2: projector.r2 });
                }
            } else {
                //the lines intersect, so we should find that intersection and split the wedge with a ray in that direction
                //we want to get the intersection of the two bisectors as a linear combination of the projectors's rays, so we will get them in a basis matrix
                let basis = mat2(projector.r1, projector.r2);
                let m_inverse = (line_intersection_system * basis).inverse();
                //find the dot products for points on the lines (relative to the cell site) and the line normals
                let b = vec2(
                    (r1_hit.intersection - cell.site).dot(r1_hit.normal),
                    (r2_hit.intersection - cell.site).dot(r2_hit.normal)
                );
                //the coefficients of the linear combination will be the inverse matrix * those dot products
                let coeffs = m_inverse * b;
                //we can get the intersection by multiplying by the basis matrix (of rays) and adding the site position
                let intersection = basis * coeffs + cell.site;
                if coeffs.x > -EPSILON && coeffs.y > -EPSILON { //if both coefficients are positive, the intersection is in the projector
                    let intersection_nearest = self.kdtree.nearest_one::<SquaredEuclidean>(&intersection.into());
                    let (intersection_nearest_dist, intersection_nearest_cell) = (intersection_nearest.distance, intersection_nearest.item as usize);
                    if (intersection_nearest_cell == cell_idx || intersection_nearest_dist > cell.site.distance_squared(intersection) - 0.001) 
                        && intersection.x.abs() < self.bounds.x + EPSILON && intersection.y.abs() < self.bounds.y + EPSILON
                    {
                        //the intersection is nearest to this cell, so we've successfully found a vertex and can stop searching his projector
                        if let Some(neighbor) = r1_hit.cell_idx { neighbors.insert(neighbor); /* println!("Found neighbor {:}", neighbor); */ }
                        if let Some(neighbor) = r2_hit.cell_idx { neighbors.insert(neighbor); /* println!("Found neighbor {:}", neighbor); */ }
                        wedges.push((r1_hit.intersection, intersection));
                        wedges.push((intersection, r2_hit.intersection));
                        continue;
                    }
                }
                //either the intersection is not in the cell or the intersection is not in the projector--either way we need to split
                //this projector in two by a ray in the direction of the intersection (or opposite it)
                let ray = {
                    let t = (intersection - cell.site).normalize();
                    if coeffs.x > -EPSILON && coeffs.y > -EPSILON { t } else { -t }
                };

                // println!("Splitting projector into {:} : {:} : {:}", projector.r1, ray, projector.r2);
                projectors.push_back(Projector { r1: projector.r1, r2: ray });
                projectors.push_back(Projector { r1: ray, r2: projector.r2 });
            }
        }

        (neighbors.into_iter().collect(), wedges)
    }

    
    //There is a line bisecting space between each pair of cells
    //this function finds the nearest intersection between a ray and one of those lines,
    //and returns the location of that intersection along with the index of the neighboring cell
    //(specifically, it will return the nearest neighbor to the cell that ray_origin lies in along
    //the direction ray_dir)
    fn nearest_cell_along_ray(&self,
        ray_origin: Vec2, ray_direction: Vec2
    ) -> RayTraceResult {    
        assert!(ray_origin.abs().x < self.bounds.x && ray_origin.abs().y < self.bounds.y);
        assert!((ray_direction.length() - 1.0).abs() < EPSILON);

        let initial_cell_idx = self.kdtree.nearest_one::<SquaredEuclidean>(&ray_origin.into()).item;
        let initial_cell = &self.cells[initial_cell_idx as usize];

        let (initial_intersection, initial_intersection_normal) = {
            //the bounding box is a rectangle with side lengths 2 * bounds.xy, centered at the origin
            //since the ray_origin must be inside the bounding box, the rays will intersect with the
            //bounding edge with the sign of the ray along each axis
            let bounding_line = self.bounds.copysign(ray_direction);
            
            //find the intersection distances for each axis
            let t = (bounding_line - ray_origin) / ray_direction;
            
            //return the closest of the intersections
            //create a mask {x < y, y < x};
            let mask = t.xy().cmplt(t.yx());
            //the intersection will be the along the ray with the minimum 't' value--the normal of that intersection will be a unit vector along the axis
            //the intersection occurs along (corresponding to the minimum 't'), in the opposite direction of the ray
            (ray_origin + ray_direction * t.min_element(), Vec2::select(mask, -ray_direction.signum(), Vec2::ZERO))
        };
        
        //before beginning the intersection finding, we need to find another cell that lies along the ray (or is near to a point along it)
        //For optimal searching, that cell should be fairly close to the original cell--but if the original cell is large, that could still be a good distance
        //This controls the number of tested points between the ray origin and the boundary (spaced "quadratically" far from the ray origin)
        const NUM_INITIAL_TESTS: usize = 4;

        let mut nearest_cell_to_intersection_idx = initial_cell_idx;
        for i in 1..=NUM_INITIAL_TESTS {
            let ii = i * i;
            let max = NUM_INITIAL_TESTS * NUM_INITIAL_TESTS;
            let t = ii as f32 / max as f32;
            
            nearest_cell_to_intersection_idx = self.kdtree.nearest_one::<SquaredEuclidean>(&(initial_cell.site * (1.0 - t) + initial_intersection * t).into()).item;
            if nearest_cell_to_intersection_idx != initial_cell_idx { break; }
        }
        
        while nearest_cell_to_intersection_idx != initial_cell_idx {
            let nearest_cell = &self.cells[nearest_cell_to_intersection_idx as usize];
            
            //get the vector pointing from nearest_cell to initial_cell
            let directed_difference = initial_cell.site - nearest_cell.site;
            let midpoint = nearest_cell.site + directed_difference / 2.0;
            let bisector_normal = directed_difference.normalize();

            if let Some(dist) = ray_line_intersection_dist(ray_origin, ray_direction, midpoint, bisector_normal) {
                let intersection = ray_origin + ray_direction * dist;
                let new_nearest_cell_to_intersection_idx = self.kdtree.nearest_one::<SquaredEuclidean>(&intersection.into()).item;

                if new_nearest_cell_to_intersection_idx == nearest_cell_to_intersection_idx ||
                    new_nearest_cell_to_intersection_idx == initial_cell_idx {
                    //we've found the same nearest cell twice or made it back to the initial cell, so the bisector
                    //must be the closest one along the ray. Return the non-initial cell & intersection
                    return RayTraceResult { intersection, normal: bisector_normal, cell_idx: Some(nearest_cell_to_intersection_idx as usize) };
                } else {
                    nearest_cell_to_intersection_idx = new_nearest_cell_to_intersection_idx;
                }
            } else { 
                //there was no intersection between the ray and the bisector of the two cells,
                //so we should stop searching and return the bounding intersection 
                break; 
            }
        }

        RayTraceResult { intersection: initial_intersection, normal: initial_intersection_normal, cell_idx: None } 
    }
}

fn ray_line_intersection_dist(
    ray_origin: Vec2, ray_direction: Vec2,
    line_origin: Vec2, line_normal: Vec2 
) -> Option<f32> {
    if ray_direction.dot(line_normal).abs() < EPSILON { return None; } 
    
    Some((line_origin - ray_origin).dot(line_normal) / (ray_direction).dot(line_normal))
}

#[test]
fn test_nearest_cell_along_ray_hits() {
    let voronoi = Voronoi {
        cells: vec![
            Cell { site: vec2(-1.0, 0.0), neighbor_indices: vec![1], info: None },
            Cell { site: vec2(1.0, 0.0), neighbor_indices: vec![0], info: None }
        ],
        bounds: vec2(3.0, 3.0),
        kdtree: Accelerator::from(&vec![[-1.0, 0.0], [1.0, 0.0]][..])
    };
    
    let ray_miss = voronoi.nearest_cell_along_ray(vec2(1.0, 0.0), vec2(1.0, 0.0));
    assert_eq!(ray_miss.cell_idx, None);
    assert!(ray_miss.intersection.distance(vec2(3.0, 0.0)) < EPSILON);
    assert!(ray_miss.normal.distance(vec2(-1.0, 0.0)) < EPSILON);
    
    assert_eq!(voronoi.nearest_cell_along_ray(vec2(-1.0, 0.0), vec2(1.0, 0.0)).cell_idx, Some(1));
    assert_eq!(voronoi.nearest_cell_along_ray(vec2(1.0, 0.0), vec2(-1.0, 0.0)).cell_idx, Some(0));
} 

#[test]
fn test_nearest_cell_along_ray_misses() {
    let voronoi = Voronoi {
        cells: vec![
            Cell { site: vec2(-1.0, 0.0), neighbor_indices: vec![1], info: None },
            Cell { site: vec2(1.0, 0.0), neighbor_indices: vec![0], info: None }
        ],
        bounds: vec2(3.0, 3.0),
        kdtree: Accelerator::from(&vec![[-1.0, 0.0], [1.0, 0.0]][..])
    };
    
    let ray_miss = voronoi.nearest_cell_along_ray(vec2(1.0, 0.0), vec2(1.0, 0.0));
    assert_eq!(ray_miss.cell_idx, None);
    assert!(ray_miss.intersection.distance(vec2(3.0, 0.0)) < EPSILON);
    assert!(ray_miss.normal.distance(vec2(-1.0, 0.0)) < EPSILON);
} 

#[test]
fn test_ray_boundary_intersections() {
    let voronoi = Voronoi {
        cells: vec![Cell { site: vec2(0.0, 0.0), neighbor_indices: vec![], info: None }],
        bounds: vec2(3.0, 3.0),
        kdtree: Accelerator::from(&vec![[0.0, 0.0]][..])
    };

    //cardinal directions
    assert!(
        voronoi.nearest_cell_along_ray(vec2(0.0, 0.0), vec2(1.0, 0.0))
        .intersection.distance(vec2(3.0, 0.0)) < EPSILON
    );
    assert!(
        voronoi.nearest_cell_along_ray(vec2(0.0, 0.0), vec2(-1.0, 0.0))
        .intersection.distance(vec2(-3.0, 0.0)) < EPSILON
    );
    assert!(
        voronoi.nearest_cell_along_ray(vec2(0.0, 0.0), vec2(0.0, 1.0))
        .intersection.distance(vec2(0.0, 3.0)) < EPSILON
    );
    assert!(
        voronoi.nearest_cell_along_ray(vec2(0.0, 0.0), vec2(0.0, -1.0))
        .intersection.distance(vec2(0.0, -3.0)) < EPSILON
    );
    //diagonals/corners
    assert!(
        voronoi.nearest_cell_along_ray(vec2(0.0, 0.0), vec2(1.0, 1.0).normalize())
        .intersection.distance(vec2(3.0, 3.0)) < EPSILON
    );
    assert!(
        voronoi.nearest_cell_along_ray(vec2(0.0, 0.0), vec2(1.0, -1.0).normalize())
        .intersection.distance(vec2(3.0, -3.0)) < EPSILON
    );
    assert!(
        voronoi.nearest_cell_along_ray(vec2(0.0, 0.0), vec2(-1.0, 1.0).normalize())
        .intersection.distance(vec2(-3.0, 3.0)) < EPSILON
    );
    assert!(
        voronoi.nearest_cell_along_ray(vec2(0.0, 0.0), vec2(-1.0, -1.0).normalize())
        .intersection.distance(vec2(-3.0, -3.0)) < EPSILON
    );
}