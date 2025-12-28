use geometric_algebra::{ppga3d::*, OuterProduct, RegressiveProduct, SquaredMagnitude};
use glam::{mat3, swizzles::Vec3Swizzles, vec3, Vec3};

use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::SquaredEuclidean;

use rayon::prelude::*;

type Accelerator = ImmutableKdTree<f32, usize, 3, 64>;

const EPSILON: f32 = 1e-6;
use std::collections::{HashSet, VecDeque};

pub struct Cell {
    pub site: Vec3,
    pub neighbor_indices: Vec<usize>,

    pub info: Option<CellInfo>,
}

pub struct CellInfo {
    pub wedges: Vec<(Vec3, Vec3, Vec3)>,
}

pub struct Voronoi3D {
    pub cells: Vec<Cell>,
    bounds: Vec3,

    kdtree: Accelerator,
}

struct RayTraceResult {
    intersection: Vec3,
    normal: Vec3,
    cell_idx: Option<usize>,
}

struct Projector {
    r1: Vec3,
    r2: Vec3,
    r3: Vec3,
}

fn point_from_vec3(p: Vec3) -> Point {
    Point::new(1.0, p.x, p.y, p.z)
}

pub fn voronoi_3d(sites: &Vec<Vec3>, bounds: Vec3) -> Voronoi3D {
    let points: Vec<_> = sites
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect();
    let kdtree: Accelerator = Accelerator::from(&points[..]);
    drop(points);

    let mut result = Voronoi3D {
        cells: sites
            .iter()
            .map(|point| Cell {
                site: *point,
                neighbor_indices: vec![],
                info: None,
            })
            .collect(),
        bounds,
        kdtree,
    };

    result.calculate_all_cell_neighbors_parallel();

    result
}

impl Voronoi3D {
    fn calculate_all_cell_neighbors_parallel(&mut self) {
        let results: Vec<_> = (0..self.cells.len())
            .into_iter()
            .map(|i| self.calculate_cell_neighbors(i))
            .collect();
        for (i, r) in results.into_iter().enumerate() {
            self.cells[i].neighbor_indices = r.0;
            self.cells[i].info = Some(r.1);
        }
    }

    fn calculate_cell_neighbors(&self, cell_idx: usize) -> (Vec<usize>, CellInfo) {
        let mut neighbors: HashSet<usize> = HashSet::new();
        let mut info = CellInfo { wedges: Vec::new() };
        let cell = &self.cells[cell_idx];

        // DEBUG: 8 initial projectors, one for each octant
        let mut projectors = VecDeque::from([
            Projector {
                r1: Vec3::X,
                r2: Vec3::Y,
                r3: Vec3::Z,
            },
            Projector {
                r1: -Vec3::X,
                r2: Vec3::Y,
                r3: Vec3::Z,
            },
            Projector {
                r1: Vec3::X,
                r2: -Vec3::Y,
                r3: Vec3::Z,
            },
            Projector {
                r1: -Vec3::X,
                r2: -Vec3::Y,
                r3: Vec3::Z,
            },
            Projector {
                r1: Vec3::X,
                r2: Vec3::Y,
                r3: -Vec3::Z,
            },
            Projector {
                r1: -Vec3::X,
                r2: Vec3::Y,
                r3: -Vec3::Z,
            },
            Projector {
                r1: Vec3::X,
                r2: -Vec3::Y,
                r3: -Vec3::Z,
            },
            Projector {
                r1: -Vec3::X,
                r2: -Vec3::Y,
                r3: -Vec3::Z,
            },
        ]);

        while !projectors.is_empty() {
            let projector = projectors.pop_front().unwrap();
            println!(
                "Processing projector {:} - {:} - {:}",
                projector.r1, projector.r2, projector.r3
            );

            let r1_hit = self.nearest_cell_along_ray(cell.site, projector.r1);
            let r2_hit = self.nearest_cell_along_ray(cell.site, projector.r2);
            let r3_hit = self.nearest_cell_along_ray(cell.site, projector.r3);

            //store the points of intersection--these will be useful for the geometry of the cell
            let wedge = (
                r1_hit.intersection,
                r2_hit.intersection,
                r3_hit.intersection,
            );

            let p1 = Plane::new(
                -r1_hit.normal.dot(r1_hit.intersection),
                r1_hit.normal.x,
                r1_hit.normal.y,
                r1_hit.normal.z,
            );
            let p2 = Plane::new(
                -r2_hit.normal.dot(r2_hit.intersection),
                r2_hit.normal.x,
                r2_hit.normal.y,
                r2_hit.normal.z,
            );
            let p3 = Plane::new(
                -r3_hit.normal.dot(r3_hit.intersection),
                r3_hit.normal.x,
                r3_hit.normal.y,
                r3_hit.normal.z,
            );

            //the outer product of two planes is a line, the outer product of a line and a plane is a point.
            //if two of the planes are parallel to eachother, their intersection is at infinity--this will produce an 'ideal' point (direction)
            let intersection = p1.outer_product(p2).outer_product(p3);

            if intersection.squared_magnitude() < EPSILON {
                //degenerate formation--two of the three planes must be the same
                let p12_degen = p1.outer_product(p2).squared_magnitude() < EPSILON;
                let p23_degen = p2.outer_product(p3).squared_magnitude() < EPSILON;

                if p12_degen && p23_degen {
                    //all rays found the same neighbor, we can stop processing this projector
                    if let Some(neighbor) = r1_hit.cell_idx {
                        neighbors.insert(neighbor);
                        info.wedges.push(wedge);
                    }
                    continue;
                }
                //two of the planes are the same, the third isn't--rename p1 to be one of the degenerate planes and p2 to be the non-degenerate plane
                //additionally, rename r1,r2,r3 such that r1/r2 hit the degenerate planes and r3 hit the other
                let (p1, p2, r1, r2, r3) = if p12_degen {
                    (p2, p3, projector.r1, projector.r2, projector.r3)
                } else if p23_degen {
                    (p3, p1, projector.r2, projector.r3, projector.r1)
                } else
                /*p31_degen*/
                {
                    (p1, p2, projector.r3, projector.r1, projector.r2)
                };

                //find the line intersection between the two planes
                let intersection = p1.outer_product(p2);

                //this wedge needs to be cut by the plane that goes through the origin of the cell and the intersection line of the planes. This plane will cut 2 of the wedge faces,
                //so we need to know what rays along those faces align with the plane. To do that, we need the intersection of those face planes and the line intersection
                let projector_plane_1 = point_from_vec3(cell.site)
                    .regressive_product(point_from_vec3(cell.site + r1))
                    .regressive_product(point_from_vec3(cell.site + r3));
                let projector_plane_2 = point_from_vec3(cell.site)
                    .regressive_product(point_from_vec3(cell.site + r2))
                    .regressive_product(point_from_vec3(cell.site + r3));
                let a = intersection.outer_product(projector_plane_1);
                let b = intersection.outer_product(projector_plane_2);
                if a.squared_magnitude() < EPSILON {
                    //the intersection line is contained within (or very near to) the first projector plane
                    //therefore, one of the planes is basically outside of the projector and one is fully covering it
                    //the plane hit by r2 (the ray which is not in the first projector plane) should be the useful one
                    //since r2 was renamed such that it must've hit the degenerate planes, we just need to select one of them
                    let cell = if p12_degen {
                        r1_hit.cell_idx
                    } else {
                        r3_hit.cell_idx
                    };
                    if let Some(neighbor) = cell {
                        println!("Found neighbor {:}", neighbor);
                        neighbors.insert(neighbor);
                    }
                    //since the three rays *basically* hit the same plane, we can use the original wedge
                    info.wedges.push(wedge);
                    continue;
                } else if b.squared_magnitude() < EPSILON {
                    //the intersection line is contained within (or very near to) the second projector plane
                    //as above, the plane hit by r3 (the ray which is not in the second projector plane) should be the useful one
                    //since r3 was renamed such that it must've hit the degenerate planes, we just need to use the neighbor from it
                    let cell = if p12_degen {
                        r3_hit.cell_idx
                    } else if p23_degen {
                        r1_hit.cell_idx
                    } else {
                        r2_hit.cell_idx
                    };
                    if let Some(neighbor) = cell {
                        neighbors.insert(neighbor);
                        println!("Found neighbor {:}", neighbor);
                    }
                    //since the three rays *basically* hit the same plane, we can use the original wedge
                    info.wedges.push(wedge);
                    continue;
                }
                //convert to vec3 for easy manipulation
                let (intersection_1, intersection_2) = (
                    vec3(a[1] / a[0], a[2] / a[0], a[3] / a[0]),
                    vec3(b[1] / b[0], b[2] / b[0], b[3] / b[0]),
                );

                //if both intersections are nearest to this cell, we've successfully found an edge of the cell and can stop calculating
                let intersection_1_nearest = self
                    .kdtree
                    .nearest_one::<SquaredEuclidean>(&intersection_1.to_array());
                let intersection_2_nearest = self
                    .kdtree
                    .nearest_one::<SquaredEuclidean>(&intersection_2.to_array());

                if (intersection_1_nearest.item == cell_idx
                    || intersection_1_nearest.distance
                        > cell.site.distance_squared(intersection_1) - 0.001)
                    && (intersection_2_nearest.item == cell_idx
                        || intersection_2_nearest.distance
                            > cell.site.distance_squared(intersection_2) - 0.001)
                {
                    //figure out which two cells we hit
                    let (cell_1, cell_2) = if p12_degen {
                        (r1_hit.cell_idx, r3_hit.cell_idx)
                    } else if p23_degen {
                        (r1_hit.cell_idx, r2_hit.cell_idx)
                    } else {
                        (r1_hit.cell_idx, r2_hit.cell_idx)
                    };
                    if let Some(neighbor) = cell_1 {
                        neighbors.insert(neighbor);
                        println!("Found neighbor {:}", neighbor);
                    }
                    if let Some(neighbor) = cell_2 {
                        neighbors.insert(neighbor);
                        println!("Found neighbor {:}", neighbor);
                    }
                    //TODO! Add wedges to info
                    

                    continue;
                }

                let split_ray_1 = (intersection_1 - cell.site).normalize();
                let split_ray_2 = (intersection_2 - cell.site).normalize();

                println!("Biplane!");
                println!(
                    "Creating subprojector {:} - {:} - {:}",
                    r3, split_ray_1, split_ray_2
                );
                projectors.push_back(Projector {
                    r1: r3,
                    r2: split_ray_1,
                    r3: split_ray_2,
                });
                println!(
                    "Creating subprojector {:} - {:} - {:}",
                    r1, split_ray_1, split_ray_2
                );
                projectors.push_back(Projector {
                    r1: r1,
                    r2: split_ray_1,
                    r3: split_ray_2,
                });
                println!("Creating subprojector {:} - {:} - {:}", split_ray_2, r1, r2);
                projectors.push_back(Projector {
                    r1: split_ray_2,
                    r2: r1,
                    r3: r2,
                });
            } else {
                //non-degenerate intersection--we want to find the direction from the cell site to it
                let r = Point::new(1.0, cell.site.x, cell.site.y, cell.site.z)
                    .regressive_product(intersection);
                //to get the direction, find it's intersection with the "horizon" plane at infinity
                let dir = r.outer_product(Plane::new(1.0, 0.0, 0.0, 0.0));
                //extract the direction vector
                let mut ray = vec3(dir[1], dir[2], dir[3]).normalize();

                //get the decomposition of the ray into r1, r2, r3
                let mut coeffs = mat3(projector.r1, projector.r2, projector.r3).inverse() * ray;

                let intersection = if intersection[0].abs() > EPSILON {
                    //the intersection is an actual point, so get its normalized position
                    vec3(
                        intersection[1] / intersection[0],
                        intersection[2] / intersection[0],
                        intersection[3] / intersection[0],
                    )
                } else {
                    //the intersection was an ideal point, so make it really far away (so it's not closest to the cell site)
                    cell.site + vec3(intersection[1], intersection[2], intersection[3]) * 1_000f32
                };

                println!("Found intersection! {:?}", intersection);

                //if the linear combination has positive coefficients, the intersection is "in" this projector
                if coeffs.x > -EPSILON && coeffs.y > -EPSILON && coeffs.z > -EPSILON {
                    let intersection_nearest = self
                        .kdtree
                        .nearest_one::<SquaredEuclidean>(&intersection.to_array());
                    if (intersection_nearest.item == cell_idx
                        || intersection_nearest.distance
                            > cell.site.distance_squared(intersection) - 0.0001)
                        && intersection
                            .abs()
                            .cmple(self.bounds + Vec3::splat(EPSILON))
                            .all()
                    //absolute value of all components of the intersection is less than the bounds
                    {
                        //the intersection is nearest to this cell, so we've successfully found a vertex and can stop searching his projector
                        if let Some(neighbor) = r1_hit.cell_idx {
                            neighbors.insert(neighbor);
                            println!("Found neighbor {:}", neighbor);
                        }
                        if let Some(neighbor) = r2_hit.cell_idx {
                            neighbors.insert(neighbor);
                            println!("Found neighbor {:}", neighbor);
                        }
                        if let Some(neighbor) = r3_hit.cell_idx {
                            neighbors.insert(neighbor);
                            println!("Found neighbor {:}", neighbor);
                        }

                        info.wedges
                            .push((r1_hit.intersection, r2_hit.intersection, intersection));
                        info.wedges
                            .push((r2_hit.intersection, r3_hit.intersection, intersection));
                        info.wedges
                            .push((r3_hit.intersection, r1_hit.intersection, intersection));
                        continue;
                    }
                } else {
                    //if the coefficients are all negative, negate the ray (and the coefficients) to put it in the wedge
                    if coeffs.x < EPSILON && coeffs.y < EPSILON && coeffs.z < EPSILON {
                        ray = -ray;
                        coeffs = -coeffs;
                    } else {
                        //if the coefficients aren't all negative, then the ray needs to be "clamped" into the wedge.
                        if (coeffs.x < 0.0 && coeffs.y < 0.0)
                            || (coeffs.x < 0.0 && coeffs.z < 0.0)
                            || (coeffs.y < 0.0 && coeffs.z < 0.0)
                        {
                            //if 2 of the 3 coefficients are negative, the negation will be closer to the wedge than the actual ray.
                            //this will also guarantee only one of the coefficients is negative and needs to be clamped
                            ray = -ray;
                            coeffs = -coeffs;
                        }
                        //if a coefficient is negative, the component of the vector in the direction of the associated ray is in the wrong direction.
                        //To clamp it, subtract the component of the vector in the direction of that ray and zero out the coefficient.
                        if coeffs.x < 0.0 {
                            ray -= projector.r1 * coeffs.x;
                            coeffs.x = 0.0;
                        }
                        if coeffs.y < 0.0 {
                            ray -= projector.r2 * coeffs.y;
                            coeffs.y = 0.0;
                        }
                        if coeffs.z < 0.0 {
                            ray -= projector.r3 * coeffs.z;
                            coeffs.z = 0.0;
                        }
                    }
                }
                ray = ray.normalize();

                //all of the coefficients should be non negative, and if two coefficients are zero
                //then the new ray is a duplicate of one of the initial rays. Nothing new can be found
                //by splitting by it, so discard the wedge
                if (coeffs.x < EPSILON && coeffs.y < EPSILON)
                    || (coeffs.y < EPSILON && coeffs.z < EPSILON)
                    || (coeffs.z < EPSILON && coeffs.x < EPSILON)
                {
                    continue;
                }

                //split the wedge we just processed in 3 with the new ray
                //If we had to clamp the intersection into the wedge, it will be coplanar with two of the rays
                //this would create a wedge with zero volume, so we want to prevent any of those
                if coeffs.x > EPSILON.sqrt() {
                    projectors.push_back(Projector {
                        r1: ray,
                        r2: projector.r2,
                        r3: projector.r3,
                    });
                    println!(
                        "Creating subprojector {:} - {:} - {:}",
                        ray, projector.r2, projector.r3
                    );
                }
                if coeffs.y > EPSILON.sqrt() {
                    projectors.push_back(Projector {
                        r1: projector.r1,
                        r2: ray,
                        r3: projector.r3,
                    });
                    println!(
                        "Creating subprojector {:} - {:} - {:}",
                        projector.r1, ray, projector.r3
                    );
                }
                if coeffs.z > EPSILON.sqrt() {
                    projectors.push_back(Projector {
                        r1: projector.r1,
                        r2: projector.r2,
                        r3: ray,
                    });
                    println!(
                        "Creating subprojector {:} - {:} - {:}",
                        projector.r1, projector.r2, ray
                    );
                }
            }
        }
        (neighbors.into_iter().collect(), info)
    }

    //There is a line bisecting space between each pair of cells
    //this function finds the nearest intersection between a ray and one of those lines,
    //and returns the location of that intersection along with the index of the neighboring cell
    //(specifically, it will return the nearest neighbor to the cell that ray_origin lies in along
    //the direction ray_dir)
    fn nearest_cell_along_ray(&self, ray_origin: Vec3, ray_direction: Vec3) -> RayTraceResult {
        assert!(ray_origin.abs().x < self.bounds.x && ray_origin.abs().y < self.bounds.y);
        assert!((ray_direction.length() - 1.0).abs() < EPSILON);

        let initial_cell_idx = self
            .kdtree
            .nearest_one::<SquaredEuclidean>(&ray_origin.into())
            .item;
        let initial_cell = &self.cells[initial_cell_idx as usize];

        let (initial_intersection, initial_intersection_normal) = {
            //the bounding box is a rectangle with side lengths 2 * bounds.xyz, centered at the origin
            //since the ray_origin must be inside the bounding box, the rays will intersect with the
            //bounding edge with the sign of the ray along each axis
            let bounding_line = self.bounds.copysign(ray_direction);

            //find the intersection distances for each axis
            let t = (bounding_line - ray_origin) / ray_direction;

            //return the closest of the intersections
            //create a mask {x < z, y < x, z < y} && {x < y, y < z, z < x} (which will have "true" in the component corresponding to the minimum element)
            let mask = t.xyz().cmplt(t.zxy()) & t.xyz().cmplt(t.yzx());
            //the intersection will be the along the ray with the minimum 't' value--the normal of that intersection will be a unit vector along the axis
            //the intersection occurs along (corresponding to the minimum 't'), in the opposite direction of the ray
            (
                ray_origin + ray_direction * t.min_element(),
                Vec3::select(mask, -ray_direction.signum(), Vec3::ZERO),
            )
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

            nearest_cell_to_intersection_idx = self
                .kdtree
                .nearest_one::<SquaredEuclidean>(
                    &(initial_cell.site * (1.0 - t) + initial_intersection * t).into(),
                )
                .item;
            if nearest_cell_to_intersection_idx != initial_cell_idx {
                break;
            }
        }

        while nearest_cell_to_intersection_idx != initial_cell_idx {
            let nearest_cell = &self.cells[nearest_cell_to_intersection_idx as usize];

            //get the vector pointing from nearest_cell to initial_cell
            let directed_difference = initial_cell.site - nearest_cell.site;
            let midpoint = nearest_cell.site + directed_difference / 2.0;
            let bisector_normal = directed_difference.normalize();

            if let Some(dist) =
                ray_line_intersection_dist(ray_origin, ray_direction, midpoint, bisector_normal)
            {
                let intersection = ray_origin + ray_direction * dist;
                let new_nearest_cell_to_intersection = self
                    .kdtree
                    .nearest_one::<SquaredEuclidean>(&intersection.into());

                if new_nearest_cell_to_intersection.item == nearest_cell_to_intersection_idx
                    || new_nearest_cell_to_intersection.item == initial_cell_idx
                    || new_nearest_cell_to_intersection.distance < dist * dist + EPSILON
                {
                    //we've found the same nearest cell twice or made it back to the initial cell, so the bisector
                    //must be the closest one along the ray. Return the non-initial cell & intersection
                    return RayTraceResult {
                        intersection,
                        normal: bisector_normal,
                        cell_idx: Some(nearest_cell_to_intersection_idx as usize),
                    };
                } else {
                    nearest_cell_to_intersection_idx = new_nearest_cell_to_intersection.item;
                }
            } else {
                //there was no intersection between the ray and the bisector of the two cells,
                //so we should stop searching and return the bounding intersection
                break;
            }
        }

        RayTraceResult {
            intersection: initial_intersection,
            normal: initial_intersection_normal,
            cell_idx: None,
        }
    }
}

fn ray_line_intersection_dist(
    ray_origin: Vec3,
    ray_direction: Vec3,
    line_origin: Vec3,
    line_normal: Vec3,
) -> Option<f32> {
    if ray_direction.dot(line_normal).abs() < EPSILON {
        return None;
    }

    Some((line_origin - ray_origin).dot(line_normal) / (ray_direction).dot(line_normal))
}
