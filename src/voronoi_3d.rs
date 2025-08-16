use glam::{ vec3, Vec3, swizzles::Vec3Swizzles, mat3 };

use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;


type Accelerator = ImmutableKdTree<f32, usize, 3, 64>;


const EPSILON: f32 = 0.00001;
use std::collections::{VecDeque, HashSet};

pub struct Cell {
    pub site: Vec3,
    pub neighbor_indices: Vec<usize>,
}

pub struct Voronoi3D {
    pub cells: Vec<Cell>,
    bounds: Vec3,

    kdtree: Accelerator
}

struct RayTraceResult {
    intersection: Vec3,
    normal: Vec3,
    cell_idx: Option<usize>
}

struct Projector {
    r1: Vec3,
    r2: Vec3,
    r3: Vec3
}


impl Voronoi3D {
    fn calculate_cell_neighbors(&self, cell_idx: usize) -> Vec<usize> {
        let mut neighbors: HashSet<usize> = HashSet::new();
        let cell = &self.cells[cell_idx];

        // DEBUG: 8 initial projectors, one for each octant
        let mut projectors = VecDeque::from([
            Projector { r1:  Vec3::X, r2:  Vec3::Y, r3:  Vec3::Z },
            Projector { r1: -Vec3::X, r2:  Vec3::Y, r3:  Vec3::Z },
            Projector { r1:  Vec3::X, r2: -Vec3::Y, r3:  Vec3::Z },
            Projector { r1: -Vec3::X, r2: -Vec3::Y, r3:  Vec3::Z },
            Projector { r1:  Vec3::X, r2:  Vec3::Y, r3: -Vec3::Z },
            Projector { r1: -Vec3::X, r2:  Vec3::Y, r3: -Vec3::Z },
            Projector { r1:  Vec3::X, r2: -Vec3::Y, r3: -Vec3::Z },
            Projector { r1: -Vec3::X, r2: -Vec3::Y, r3: -Vec3::Z },
        ]);

        while !projectors.is_empty() {
            let projector = projectors.pop_front().unwrap();
            // println!("Processing projector {:} : {:}", projector.r1, projector.r2);

            let r1_hit = self.nearest_cell_along_ray(cell.site, projector.r1);
            let r2_hit = self.nearest_cell_along_ray(cell.site, projector.r2);
            let r3_hit = self.nearest_cell_along_ray(cell.site, projector.r3);

            let plane_intersection_system = mat3(r1_hit.normal, r2_hit.normal, r3_hit.normal).transpose();    
        
            if plane_intersection_system.determinant().abs() < EPSILON {
                //if the determinant of the plane intersection system is zero, 
                //the three plane normals are not linearly independent--so they are either colinear 
                //(making the 3 planes parallel) or coplanar (making the planes intersect in a line or not at all)
                
                //if the cross product of two vectors is zero, they are colinear
                //if the normals of two planes are colinear, the planes are parallel
                let p12_parallel = r1_hit.normal.cross(r2_hit.normal).length_squared() < EPSILON;
                let p23_parallel = r1_hit.normal.cross(r2_hit.normal).length_squared() < EPSILON;
                let p31_parallel = r1_hit.normal.cross(r2_hit.normal).length_squared() < EPSILON;

                if p12_parallel && p23_parallel { //all 3 planes are parallel
                    //since the each plane must be the closest bisector in a direction, and they're all parallel,
                    //at least 2 of the planes must actually be the same one (2 rays found the same bisector)
                    if r1_hit.cell_idx == r2_hit.cell_idx && r2_hit.cell_idx == r3_hit.cell_idx {
                        //all rays found the same neighbor, we can stop processing this projector
                        if let Some(neighbor) = r1_hit.cell_idx { neighbors.insert(neighbor); }
                        continue;
                    } else {
                        todo!("all three planes are parallel but 2 neighbor different cells!");
                    }
                } else if p12_parallel || p23_parallel || p31_parallel {
                    //two of the planes are parallel--it's useful to know which rays hit the parallel planes and the info for those,
                    //so we'll rename as necessary (r1/r2: hit parallel planes, r3: hit other plane)
                    let (repeated_hit, parallel_hit, nparallel_hit, r1, r2, r3) = {
                        if p12_parallel {
                            (
                                r1_hit.cell_idx == r2_hit.cell_idx,  //if the rays that hit parallel planes hit the same neighbor, we count it as a "repeated hit"
                                r1_hit, r3_hit, //r1 and r3 hit non-parallel planes
                                projector.r1, projector.r2, projector.r3
                            )
                        } else if p23_parallel {
                            (
                                r2_hit.cell_idx == r3_hit.cell_idx, //as above
                                r2_hit, r1_hit, //r2 and r1 hit non-parallel planes
                                projector.r2, projector.r3, projector.r1
                            )
                        } else {//p31 parallel
                            (
                                r2_hit.cell_idx == r3_hit.cell_idx, //as above
                                r1_hit, r2_hit, //r1 and r2 hit non-parallel planes
                                projector.r3, projector.r1, projector.r2
                            )
                        }
                    };

                    if repeated_hit {//we hit the same plane twice, this projector needs to be cut by the plane that goes
                        //through the origin of the cell and the intersection line of the planes. This plane will cut 2 of the projector faces,
                        //so we need to know what rays along those faces align with the plane. To do that, we need the intersection of those face planes
                        //with the two planes we've hit (for convenience, we'll get those intersections as coefficients of a linear combination of r1,r2,r3)
                        let basis = mat3(r1, r2, r3);
                        
                    }
                } else {

                }
            } else {
                //the three planes are in a non-degenerate formation, so there is a single intersection point
                //we want to get the intersection of the three planes as a linear combination of the wedge's 
                //rays, so we will get a basis matrix for them
                let basis = mat3(projector.r1, projector.r2, projector.r3);
                //m^-1 will allow us to find the unique intersection in terms of the basis matrix
                let m_inverse = (plane_intersection_system * basis).inverse();

                //the RHS of the plane intersection system is the dot products between points on each plane w/ the plane normals
                let rhs = vec3(
                    (r1_hit.intersection - cell.site).dot(r1_hit.normal),
                    (r2_hit.intersection - cell.site).dot(r2_hit.normal),
                    (r3_hit.intersection - cell.site).dot(r3_hit.normal)
                );

                //applying the inverse to the RHS gives us the solution to the SoE, and b/c we put it into the basis these are coefficients
                //for that intersection as a linear combination of the three projector rays
                let mut coeffs = m_inverse * rhs;

                let mut ray = basis * coeffs;
                let intersection = cell.site + ray;
                //if the linear combination has positive coefficients, the intersection is "in" this projector
                if coeffs.x > -EPSILON && coeffs.y > -EPSILON && coeffs.z > -EPSILON {
                    let intersection_nearest = self.kdtree.nearest_one::<SquaredEuclidean>(&intersection.to_array());
                    if (intersection_nearest.item == cell_idx || intersection_nearest.distance > cell.site.distance_squared(intersection) - 0.001) 
                        && intersection.abs().cmple(self.bounds + Vec3::splat(EPSILON)).all() //absolute value of all components of the intersection is less than the bounds
                    {
                        //the intersection is nearest to this cell, so we've successfully found a vertex and can stop searching his projector
                        if let Some(neighbor) = r1_hit.cell_idx { neighbors.insert(neighbor); /* println!("Found neighbor {:}", neighbor); */ }
                        if let Some(neighbor) = r2_hit.cell_idx { neighbors.insert(neighbor); /* println!("Found neighbor {:}", neighbor); */ }
                        if let Some(neighbor) = r3_hit.cell_idx { neighbors.insert(neighbor); /* println!("Found neighbor {:}", neighbor); */ }
                        continue;
                    }
                } else {
                    //if the coefficients are all negative, negate the ray (and the coefficients) to put it in the wedge
                    if(coeffs.x < 0.0 && coeffs.y < 0.0 && coeffs.z < 0.0) { ray = -ray; coeffs = -coeffs; }
                    else { //if the coefficients aren't all negative, then the ray needs to be "clamped" into the wedge.
                        if (coeffs.x < 0.0 && coeffs.y < 0.0) || (coeffs.x < 0.0 && coeffs.z < 0.0) || (coeffs.y < 0.0 && coeffs.z < 0.0) {
                            //if 2 of the 3 coefficients are negative, the negation will be closer to the wedge than the actual ray.
                            //this will also guarantee only one of the coefficients is negative and needs to be clamped
                            ray = -ray; coeffs = -coeffs;
                        }
                        //if a coefficient is negative, the component of the vector in the direction of the associated ray is in the wrong direction.
                        //To clamp it, subtract the component of the vector in the direction of that ray and zero out the coefficient.
                        if coeffs.x < 0.0 { ray -= projector.r1 * coeffs.x; coeffs.x = 0.0; }
                        if coeffs.y < 0.0 { ray -= projector.r2 * coeffs.y; coeffs.y = 0.0; }
                        if coeffs.z < 0.0 { ray -= projector.r3 * coeffs.z; coeffs.z = 0.0; }
                    }
                }

                //split the wedge we just processed in 3 with the new ray
                //If we had to clamp the intersection into the wedge, it will be coplanar with two of the rays
                //this would create a wedge with zero volume, so we want to prevent any of those
                if coeffs.x > EPSILON {
                    projectors.push_back(Projector {
                        r1: ray, r2: projector.r2, r3: projector.r3
                    });
                }
                if coeffs.y > EPSILON {
                    projectors.push_back(Projector {
                       r1: projector.r1, r2: ray, r3: projector.r3
                    });
                }
                if coeffs.z > EPSILON {
                    projectors.push_back(Projector {
                        r1: projector.r1, r2: projector.r2, r3: ray
                    });
                }
            }
        }

        neighbors.into_iter().collect()
    }

    //There is a line bisecting space between each pair of cells
    //this function finds the nearest intersection between a ray and one of those lines,
    //and returns the location of that intersection along with the index of the neighboring cell
    //(specifically, it will return the nearest neighbor to the cell that ray_origin lies in along
    //the direction ray_dir)
    fn nearest_cell_along_ray(&self,
        ray_origin: Vec3, ray_direction: Vec3
    ) -> RayTraceResult {    
        assert!(ray_origin.abs().x < self.bounds.x && ray_origin.abs().y < self.bounds.y);
        assert!((ray_direction.length() - 1.0).abs() < EPSILON);

        let initial_cell_idx = self.kdtree.nearest_one::<SquaredEuclidean>(&ray_origin.into()).item;
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
            (ray_origin + ray_direction * t.min_element(), Vec3::select(mask, -ray_direction.signum(), Vec3::ZERO))
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

//gets the intersection point of 3 planes, in a given basis
//returns a vec3 of coefficients which represents a vector in the given basis

fn triplane_intersection(
    p1_norm: Vec3, p1_point: Vec3,
    p2_norm: Vec3, p2_point: Vec3,
    p3_norm: Vec3, p3_point: Vec3,
    basis: glam::Mat3
) -> Option<Vec3> {
    
    None
}

fn ray_line_intersection_dist(
    ray_origin: Vec3, ray_direction: Vec3,
    line_origin: Vec3, line_normal: Vec3 
) -> Option<f32> {
    if ray_direction.dot(line_normal).abs() < EPSILON { return None; } 
    
    Some((line_origin - ray_origin).dot(line_normal) / (ray_direction).dot(line_normal))
}