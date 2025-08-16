use std::f32::consts::PI;

use epson::utils::{generate_perturbed_grid, generate_points_white_noise};
use epson::voronoi_2d::*;

use glam::vec2;

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::RenderEvent;
use piston::window::WindowSettings;

fn main() {
    //let sites = generate_perturbed_grid(12);
    let sites = generate_points_white_noise(1000);

    let voronoi = voronoi(&sites, vec2(3.0, 3.0));

    let mut window: Window = WindowSettings::new("Voronoi Diagram", [800, 800])
        .graphics_api(OpenGL::V3_2)
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut gl = GlGraphics::new(OpenGL::V3_2);

    let mut events = Events::new(EventSettings::new());

    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            use graphics::*;

            gl.draw(args.viewport(), |c, g| {
                clear([0.1, 0.2, 0.3, 1.0], g);
                let t = c
                    .transform
                    .trans(400.0, 400.0)
                    .scale(800.0, 800.0)
                    .scale(1.0 / 6.0, 1.0 / 6.0);

                for (i, cell) in voronoi.cells.iter().enumerate() {
                    for wedge in cell.get_wedges().unwrap() {
                        polygon(
                            math::hsv(
                                [0.0, 1.0, 0.0, 0.5],
                                4.0f32 * PI * (i as f32 / voronoi.cells.len() as f32),
                                1.0,
                                1.0,
                            ),
                            &[
                                [cell.site.x as f64, cell.site.y as f64],
                                [wedge.0.x as f64, wedge.0.y as f64],
                                [wedge.1.x as f64, wedge.1.y as f64],
                            ],
                            t,
                            g,
                        );
                        line_from_to(
                            [0.0, 0.0, 0.0, 0.5],
                            4.0 / 800.0,
                            [cell.site.x as f64, cell.site.y as f64],
                            [wedge.0.x as f64, wedge.0.y as f64],
                            t,
                            g,
                        );
                        line_from_to(
                            [0.0, 0.0, 0.0, 0.5],
                            4.0 / 800.0,
                            [cell.site.x as f64, cell.site.y as f64],
                            [wedge.1.x as f64, wedge.1.y as f64],
                            t,
                            g,
                        );
                        line_from_to(
                            [0.0, 0.0, 1.0, 1.0],
                            4.0 / 800.0,
                            [wedge.0.x as f64, wedge.0.y as f64],
                            [wedge.1.x as f64, wedge.1.y as f64],
                            t,
                            g,
                        );
                    }
                }
            })
        }
    }
}
