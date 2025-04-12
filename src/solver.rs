#![allow(dead_code)]
use crate::boundary::bc2d::BoundaryConditions2D;
use crate::domain::grid2d::Grid2D;
use crate::error::SolverError;
use std::io;
use std::time::Instant;
use tracing::{error, info, info_span, warn};

use crate::json_io::JsonOutputManager;
use crate::numerical::{
    apply_pressure_corrections, calculate_divergence_term, calculate_intermediate_velocity,
    calculate_nu, calculate_nv, calculate_viscous_x, calculate_viscous_y,
    interpolate_u_to_cell_edges, interpolate_v_to_cell_edges,
};
use crate::poisson::solve_poisson_equation_direct;

#[derive(Debug)]
pub struct Solver {
    pub grid: Grid2D,
    pub re: f64,
    pub dt: f64,
    pub time: f64,
    pub bcs: BoundaryConditions2D,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    output_manager: Option<JsonOutputManager>,
}

impl Solver {
    pub fn new(
        grid: Grid2D,
        re: f64,
        dt: f64,
        bcs: BoundaryConditions2D,
        output_frequency: Option<usize>,
        output_filepath: Option<String>,
    ) -> Result<Self, SolverError> {
        if re <= 0.0 {
            return Err(SolverError::InvalidParameter(
                "Reynolds number must be positive".to_string(),
            ));
        }
        if dt <= 0.0 {
            return Err(SolverError::InvalidParameter(
                "Time step dt must be positive".to_string(),
            ));
        }

        let nx = grid.dimensions.0;
        let ny = grid.dimensions.1;
        let dx = grid.cell_size.0;
        let dy = grid.cell_size.1;

        let output_manager = match output_filepath {
            Some(filepath) => {
                match JsonOutputManager::new(filepath, output_frequency, nx, ny /*, dx, dy */) {
                    Ok(manager) => Some(manager),
                    Err(io_err) => {
                        return Err(SolverError::IoError(format!(
                            "Failed to initialize JSON output manager: {}",
                            io_err
                        )));
                    }
                }
            }
            None => None,
        };

        info!(
            "Solver initialized. Output enabled: {}",
            output_manager.is_some()
        );

        Ok(Self {
            grid,
            re,
            dt,
            time: 0.0,
            bcs,
            nx,
            ny,
            dx,
            dy,
            output_manager,
        })
    }

    pub fn step(&mut self) -> Result<(), String> {
        self.grid.apply_lid_driven_bcs(1.0);

        let lux = calculate_viscous_x(&self.grid.u, self.dx);
        let luy = calculate_viscous_y(&self.grid.u, self.dy);
        let lvx = calculate_viscous_x(&self.grid.v, self.dx);
        let lvy = calculate_viscous_y(&self.grid.v, self.dy);

        let uce = crate::numerical::interpolate_u_to_cell_centers(&self.grid.u);
        let uco = interpolate_u_to_cell_edges(&self.grid.u);
        let vco = interpolate_v_to_cell_edges(&self.grid.v);
        let vce = crate::numerical::interpolate_v_to_cell_centers(&self.grid.v);
        let uuce = uce.component_mul(&uce);
        let uvco = uco.component_mul(&vco);
        let vvce = vce.component_mul(&vce);
        let nu = calculate_nu(&uuce, &uvco, self.dx, self.dy);
        let nv = calculate_nv(&vvce, &uvco, self.dx, self.dy);

        let (mut u_star, mut v_star) = calculate_intermediate_velocity(
            &self.grid.u,
            &self.grid.v,
            &nu,
            &nv,
            &lux,
            &luy,
            &lvx,
            &lvy,
            self.dt,
            self.re,
        );

        {
            let mut temp_grid = self.grid.clone();
            temp_grid.u = u_star.clone();
            temp_grid.v = v_star.clone();
            temp_grid.apply_lid_driven_bcs(1.0);
            u_star = temp_grid.u;
            v_star = temp_grid.v;
        }

        let b = calculate_divergence_term(&u_star, &v_star, self.dx, self.dy);
        let rhs = b * (1.0 / self.dt);

        let p_correction =
            match solve_poisson_equation_direct(&rhs, self.nx, self.ny, self.dx, self.dy, false) {
                Ok(p) => p,
                Err(e) => return Err(format!("Poisson solve failed: {}", e)),
            };
        self.grid.pressure = p_correction.clone();

        let p_correction_scaled = p_correction.scale(self.dt);
        apply_pressure_corrections(
            &mut u_star,
            &mut v_star,
            &p_correction_scaled,
            self.dx,
            self.dy,
        );

        self.grid.u = u_star;
        self.grid.v = v_star;

        self.grid.apply_lid_driven_bcs(1.0);

        self.time += self.dt;

        Ok(())
    }

    pub fn run(&mut self, num_steps: usize) -> Result<(), io::Error> {
        let run_span = info_span!("simulation_run", num_steps = num_steps).entered();
        info!(
            "Starting simulation: {} steps, dt={}, Re={}, Output: {}",
            num_steps,
            self.dt,
            self.re,
            self.output_manager
                .as_ref()
                .map_or("Disabled", |m| m.output_filepath.as_str())
        );
        let start_time = Instant::now();
        let mut steps_completed = 0;

        if let Some(manager) = self.output_manager.as_mut() {
            if manager.should_collect(0, num_steps, false) {
                manager.collect_timestep(0, self.time, &self.grid)?;
            }
        }

        for i in 0..num_steps {
            let current_step = i + 1;
            let step_span = info_span!("time_step", step = current_step).entered();
            let step_start = Instant::now();

            match self.step() {
                Ok(_) => {
                    steps_completed += 1;
                    let div_check_term =
                        calculate_divergence_term(&self.grid.u, &self.grid.v, self.dx, self.dy);
                    let div_norm = div_check_term.norm();
                    let step_time_elapsed = step_start.elapsed();
                    info!(
                        "Step {}: div_norm = {:.2e}, time = {:.2}ms",
                        current_step,
                        div_norm,
                        step_time_elapsed.as_secs_f64() * 1000.0
                    );

                    if let Some(manager) = self.output_manager.as_mut() {
                        let is_final = current_step == num_steps;
                        if manager.should_collect(current_step, num_steps, is_final) {
                            if let Err(e) =
                                manager.collect_timestep(current_step, self.time, &self.grid)
                            {
                                warn!(
                                    "Failed to collect timestep data for step {}: {}",
                                    current_step, e
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    let error_msg = format!("Error during step {}: {}", current_step, e);
                    error!(error=%e, step=current_step, "Simulation step failed");
                    if let Some(manager) = self.output_manager.as_ref() {
                        warn!("Attempting to write partial results due to simulation error...");
                        if let Err(write_err) = manager.write_final_output(
                            self.re,
                            self.dt,
                            self.dx,
                            self.dy,
                            steps_completed,
                        ) {
                            error!("Failed to write partial JSON output: {}", write_err);
                        } else {
                            info!("Partial results saved successfully.");
                        }
                    }
                    return Err(io::Error::new(io::ErrorKind::Other, error_msg));
                }
            }
            drop(step_span);
        } // --- End Time Stepping Loop ---

        let total_time = start_time.elapsed();
        info!(
            "Simulation loop finished in {:.2}s",
            total_time.as_secs_f64()
        );
        if let Some(manager) = self.output_manager.as_ref() {
            info!("Writing final JSON output...");
            manager.write_final_output(self.re, self.dt, self.dx, self.dy, steps_completed)?;
        }

        drop(run_span);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::bc2d::{
        BoundaryCondition, BoundaryConditions2D, FaceBoundary, SquareBoundary,
    };
    use crate::domain::grid2d::{CellSize2D, GridDimensions2D};
    use std::fs;
    use tempfile::tempdir;
    fn setup_test_grid_basic(nx: usize, ny: usize) -> Grid2D {
        let dims = GridDimensions2D(nx, ny);
        let cell_size = CellSize2D(1.0 / nx as f64, 1.0 / ny as f64);
        Grid2D::new(dims, cell_size).unwrap()
    }

    fn setup_test_bcs_basic() -> BoundaryConditions2D {
        let u_bc = SquareBoundary {
            x: FaceBoundary(
                BoundaryCondition::Dirichlet(0.0),
                BoundaryCondition::Dirichlet(0.0),
            ),
            y: FaceBoundary(
                BoundaryCondition::Dirichlet(0.0),
                BoundaryCondition::Dirichlet(0.0),
            ),
        };
        let v_bc = SquareBoundary {
            x: FaceBoundary(
                BoundaryCondition::Dirichlet(0.0),
                BoundaryCondition::Dirichlet(0.0),
            ),
            y: FaceBoundary(
                BoundaryCondition::Dirichlet(0.0),
                BoundaryCondition::Dirichlet(0.0),
            ),
        };
        let p_bc = SquareBoundary {
            x: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
            y: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
        };
        BoundaryConditions2D {
            u: u_bc,
            v: v_bc,
            p: p_bc,
        }
    }

    #[test]
    fn test_solver_new_with_output_manager() -> Result<(), Box<dyn std::error::Error>> {
        let grid = setup_test_grid_basic(5, 5);
        let bcs = setup_test_bcs_basic();
        let dir = tempdir()?;
        let filepath = dir.path().join("output.json");
        let filepath_str = filepath.to_str().unwrap().to_string();

        let solver = Solver::new(grid, 100.0, 0.1, bcs, Some(10), Some(filepath_str.clone()))?;

        assert!(solver.output_manager.is_some());
        let manager = solver.output_manager.unwrap();
        assert_eq!(manager.output_frequency, Some(10));
        assert_eq!(manager.output_filepath, filepath_str);
        assert_eq!(manager.nx, 5);
        assert_eq!(manager.ny, 5);

        dir.close()?;
        Ok(())
    }

    #[test]
    fn test_solver_new_output_disabled() -> Result<(), SolverError> {
        let grid = setup_test_grid_basic(5, 5);
        let bcs = setup_test_bcs_basic();
        let solver = Solver::new(grid, 100.0, 0.1, bcs, Some(10), None)?;
        assert!(solver.output_manager.is_none());
        Ok(())
    }

    #[test]
    fn test_solver_run_with_json_integration() -> Result<(), Box<dyn std::error::Error>> {
        let nx = 4;
        let ny = 3;
        let grid = setup_test_grid_basic(nx, ny);
        let bcs = setup_test_bcs_basic();
        let dir = tempdir()?;
        let filepath = dir.path().join("solver_run_output.json");
        let filepath_str = filepath.to_str().unwrap().to_string();
        let output_freq = 2;
        let num_steps = 5;

        let mut solver = Solver::new(
            grid,
            50.0,
            0.05,
            bcs,
            Some(output_freq),
            Some(filepath_str.clone()),
        )?;
        solver.run(num_steps)?;

        assert!(filepath.exists());
        let json_content = fs::read_to_string(filepath)?;
        let output_value: serde_json::Value = serde_json::from_str(&json_content)?;

        assert_eq!(output_value["metadata"]["nx"], nx);
        assert_eq!(output_value["metadata"]["re"], 50.0);
        assert_eq!(output_value["metadata"]["num_steps_completed"], num_steps);
        assert_eq!(output_value["metadata"]["output_frequency"], output_freq);

        let expected_steps = vec![0, 2, 4, 5];
        let data_array = output_value["data"]
            .as_array()
            .expect("Data field not an array");
        assert_eq!(data_array.len(), expected_steps.len());
        for (i, step) in expected_steps.iter().enumerate() {
            assert_eq!(data_array[i]["step"], *step as i64);
            assert_eq!(
                data_array[i]["u_centers"].as_array().unwrap().len(),
                nx * ny
            );
        }
        dir.close()?;
        Ok(())
    }
}
