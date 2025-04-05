#![allow(dead_code)] // Allow unused code for example purposes
use crate::domain::grid2d::Grid2D;
use crate::boundary::bc2d::BoundaryConditions2D;
use crate::error::SolverError;
use tracing::{info, info_span, warn};

// --- Assume these functions exist in the same module or are imported ---
// --- from your previous code files (lib.rs or wherever they are)   ---
use crate::numerical::{
    calculate_viscous_x, calculate_viscous_y,
    interpolate_u_to_cell_centers, interpolate_v_to_cell_centers,
    interpolate_u_to_cell_edges, interpolate_v_to_cell_edges,
    calculate_nu, calculate_nv,
    calculate_intermediate_velocity,
    calculate_divergence_term,
    apply_pressure_corrections,
};
use crate::poisson::solve_poisson_equation_direct;
// --- End of assumed imports ---


#[derive(Debug)]
pub struct Solver {
    pub grid: Grid2D,
    pub re: f64,     // Reynolds number
    pub dt: f64,     // Time step
    pub time: f64,   // Current simulation time
    pub bcs: BoundaryConditions2D,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
}

impl Solver {
    pub fn new(grid: Grid2D, re: f64, dt: f64, bcs: BoundaryConditions2D) -> Result<Self, SolverError> {
        if re <= 0.0 {
             return Err(SolverError::InvalidParameter("Reynolds number must be positive".to_string()));
        }
        if dt <= 0.0 {
            return Err(SolverError::InvalidParameter("Time step dt must be positive".to_string()));
        }
        let nx = grid.dimensions.0;
        let ny = grid.dimensions.1;
        let dx = grid.cell_size.0;
        let dy = grid.cell_size.1;

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
        })
    }

    /// Performs one time step using the fractional step method.
    pub fn step(&mut self) -> Result<(), String> {
        // --- Apply BCs to current velocity ---
        self.grid.apply_lid_driven_bcs(1.0);

        // --- Calculate Viscous Terms ---
        let lux = calculate_viscous_x(&self.grid.u, self.dx);
        let luy = calculate_viscous_y(&self.grid.u, self.dy);
        let lvx = calculate_viscous_x(&self.grid.v, self.dx);
        let lvy = calculate_viscous_y(&self.grid.v, self.dy);

        // --- Calculate Convective Terms ---
        // 1. Interpolate
        let uce = interpolate_u_to_cell_centers(&self.grid.u);
        let uco = interpolate_u_to_cell_edges(&self.grid.u);
        let vco = interpolate_v_to_cell_edges(&self.grid.v);
        let vce = interpolate_v_to_cell_centers(&self.grid.v);

        // 2. Multiply
        let uuce = uce.component_mul(&uce);
        let uvco = uco.component_mul(&vco);
        let vvce = vce.component_mul(&vce);

        // 3. Calculate Nu and Nv derivatives
        let nu = calculate_nu(&uuce, &uvco, self.dx, self.dy);
        let nv = calculate_nv(&vvce, &uvco, self.dx, self.dy);

        // --- Predictor Step ---
        let (mut u_star, mut v_star) = calculate_intermediate_velocity(
            &self.grid.u, &self.grid.v,
            &nu, &nv,
            &lux, &luy, &lvx, &lvy,
            self.dt, self.re
        );

        // --- Calculate Divergence ---
        let b = calculate_divergence_term(&u_star, &v_star, self.dx, self.dy);

        // --- Solve Pressure Poisson Equation ---
        let p = match solve_poisson_equation_direct(&b, self.nx, self.ny, self.dx, self.dy, false) {
            Ok(p) => p,
            Err(e) => {
                return Err(e);
            }
        };
        self.grid.pressure = p.clone();

        // --- Corrector Step ---
        apply_pressure_corrections(&mut u_star, &mut v_star, &p, self.dx, self.dy);

        // Update grid velocities
        self.grid.u = u_star;
        self.grid.v = v_star;

        // --- Final BCs ---
        self.grid.apply_lid_driven_bcs(1.0);

        // --- Advance Time ---
        self.time += self.dt;
        
        Ok(())
    }

    /// Runs the simulation for a specified number of time steps.
    pub fn run(&mut self, num_steps: usize) -> Result<(), String> {
        let run_span = info_span!("simulation_run", num_steps = num_steps).entered();
        info!("Starting simulation with {} steps", num_steps);
        
        let start_time = std::time::Instant::now();
        
        for i in 0..num_steps {
            let step_span = info_span!("time_step", step = i + 1).entered();
            let step_start = std::time::Instant::now();
            
            match self.step() {
                Ok(_) => {
                    // Calculate divergence for monitoring
                    let b_check = calculate_divergence_term(&self.grid.u, &self.grid.v, self.dx, self.dy);
                    let div_norm = b_check.norm();
                    
                    // Log step time and divergence
                    let step_time = step_start.elapsed();
                    info!("Step {}: time={:.4}, div={:.3e}, elapsed={:.2}ms", 
                          i + 1, self.time, div_norm, step_time.as_millis());
                }
                Err(e) => {
                    let error_msg = format!("Error during step {}: {}", i + 1, e);
                    warn!(%error_msg, "Simulation step failed");
                    return Err(error_msg);
                }
            }
            drop(step_span);
        }
        
        let total_time = start_time.elapsed();
        info!("Simulation finished in {:.2}s", total_time.as_secs_f64());
        
        drop(run_span);
        Ok(())
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid2d::{GridDimensions2D, CellSize2D};
    use crate::boundary::bc2d::*;
    use nalgebra::dmatrix; // If needed for test setup
    use approx::assert_relative_eq; // If needed for float comparisons

    // Helper function to create a simple Grid2D for testing
    fn setup_test_grid() -> Grid2D {
         let nx = 5;
         let ny = 5;
         let dx = 1.0;
         let dy = 1.0;
         let dims = GridDimensions2D(nx, ny);
         let cell_size = CellSize2D(dx, dy);
         let mut grid = Grid2D::new(dims, cell_size).unwrap();

         // Use initial conditions from previous full step test
         let u_initial = dmatrix![
             30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
             38.0, 47.0,  7.0,  9.0, 18.0, 27.0, 29.0;
             46.0,  6.0,  8.0, 17.0, 26.0, 35.0, 37.0;
              5.0, 14.0, 16.0, 25.0, 34.0, 36.0, 45.0;
             13.0, 15.0, 24.0, 33.0, 42.0, 44.0,  4.0;
             21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0
         ];
         let v_initial = dmatrix![
             30.0, 39.0, 48.0,  1.0, 10.0, 19.0;
             38.0, 47.0,  7.0,  9.0, 18.0, 27.0;
             46.0,  6.0,  8.0, 17.0, 26.0, 35.0;
              5.0, 14.0, 16.0, 25.0, 34.0, 36.0;
             13.0, 15.0, 24.0, 33.0, 42.0, 44.0;
             21.0, 23.0, 32.0, 41.0, 43.0,  3.0;
             22.0, 31.0, 40.0, 49.0,  2.0, 11.0
         ];
         grid.u = u_initial;
         grid.v = v_initial;
         grid
    }

     // Helper function for basic boundary conditions
     fn setup_test_bcs() -> BoundaryConditions2D {
         let u_bc = SquareBoundary { // Example: all walls zero velocity
             x: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
             y: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
         };
         let v_bc = SquareBoundary {
             x: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
             y: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
         };
         let p_bc = SquareBoundary { // Standard Neumann for pressure
             x: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
             y: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
         };
         BoundaryConditions2D { u: u_bc, v: v_bc, p: p_bc }
     }

    #[test]
    fn test_solver_new() -> Result<(), SolverError> {
        let grid = setup_test_grid();
        let bcs = setup_test_bcs();
        let solver = Solver::new(grid, 100.0, 0.1, bcs)?;
        assert_eq!(solver.re, 100.0);
        assert_eq!(solver.dt, 0.1);
        assert_eq!(solver.time, 0.0);
        assert_eq!(solver.nx, 5);
        assert_eq!(solver.ny, 5);
        Ok(())
    }

    #[test]
    fn test_solver_new_invalid_params() {
        let grid = setup_test_grid();
        let bcs = setup_test_bcs();
        assert!(Solver::new(grid.clone(), 0.0, 0.1, bcs.clone()).is_err()); // Invalid Re
        assert!(Solver::new(grid.clone(), 100.0, 0.0, bcs.clone()).is_err()); // Invalid dt
    }

    #[test]
    fn test_solver_step() -> Result<(), String> {
        let grid = setup_test_grid();
        let bcs = setup_test_bcs();
        let mut solver = Solver::new(grid, 100.0, 0.1, bcs).unwrap();

        let u_before = solver.grid.u.clone();
        let v_before = solver.grid.v.clone();
        let time_before = solver.time;

        solver.step()?; // Execute one step

        let time_after = solver.time;
        let u_after = solver.grid.u.clone();
        let v_after = solver.grid.v.clone();

        // Basic checks: time should advance, fields should (probably) change
        assert_relative_eq!(time_after, time_before + solver.dt, epsilon = 1e-9);
        assert_ne!(u_after, u_before, "u field did not change after step");
        assert_ne!(v_after, v_before, "v field did not change after step");

        // Compare final state to the expected result from the full step test
        let expected_u_final = dmatrix![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; // Updated by BCs
            0.0, 59.254, 50.174, 20.644, 16.916, 28.012, 0.0; // Interior updated by step, sides by BCs
            0.0, 60.48, 49.0, 25.167, 19.677, 31.677, 0.0;
            0.0, 45.389, 33.193, 22.728, 18.241, 35.45, 0.0;
            0.0, 36.361, 27.563, 27.429, 34.168, -1.5212, 0.0;
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  // Updated by BCs
        ];
        let expected_v_final = dmatrix![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0; // Updated by BCs
            0.0, 64.746, 62.572, 42.928, 36.012, 0.0; // Interior updated by step, sides by BCs
            0.0, 44.774, 45.949, 41.426, 38.665, 0.0;
            0.0, 20.091, 35.898, 38.336, 39.773, 0.0;
            0.0, 22.027, 27.658, 22.956,  7.0288, 0.0;
            0.0, 34.361, 29.924, 16.353,  7.5212, 0.0;
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0 // Updated by BCs
        ];

        // Need to adjust expected values for the zero velocity BCs applied at the end
        println!("Solver u after step:\n{}", solver.grid.u);
        println!("Expected u after step & BCs:\n{}", expected_u_final);
        assert_relative_eq!(solver.grid.u, expected_u_final, epsilon = 1e-3);

        println!("Solver v after step:\n{}", solver.grid.v);
        println!("Expected v after step & BCs:\n{}", expected_v_final);
        assert_relative_eq!(solver.grid.v, expected_v_final, epsilon = 1e-3);

        Ok(())
    }

    #[test]
    fn test_solver_run() -> Result<(), String> {
        let grid = setup_test_grid();
        let bcs = setup_test_bcs();
        let mut solver = Solver::new(grid, 100.0, 0.1, bcs).unwrap();
        let num_steps = 5;
        let initial_time = solver.time;

        solver.run(num_steps)?;

        // Check if time advanced correctly
        let expected_time = initial_time + num_steps as f64 * solver.dt;
        assert_relative_eq!(solver.time, expected_time, epsilon = 1e-9);

        // Optional: Add more checks on the final state if a known solution exists for 5 steps
        println!("Solver state after run({} steps):\nTime: {:.4}", num_steps, solver.time);
        println!("Final u (sample):\n{}", solver.grid.u.fixed_view::<2, 2>(1, 1)); // Print a small part
        println!("Final v (sample):\n{}", solver.grid.v.fixed_view::<2, 2>(1, 1));

        Ok(())
    }
}