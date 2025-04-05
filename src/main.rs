// Import necessary modules from your crate
// Adjust paths based on your project structure
mod domain; // Contains grid2d.rs
mod boundary; // Contains bc2d.rs
mod error; // Contains error definitions
mod numerical; // Contains numerical operations (interpolate, viscous, nu/nv, etc.)
mod solver; // Contains solver.rs

use crate::domain::grid2d::{Grid2D, GridDimensions2D, CellSize2D};
use crate::boundary::bc2d::{BoundaryCondition, BoundaryConditions2D, FaceBoundary, SquareBoundary};
use crate::solver::Solver;

fn main() -> Result<(), Box<dyn std::error::Error>> { // Use standard Error trait
    println!("Setting up Lid-Driven Cavity Simulation...");

    // --- Simulation Parameters (Matching MATLAB) ---
    let re = 500.0;         // Reynolds number
    let num_steps = 2000;   // Total time steps
    let dt = 0.01;          // Time step
    let grid_dims = GridDimensions2D(80, 80);
    let cell_size = CellSize2D(1.0/80.0, 1.0/80.0);

    // --- Initial Conditions & Grid ---
    // Grid2D::new initializes u, v, p to zeros
    let grid = Grid2D::new(grid_dims, cell_size)?;

    // --- Boundary Conditions ---
    // Initial BCs: Top lid moves with U=1.0
    let initial_lid_velocity = 1.0;
    let bcs = setup_lid_driven_bcs(initial_lid_velocity); // Use helper function

    // --- Create Solver ---
    let mut solver = Solver::new(grid, re, dt, bcs.clone())?; // Clone BCs for solver

    println!("Setup Complete:");
    println!("  Grid: {}x{} cells", grid_dims.0, grid_dims.1);
    println!("  Cell Size: dx={:.4e}, dy={:.4e}", cell_size.0, cell_size.1);
    println!("  Re = {:.1}, dt = {:.4e}", re, solver.dt);
    println!("  Total Steps = {}", num_steps);
    println!("------------------------------------");

    // --- Simulation Loop ---
    println!("Starting simulation loop...");
    for i in 0..num_steps {

        // --- Perform One Time Step ---
        match solver.step() {
            Ok(_) => {
                print!("Completed step {:<5} / {} | Time = {:.4} | ", i + 1, num_steps, solver.time);
                // Optional: Calculate and print divergence norm
                let b_check = numerical::calculate_divergence_term(&solver.grid.u, &solver.grid.v, solver.dx, solver.dy);
                println!("Divergence Norm: {:.3e}", b_check.norm());
            }
            Err(e) => {
                eprintln!("Error during step {}: {}", i + 1, e);
                // Decide how to handle errors: break, return Err, etc.
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)));
            }
        }
    }

    println!("------------------------------------");
    println!("Simulation finished successfully.");
    println!("Final time: {:.4}", solver.time);

    // --- Print Final State (or save to file, etc.) ---
    // Printing large matrices might be slow/unwieldy. Print samples?
    println!("\nFinal u velocity (sample top-left 4x4 interior):");
    println!("{}", solver.grid.u.fixed_view::<4, 4>(1, 1)); // View from (1,1) size 4x4

    println!("\nFinal v velocity (sample top-left 4x4 interior):");
    println!("{}", solver.grid.v.fixed_view::<4, 4>(1, 1)); // View from (1,1) size 4x4

    println!("\nFinal pressure (sample top-left 4x4):");
    println!("{}", solver.grid.pressure.fixed_view::<4, 4>(0, 0)); // View from (0,0) size 4x4

    Ok(())
}


// --- Helper Function for Boundary Conditions ---
// (Could be moved to boundary/bc2d.rs if preferred)
fn setup_lid_driven_bcs(lid_velocity: f64) -> BoundaryConditions2D {
     let u_bc = SquareBoundary {
         x: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)), // Left, Right walls
         y: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(lid_velocity)), // Bottom wall, Top lid
     };
     let v_bc = SquareBoundary { // v is zero on all walls/lid
         x: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)), // Left, Right walls
         y: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)), // Bottom wall, Top lid
     };
     let p_bc = SquareBoundary { // Standard Neumann for pressure
         x: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
         y: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
     };
     BoundaryConditions2D { u: u_bc, v: v_bc, p: p_bc }
 }