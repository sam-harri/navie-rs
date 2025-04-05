// Import necessary modules from your crate
// Adjust paths based on your project structure
mod domain; // Contains grid2d.rs
mod boundary; // Contains bc2d.rs
mod error; // Contains error definitions
mod numerical; // Contains numerical operations (interpolate, viscous, nu/nv, etc.)
mod solver; // Contains solver.rs
mod poisson;
mod io;

use crate::domain::grid2d::{Grid2D, GridDimensions2D, CellSize2D};
use crate::boundary::bc2d::{BoundaryCondition, BoundaryConditions2D, FaceBoundary, SquareBoundary};
use crate::solver::Solver;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> { // Use standard Error trait
    let _ = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_level(true)
        .with_ansi(true)
        .init();

    info!("Setting up Lid-Driven Cavity Simulation...");

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
    let lid_velocity = 1.0;
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
    let bcs = BoundaryConditions2D { u: u_bc, v: v_bc, p: p_bc };
    let mut solver = Solver::new(grid, re, dt, bcs.clone())?; // Clone BCs for solver
    solver.run(num_steps)?;

    Ok(())
}