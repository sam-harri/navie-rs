// src/main.rs

// Import necessary modules from your crate
mod domain;
mod boundary;
mod error;
mod numerical;
mod solver;
mod poisson;
mod json_io; // <-- Make sure this is declared

use crate::domain::grid2d::{Grid2D, GridDimensions2D, CellSize2D};
use crate::boundary::bc2d::{BoundaryCondition, BoundaryConditions2D, FaceBoundary, SquareBoundary};
use crate::solver::Solver; // Use the solver
use tracing::{info, error, Level}; // Added error
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_level(true)
        .with_ansi(true)
        .init();

    info!("Setting up Lid-Driven Cavity Simulation...");

    // --- Simulation Parameters ---
    let re = 500.0;
    let num_steps = 10000; // Reduced for quicker testing initially?
    let dt = 0.001;
    let grid_dims = GridDimensions2D(80, 80);
    let cell_size = CellSize2D(1.0 / grid_dims.0 as f64, 1.0 / grid_dims.1 as f64);

    // --- Output Parameters ---
    let output_frequency = Some(250); // Output every 100 steps (use Some(0) for initial/final only, None to disable)
    let output_filename = Some("output/lid_driven_results_refactored.json".to_string()); // Path relative to execution dir

    // --- Initial Conditions & Grid ---
    let grid = Grid2D::new(grid_dims, cell_size)?;

    // --- Boundary Conditions ---
    let lid_velocity = 1.0;
    let u_bc = SquareBoundary {
        x: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
        y: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(lid_velocity)),
    };
    let v_bc = SquareBoundary {
        x: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
        y: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
    };
    let p_bc = SquareBoundary {
        x: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
        y: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
    };
    let bcs = BoundaryConditions2D { u: u_bc, v: v_bc, p: p_bc };

    // --- Create Solver ---
    // Pass the output config options directly to Solver::new
    let mut solver = Solver::new(
        grid,
        re,
        dt,
        bcs.clone(),
        output_frequency, // Pass Option<usize>
        output_filename.clone(), // Pass Option<String> (clone if needed later)
    )?;

    // --- Run Simulation ---
    match solver.run(num_steps) {
        Ok(()) => info!(
            "Simulation completed successfully. Output written to {}",
            output_filename.unwrap_or_else(|| "N/A (Output Disabled)".to_string())
        ),
        Err(e) => {
            let filename = output_filename.unwrap_or_else(|| "N/A".to_string());
            error!("Simulation failed: {}", e);
            error!("Partial results might be available in {}", filename);
            return Err(Box::new(e)); // Propagate the error
        }
    }

    Ok(())
}