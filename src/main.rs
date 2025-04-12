mod boundary;
mod domain;
mod error;
mod json_io;
mod numerical;
mod poisson;
mod solver;

use crate::boundary::bc2d::{
    BoundaryCondition, BoundaryConditions2D, FaceBoundary, SquareBoundary,
};
use crate::domain::grid2d::{CellSize2D, Grid2D, GridDimensions2D};
use crate::solver::Solver;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_level(true)
        .with_ansi(true)
        .init();

    info!("Setting up Lid-Driven Cavity Simulation...");

    let re = 1000.0;
    let num_steps = 20000;
    let dt = 0.002;
    let grid_dims = GridDimensions2D(100, 50);
    let cell_size = CellSize2D(1.0 / grid_dims.1 as f64, 1.0 / grid_dims.1 as f64);

    let output_frequency = Some(250);
    let output_filename = Some("output/wide.json".to_string());

    let grid = Grid2D::new(grid_dims, cell_size)?;

    let lid_velocity = 1.0;
    let u_bc = SquareBoundary {
        x: FaceBoundary(
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(0.0),
        ),
        y: FaceBoundary(
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(lid_velocity),
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
    let bcs = BoundaryConditions2D {
        u: u_bc,
        v: v_bc,
        p: p_bc,
    };

    let mut solver = Solver::new(
        grid,
        re,
        dt,
        bcs.clone(),
        output_frequency,
        output_filename.clone(),
    )?;

    match solver.run(num_steps) {
        Ok(()) => info!(
            "Simulation completed successfully. Output written to {}",
            output_filename.unwrap_or_else(|| "N/A (Output Disabled)".to_string())
        ),
        Err(e) => {
            let filename = output_filename.unwrap_or_else(|| "N/A".to_string());
            error!("Simulation failed: {}", e);
            error!("Partial results might be available in {}", filename);
            return Err(Box::new(e));
        }
    }

    Ok(())
}
