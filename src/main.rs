#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use boundary::bc2d::{BoundaryCondition, BoundaryConditions2D, FaceBoundary, SquareBoundary};
use domain::grid2d::{CellSize2D, Grid2D, GridDimensions2D};
use solver::simple::SimpleSolver;

mod solver;
mod domain;
mod boundary;
mod error;
mod numerical;

fn main() -> Result<(), Box<dyn std::error::Error>>{

    let dimensions = GridDimensions2D(500, 500);
    let cell_size = CellSize2D(1.0, 1.0);
    let grid = Grid2D::new(dimensions, cell_size)?;

    let bcs = BoundaryConditions2D {
        u: SquareBoundary {
            x: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
            y: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(1.0)),
        },
        v: SquareBoundary {
            x: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
            y: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
        },
        p: SquareBoundary {
            x: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
            y: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
        }
    };


    let dt = 0.01;
    let max_iterations = 100;
    let tolerance = 1e-3;
    let alpha = 0.5;

    // Fluid properties for water at ~25°C:
    let rho = 997.0;        // density in kg/m^3
    let nu = 8.9e-7;        // kinematic viscosity in m^2/s

    let mut solver = SimpleSolver::new(grid, dt, max_iterations, tolerance, alpha, bcs, rho, nu)?;

    // --- Advance the solution in time ---
    // For demonstration, we run 100 time steps.
    solver.solve(100);

    // --- Output final fields (for inspection) ---
    println!("Final pressure field:\n{:?}", solver.grid.pressure);
    println!("Final u-velocity field:\n{:?}", solver.grid.u);
    println!("Final v-velocity field:\n{:?}", solver.grid.v);

    Ok(())
}
