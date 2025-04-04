use ndarray::s;
use crate::domain::grid2d::Grid2D;
use crate::numerical::derive::{
    central_difference, face_difference, derivative_same_shape, DerivativeDirection
};
use crate::numerical::interpolate::{interpolate_simple, InterpAxis};
use crate::boundary::bc2d::BoundaryConditions2D;
use crate::error::SolverError;

pub struct SimpleSolver {
    pub grid: Grid2D,
    pub dt: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub alpha: f64, // under-relaxation factor for pressure correction
    pub bcs: BoundaryConditions2D,
    pub rho: f64,   // density [kg/m^3]
    pub nu: f64,    // kinematic viscosity [m^2/s]
}

impl SimpleSolver {
    pub fn new(
        grid: Grid2D,
        dt: f64,
        max_iterations: usize,
        tolerance: f64,
        alpha: f64,
        bcs: BoundaryConditions2D,
        rho: f64,
        nu: f64,
    ) -> Result<Self, SolverError> {
        Ok(Self {
            grid,
            dt,
            max_iterations,
            tolerance,
            alpha,
            bcs,
            rho,
            nu,
        })
    }

    pub fn solve_timestep(&mut self) {

        for iter in 0..self.max_iterations {
            
        }
            
    }

    pub fn solve(&mut self, steps: usize) {
        for step in 0..steps {
            println!("Time step {}", step);
            self.solve_timestep();
            self.grid.apply_boundary_conditions(self.bcs.clone());
        }
    }
}
