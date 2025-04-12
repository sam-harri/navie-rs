#![allow(dead_code)]
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GridError {
    #[error("Invalid grid size: {0}")]
    InvalidGridSize(String),
}

#[derive(Error, Debug, Clone)]
pub enum BoundaryError {
    #[error("Invalid boundary condition for {0}")]
    InvalidBoundaryCondition(String),
}

#[derive(Error, Debug)]
pub enum SolverError {
    #[error("Invalid grid dimensions: {0}")]
    InvalidGridDimensions(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Boundary condition error: {0}")]
    BoundaryConditionError(String),

    #[error("Poisson solver failed: {0}")]
    PoissonError(String),

    #[error("I/O error: {0}")]
    IoError(String),

    #[error("Unknown error: {0}")]
    Other(String),
}
