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

#[derive(Error, Debug, Clone)]
pub enum SolverError {

}