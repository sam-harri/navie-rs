use crate::error::BoundaryError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    Dirichlet(f64), // Fixed value, e.g., u = 1.0
    Neumann,   // Fixed gradient at zero
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBoundary(pub BoundaryCondition, pub BoundaryCondition);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquareBoundary {
    pub x: FaceBoundary,
    pub y: FaceBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryConditions2D {
    pub u: SquareBoundary,
    pub v: SquareBoundary,
    pub p: SquareBoundary,
}

impl SquareBoundary {
    pub fn new(
        x: FaceBoundary,
        y: FaceBoundary,
    ) -> Result<Self, BoundaryError> {
        // valiation logic needs to be added
        Ok(Self { x, y})
    }
}

impl BoundaryConditions2D {
    pub fn new(
        u: SquareBoundary,
        v: SquareBoundary,
        p: SquareBoundary,
    ) -> Result<Self, BoundaryError> {
        // valiation logic needs to be added
        Ok(Self { u, v, p })
    }
}
