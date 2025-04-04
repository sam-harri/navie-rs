use std::mem;

use ndarray::{Array1, Array2, ArrayView1};
use crate::error::GridError;
use crate::boundary::bc2d::{BoundaryCondition, BoundaryConditions2D, FaceBoundary, SquareBoundary};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridDimensions2D(pub usize, pub usize);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSize2D(pub f64, pub f64); 

#[derive(Debug, Clone)]
pub struct Grid2D {
    pub dimensions: GridDimensions2D,
    pub cell_size: CellSize2D,
    pub pressure: Array2<f64>,
    pub u: Array2<f64>,
    pub v: Array2<f64>,
}

impl Grid2D {
    /// Create a new computational grid wth 3 cells by 6 cells
    /// the pressure field is 5x8
    /// the u field is 5x7
    /// the v field is 4x8
    /// •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
    ///     |       |       |       |       |       |       |    
    /// ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
    ///     |       |       |       |       |       |       |    
    /// •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
    ///     |       |       |       |       |       |       |    
    /// ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
    ///     |       |       |       |       |       |       |    
    /// •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
    ///     |       |       |       |       |       |       |    
    /// ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
    ///     |       |       |       |       |       |       |    
    /// •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
    ///     |       |       |       |       |       |       |    
    /// ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
    ///     |       |       |       |       |       |       |    
    /// •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
    pub fn new(dimensions: GridDimensions2D, cell_size: CellSize2D) -> Result<Self, GridError> {
        let GridDimensions2D(nx, ny) = dimensions;

        if nx <= 2 || ny <= 2 {
            return Err(GridError::InvalidGridSize(
                "Grid dimensions must be greater than two.".to_string(),
            ));
        }


        Ok(Self {
            dimensions,
            cell_size,
            pressure: Array2::<f64>::zeros((nx + 2, ny + 2)),
            u: Array2::<f64>::zeros((nx + 1, ny + 2)), // Staggered in x
            v: Array2::<f64>::zeros((nx + 2, ny + 1)), // Staggered in y
        })
    }

    pub fn apply_boundary_conditions(&mut self, bcs: BoundaryConditions2D) {
        let BoundaryConditions2D { u, v, p } = bcs;
        self.u = Self::bc(mem::take(&mut self.u), u);
        self.v = Self::bc(mem::take(&mut self.v), v);
        self.pressure = Self::bc(mem::take(&mut self.pressure), p);
    }

    pub fn bc(mut domain: Array2<f64>, bc: SquareBoundary) -> Array2<f64> {
        let FaceBoundary(x_lower, x_upper) = bc.x;
        let FaceBoundary(y_lower, y_upper) = bc.y;

        let nrows = domain.nrows();
        let ncols = domain.ncols();

        let new_top = Self::replacement_vec(y_upper, Some(domain.row(1)), ncols);
        let new_bottom = Self::replacement_vec(y_lower, Some(domain.row(nrows - 2)), ncols);
        let new_left = Self::replacement_vec(x_lower, Some(domain.column(1)), nrows);
        let new_right = Self::replacement_vec(x_upper, Some(domain.column(ncols - 2)), nrows);

        domain.row_mut(0).assign(&new_top);
        domain.row_mut(nrows - 1).assign(&new_bottom);
        domain.column_mut(0).assign(&new_left);
        domain.column_mut(ncols - 1).assign(&new_right);

        domain[[0, 0]] = 0.5 * (domain[[1, 0]] + domain[[0, 1]]);
        domain[[0, ncols - 1]] = 0.5 * (domain[[0, ncols - 2]] + domain[[1, ncols - 1]]);
        domain[[nrows - 1, 0]] = 0.5 * (domain[[nrows - 2, 0]] + domain[[nrows - 1, 1]]);
        domain[[nrows - 1, ncols - 1]] = 0.5 * (domain[[nrows - 2, ncols - 1]] + domain[[nrows - 1, ncols - 2]]);

        domain
    }

    pub fn replacement_vec(
        bc: BoundaryCondition,
        interior: Option<ArrayView1<f64>>,
        size: usize,
    ) -> Array1<f64> {
        match bc {
            BoundaryCondition::Dirichlet(scalar) => Array1::from_elem(size, scalar),
            BoundaryCondition::Neumann => {
                interior
                    .expect("Neumann boundary condition requires an interior row/column. Typically caused by a domain containing 2 cells or less in the direction of the Neuman Boundary")
                    .to_owned()
            }
        }
    }

    
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_bc_all_dirichlet() {
        let domain = Array2::<f64>::from_shape_vec((3, 3), vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]).unwrap();

        let bc = SquareBoundary {
            x: FaceBoundary(BoundaryCondition::Dirichlet(20.0), BoundaryCondition::Dirichlet(40.0)),
            y: FaceBoundary(BoundaryCondition::Dirichlet(10.0), BoundaryCondition::Dirichlet(30.0)),
        };

        let updated = Grid2D::bc(domain, bc);

        let expected = Array2::<f64>::from_shape_vec((3, 3), vec![
            25.0, 30.0, 35.0,
            20.0, 0.0, 40.0,
            15.0, 10.0, 25.0,
        ]).unwrap();
        assert_eq!(updated, expected);
    }

    #[test]
    fn test_bc_all_neumann() {
        let domain = Array2::<f64>::from_shape_vec((4, 4), vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 2.0, 0.0,
            0.0, 4.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]).unwrap();

        let bc = SquareBoundary {
            x: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
            y: FaceBoundary(BoundaryCondition::Neumann, BoundaryCondition::Neumann),
        };

        let updated = Grid2D::bc(domain, bc);

        let expected = Array2::<f64>::from_shape_vec((4, 4), vec![
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
        ]).unwrap();
        assert_eq!(updated, expected);
    }
}
    
