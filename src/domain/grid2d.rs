#![allow(dead_code)] // Allow unused code for example purposes

use nalgebra::DMatrix; // Only need DMatrix
use crate::error::GridError;
use crate::boundary::bc2d::{BoundaryCondition, BoundaryConditions2D, FaceBoundary, SquareBoundary};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridDimensions2D(pub usize, pub usize); // nx, ny (interior cells)

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSize2D(pub f64, pub f64); // dx, dy

#[derive(Debug, Clone)]
pub struct Grid2D {
    pub dimensions: GridDimensions2D,
    pub cell_size: CellSize2D,
    pub pressure: DMatrix<f64>, // nx x ny
    pub u: DMatrix<f64>,        // (nx+1) x (ny+2)
    pub v: DMatrix<f64>,        // (nx+2) x (ny+1)
}

impl Grid2D {
    ///     →       →       →       →       →       →       →
    ///     |       |       |       |       |       |       |
    /// ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
    ///     |       |       |       |       |       |       |
    ///     →   •   →   •   →   •   →   •   →   •   →   •   →
    ///     |       |       |       |       |       |       |
    /// ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
    ///     |       |       |       |       |       |       |
    ///     →   •   →   •   →   •   →   •   →   •   →   •   →
    ///     |       |       |       |       |       |       |
    /// ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
    ///     |       |       |       |       |       |       |
    ///     →   •   →   •   →   •   →   •   →   •   →   •   →
    ///     |       |       |       |       |       |       |
    /// ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
    ///     |       |       |       |       |       |       |
    ///     →       →       →       →       →       →       →
    pub fn new(dimensions: GridDimensions2D, cell_size: CellSize2D) -> Result<Self, GridError> {
        let GridDimensions2D(nx, ny) = dimensions;
        if nx < 1 || ny < 1 {
             return Err(GridError::InvalidGridSize(
                 "Grid dimensions (nx, ny) must be at least 1x1 for interior cells.".to_string(),
             ));
        }
        Ok(Self {
            dimensions,
            cell_size,
            pressure: DMatrix::<f64>::zeros(nx, ny),
            u: DMatrix::<f64>::zeros(nx + 1, ny + 2),
            v: DMatrix::<f64>::zeros(nx + 2, ny + 1),
        })
    }

    /// Applies boundary conditions to the u and v fields by setting ghost cell values.
    /// Mirrors the specific sequential logic from the MATLAB example for lid-driven cavity.
    pub fn apply_boundary_conditions(&mut self, bcs: BoundaryConditions2D) {
        // Only handle Dirichlet BCs for velocity matching the MATLAB snippet provided
        Self::apply_velocity_bc(&mut self.u, bcs.u);
        Self::apply_velocity_bc(&mut self.v, bcs.v);
    }

    /// Applies velocity BCs mimicking the MATLAB script's sequential logic
    /// and specific Dirichlet conditions for the lid-driven cavity.
    /// NOTE: Uses Rust's 0-based indexing.
    pub fn apply_velocity_bc(field: &mut DMatrix<f64>, bc: SquareBoundary) {
        let nrows = field.nrows();
        let ncols = field.ncols();
        // Need at least 2 rows/cols for basic ghost+interior setup
        if nrows < 2 || ncols < 2 {
             eprintln!("Warning: Field dimensions ({nrows}x{ncols}) too small for BC application. Skipping.");
             return;
        }

        let FaceBoundary(x_lower_bc, x_upper_bc) = bc.x; // Left, Right
        let FaceBoundary(y_lower_bc, y_upper_bc) = bc.y; // Bottom, Top

        // Check that all boundaries are Dirichlet type
        match (x_lower_bc, x_upper_bc, y_lower_bc, y_upper_bc) {
            (BoundaryCondition::Dirichlet(left), BoundaryCondition::Dirichlet(right), 
             BoundaryCondition::Dirichlet(bottom), BoundaryCondition::Dirichlet(top)) => {
                // Apply Dirichlet BCs using the MATLAB approach where the average between the two ghost cells
                // equals the boundary condition value
                
                // Left boundary (x_lower)
                for i in 0..nrows {
                    field[(i, 0)] = 2.0 * left - field[(i, 1)];
                }
                
                // Right boundary (x_upper)
                for i in 0..nrows {
                    field[(i, ncols-1)] = 2.0 * right - field[(i, ncols-2)];
                }
                // Top boundary (y_upper)
                for j in 0..ncols {
                    field[(0, j)] = 2.0 * top - field[(1, j)];
                }
                
                // Bottom boundary (y_lower)
                for j in 0..ncols {
                    field[(nrows-1, j)] = 2.0 * bottom - field[(nrows-2, j)];
                }
                
            }
            _ => panic!("All boundary conditions must be Dirichlet type")
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use crate::boundary::bc2d::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_grid_creation() {
        let dims = GridDimensions2D(3, 6);
        let cell_size = CellSize2D(0.1, 0.1);
        let grid = Grid2D::new(dims, cell_size).unwrap();
        assert_eq!(grid.dimensions, dims);
        assert_eq!(grid.cell_size, cell_size);
        assert_eq!(grid.pressure.nrows(), 3); assert_eq!(grid.pressure.ncols(), 6);
        assert_eq!(grid.u.nrows(), 4); assert_eq!(grid.u.ncols(), 8); // (nx+1) x (ny+2) = 4x8
        assert_eq!(grid.v.nrows(), 5); assert_eq!(grid.v.ncols(), 7); // (nx+2) x (ny+1) = 5x7
    }
    
     #[test]
    fn test_grid_creation_invalid_size() {
        let dims = GridDimensions2D(0, 5);
        let cell_size = CellSize2D(0.1, 0.1);
        assert!(Grid2D::new(dims, cell_size).is_err());
        let dims = GridDimensions2D(5, 0);
        assert!(Grid2D::new(dims, cell_size).is_err());
    }
    
    #[test]
    fn test_apply_boundary_conditions_lid_driven() {
        let dims = GridDimensions2D(2, 2); // nx=2, ny=2
        let cell_size = CellSize2D(0.5, 0.5);
        let mut grid = Grid2D::new(dims, cell_size).unwrap();
    
        let bctop = 1.0;
        let u_bc = SquareBoundary {
            x: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(0.0)),
            y: FaceBoundary(BoundaryCondition::Dirichlet(0.0), BoundaryCondition::Dirichlet(bctop)),
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
    
        // Apply BCs using the MATLAB-style function
        Grid2D::apply_velocity_bc(&mut grid.u, bcs.u);
        Grid2D::apply_velocity_bc(&mut grid.v, bcs.v);

        // Based on the ASCII diagram, the top boundary should have a value of 1.0
        // and the ghost cells should be set to 2.0 to ensure the average is 1.0
        let expected_u = dmatrix![
            2.0, 2.0, 2.0, 2.0;
            0.0, 0.0, 0.0, 0.0;
            0.0, 0.0, 0.0, 0.0;
        ];
    
         let expected_v = dmatrix![
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0;
        ];
    
        println!("Expected u:\n{}", expected_u);
        println!("Computed u:\n{}", grid.u);

        println!("Expected v:\n{}", expected_v);
        println!("Computed v:\n{}", grid.v);

        assert_relative_eq!(grid.u, expected_u, epsilon = 1e-9);
        assert_relative_eq!(grid.v, expected_v, epsilon = 1e-9);
    }
}