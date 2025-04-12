#![allow(dead_code)]

use crate::error::GridError;
use nalgebra::DMatrix;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridDimensions2D(pub usize, pub usize);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSize2D(pub f64, pub f64);

#[derive(Debug, Clone)]
pub struct Grid2D {
    pub dimensions: GridDimensions2D,
    pub cell_size: CellSize2D,
    pub pressure: DMatrix<f64>,
    pub u: DMatrix<f64>,
    pub v: DMatrix<f64>,
}

impl Grid2D {
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

    pub fn apply_lid_driven_bcs(&mut self, bctop: f64) {
        let GridDimensions2D(nx, ny) = self.dimensions;

        let u_nrows = nx + 1;
        let u_ncols = ny + 2;
        let v_nrows = nx + 2;
        let v_ncols = ny + 1;

        if u_nrows < 2 || u_ncols < 2 || v_nrows < 2 || v_ncols < 2 {
            return;
        }

        for c in 1..=ny {
            self.u[(0, c)] = 0.0;
        }

        self.u[(0, 0)] = 0.0;
        self.u[(0, ny + 1)] = 0.0;

        for c in 1..=ny {
            self.u[(nx, c)] = 0.0;
        }

        self.u[(nx, 0)] = 0.0;
        self.u[(nx, ny + 1)] = 0.0;

        for r in 1..nx {
            self.u[(r, 0)] = -self.u[(r, 1)];
        }

        self.u[(0, 0)] = 0.0;
        self.u[(nx, 0)] = 0.0;

        let last_u_col = ny + 1;
        let second_last_u_col = ny;
        for r in 1..nx {
            self.u[(r, last_u_col)] = 2.0 * bctop - self.u[(r, second_last_u_col)];
        }

        self.u[(0, last_u_col)] = 0.0;
        self.u[(nx, last_u_col)] = 0.0;

        for c in 1..ny {
            self.v[(0, c)] = -self.v[(1, c)];
        }

        self.v[(0, 0)] = 0.0;
        self.v[(0, ny)] = 0.0;

        let last_v_row = nx + 1;
        let second_last_v_row = nx;
        for c in 1..ny {
            self.v[(last_v_row, c)] = -self.v[(second_last_v_row, c)];
        }

        self.v[(last_v_row, 0)] = 0.0;
        self.v[(last_v_row, ny)] = 0.0;

        for r in 1..=nx {
            self.v[(r, 0)] = 0.0;
        }

        self.v[(0, 0)] = 0.0;
        self.v[(last_v_row, 0)] = 0.0;

        let last_v_col = ny;
        for r in 1..=nx {
            self.v[(r, last_v_col)] = 0.0;
        }

        self.v[(0, last_v_col)] = 0.0;
        self.v[(last_v_row, last_v_col)] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;

    #[test]
    fn test_grid_creation() {
        let dims = GridDimensions2D(5, 5);
        let cell_size = CellSize2D(0.1, 0.1);
        let grid = Grid2D::new(dims, cell_size).unwrap();
        assert_eq!(grid.dimensions, dims);
        assert_eq!(grid.cell_size, cell_size);
        assert_eq!(grid.pressure.nrows(), 5);
        assert_eq!(grid.pressure.ncols(), 5);
        assert_eq!(grid.u.nrows(), 6);
        assert_eq!(grid.u.ncols(), 7);
        assert_eq!(grid.v.nrows(), 7);
        assert_eq!(grid.v.ncols(), 6);
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
        let dims = GridDimensions2D(5, 5);
        let cell_size = CellSize2D(0.5, 0.5);
        let mut grid = Grid2D::new(dims, cell_size).unwrap();

        let bctop = 1.0;

        grid.u = dmatrix![
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        ];

        grid.v = dmatrix![
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        ];

        grid.apply_lid_driven_bcs(bctop);

        let expected_u = dmatrix![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5;
            -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5;
            -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5;
            -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5;
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ];

        let expected_v = dmatrix![
            0.0, -0.5,-0.5,-0.5,-0.5, 0.0;
            0.0, 0.5, 0.5, 0.5, 0.5, 0.0;
            0.0, 0.5, 0.5, 0.5, 0.5, 0.0;
            0.0, 0.5, 0.5, 0.5, 0.5, 0.0;
            0.0, 0.5, 0.5, 0.5, 0.5, 0.0;
            0.0, 0.5, 0.5, 0.5, 0.5, 0.0;
            0.0, -0.5,-0.5,-0.5,-0.5, 0.0;
        ];

        println!("Expected u:\n{}", expected_u);
        println!("Computed u:\n{}", grid.u);

        println!("Expected v:\n{}", expected_v);
        println!("Computed v:\n{}", grid.v);

        assert_relative_eq!(grid.u, expected_u, epsilon = 1e-9);
        assert_relative_eq!(grid.v, expected_v, epsilon = 1e-9);
    }
}
