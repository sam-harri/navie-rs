#![allow(dead_code)] // Allow unused code for example purposes

use nalgebra::DMatrix; // Only need DMatrix
use crate::error::GridError;

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

    pub fn apply_lid_driven_bcs(&mut self, bctop: f64) {
        let GridDimensions2D(nx, ny) = self.dimensions;

        // Dimensions including ghost cells
        let u_nrows = nx + 1; // 6 (Indices 0..=5)
        let u_ncols = ny + 2; // 7 (Indices 0..=6)
        let v_nrows = nx + 2; // 7 (Indices 0..=6)
        let v_ncols = ny + 1; // 6 (Indices 0..=5)

        if u_nrows < 2 || u_ncols < 2 || v_nrows < 2 || v_ncols < 2 { return; }

        // --- Apply BCs to u field (horizontal velocity, 6x7) ---
        // Indices: u[x_idx, y_idx] where x_idx = 0..nx, y_idx = 0..ny+1

        // Left Wall (x=0): u=0 (no penetration). Set boundary u values.
        // Corresponds to u[0, c] for interior y points c=1..ny
        for c in 1..=ny { self.u[(0, c)] = 0.0; }
        // Also set ghost corners
        self.u[(0, 0)] = 0.0;      // Bottom-left ghost
        self.u[(0, ny + 1)] = 0.0; // Top-left ghost

        // Right Wall (x=Lx): u=0 (no penetration). Set boundary u values.
        // Corresponds to u[nx, c] for interior y points c=1..ny
        for c in 1..=ny { self.u[(nx, c)] = 0.0; }
        // Also set ghost corners
        self.u[(nx, 0)] = 0.0;      // Bottom-right ghost
        self.u[(nx, ny + 1)] = 0.0; // Top-right ghost

        // Bottom Wall (y=0): u=0 (no slip). Set ghost column 0.
        // u[r, 0] = -u[r, 1] for interior x points r=1..nx-1
        for r in 1..nx { // r = 1..4 for nx=5
             self.u[(r, 0)] = -self.u[(r, 1)];
        }
        // Use wall conditions for corners
        self.u[(0, 0)] = 0.0; // Already set by left wall
        self.u[(nx, 0)] = 0.0; // Already set by right wall


        // Top Wall (Lid, y=Ly): u=bctop (tangential velocity). Set ghost column ny+1.
        // u[r, ny+1] = 2*bctop - u[r, ny] for interior x points r=1..nx-1
        let last_u_col = ny + 1;
        let second_last_u_col = ny;
        for r in 1..nx { // r = 1..4 for nx=5
            self.u[(r, last_u_col)] = 2.0 * bctop - self.u[(r, second_last_u_col)];
        }
         // Use wall conditions for corners
        self.u[(0, last_u_col)] = 0.0;       // Already set by left wall
        self.u[(nx, last_u_col)] = 0.0;       // Already set by right wall


        // --- Apply BCs to v field (vertical velocity, 7x6) ---
        // Indices: v[x_idx, y_idx] where x_idx = 0..nx+1, y_idx = 0..ny

        // Left Wall (x=0): v=0 (no slip). Set ghost row 0.
        // v[0, c] = -v[1, c] for interior y points c=1..ny-1
        for c in 1..ny { // c = 1..4 for ny=5
            self.v[(0, c)] = -self.v[(1, c)];
        }
        // Use wall conditions for corners
        self.v[(0, 0)] = 0.0;       // Consistent with Bottom wall v=0
        self.v[(0, ny)] = 0.0;       // Consistent with Top wall v=0


        // Right Wall (x=Lx): v=0 (no slip). Set ghost row nx+1.
        // v[nx+1, c] = -v[nx, c] for interior y points c=1..ny-1
        let last_v_row = nx + 1;
        let second_last_v_row = nx;
        for c in 1..ny { // c = 1..4 for ny=5
            self.v[(last_v_row, c)] = -self.v[(second_last_v_row, c)];
        }
        // Use wall conditions for corners
        self.v[(last_v_row, 0)] = 0.0; // Consistent with Bottom wall v=0
        self.v[(last_v_row, ny)] = 0.0; // Consistent with Top wall v=0


        // Bottom Wall (y=0): v=0 (no penetration). Set boundary v values.
        // Corresponds to v[r, 0] for interior x points r=1..nx
        for r in 1..=nx { self.v[(r, 0)] = 0.0; }
        // Also set ghost corners
        self.v[(0, 0)] = 0.0;          // Already set by left wall
        self.v[(last_v_row, 0)] = 0.0; // Already set by right wall


        // Top Wall (Lid, y=Ly): v=0 (no penetration). Set boundary v values.
        // Corresponds to v[r, ny] for interior x points r=1..nx
        let last_v_col = ny;
        for r in 1..=nx { self.v[(r, last_v_col)] = 0.0; }
         // Also set ghost corners
        self.v[(0, last_v_col)] = 0.0;          // Already set by left wall
        self.v[(last_v_row, last_v_col)] = 0.0; // Already set by right wall
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use approx::assert_relative_eq;

    #[test]
    fn test_grid_creation() {
        let dims = GridDimensions2D(5, 5);
        let cell_size = CellSize2D(0.1, 0.1);
        let grid = Grid2D::new(dims, cell_size).unwrap();
        assert_eq!(grid.dimensions, dims);
        assert_eq!(grid.cell_size, cell_size);
        assert_eq!(grid.pressure.nrows(), 5); assert_eq!(grid.pressure.ncols(), 5);
        assert_eq!(grid.u.nrows(), 6); assert_eq!(grid.u.ncols(), 7); // (nx+1) x (ny+2) = 4x8
        assert_eq!(grid.v.nrows(), 7); assert_eq!(grid.v.ncols(), 6); // (nx+2) x (ny+1) = 5x7
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

        // Based on the ASCII diagram, the top boundary should have a value of 1.0
        // and the ghost cells should be set to 2.0 to ensure the average is 1.0
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