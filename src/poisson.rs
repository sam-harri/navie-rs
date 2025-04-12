use nalgebra::DMatrix;
use rsparse::data::{Sprs, Trpl};
use rsparse::lusol;

fn construct_poisson_matrix_sparse(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Result<Sprs<f64>, String> {
    let n = nx * ny;
    if n == 0 {
        return Ok(Sprs::<f64> {
            m: 0,
            n: 0,
            nzmax: 0,
            p: vec![0],
            i: vec![],
            x: vec![],
        });
    }
    let mut collected_triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(5 * n);

    let dx2 = dx * dx;
    let dy2 = dy * dy;
    if dx.abs() < 1e-15 || dy.abs() < 1e-15 {
        return Err("dx or dy is too small or zero.".to_string());
    }
    if dx2.abs() < 1e-15 || dy2.abs() < 1e-15 {
        return Err(
            "dx^2 or dy^2 is too small, leading to potential overflow/division issues".to_string(),
        );
    }
    let inv_dx2 = 1.0 / dx2;
    let inv_dy2 = 1.0 / dy2;

    for c in 0..ny {
        for r in 0..nx {
            let k = r + c * nx;

            if k == 0 {
                collected_triplets.push((0, 0, 1.0));
                continue;
            }

            let effective_diag_x = if r == 0 || r == nx - 1 { -1.0 } else { -2.0 };
            let effective_diag_y = if c == 0 || c == ny - 1 { -1.0 } else { -2.0 };
            let expected_diag = effective_diag_x * inv_dx2 + effective_diag_y * inv_dy2;
            collected_triplets.push((k, k, expected_diag));

            if r > 0 {
                collected_triplets.push((k, k - 1, inv_dx2));
            }
            if r < nx - 1 {
                collected_triplets.push((k, k + 1, inv_dx2));
            }
            if c > 0 {
                collected_triplets.push((k, k - nx, inv_dy2));
            }
            if c < ny - 1 {
                collected_triplets.push((k, k + nx, inv_dy2));
            }
        }
    }

    let num_triplets = collected_triplets.len();
    let mut trpl_mat = Trpl::<f64> {
        m: n,
        n: n,
        p: Vec::with_capacity(num_triplets),
        i: Vec::with_capacity(num_triplets),
        x: Vec::with_capacity(num_triplets),
    };

    for (row_idx, col_idx, value) in collected_triplets.iter() {
        trpl_mat.i.push(*row_idx);
        trpl_mat.p.push(*col_idx as isize);
        trpl_mat.x.push(*value);
    }

    if trpl_mat.i.len() != num_triplets
        || trpl_mat.p.len() != num_triplets
        || trpl_mat.x.len() != num_triplets
    {
        return Err("Failed to populate Triplet matrix vectors correctly".to_string());
    }

    let mut sprs_mat = Sprs::<f64>::new();
    sprs_mat.from_trpl(&trpl_mat);

    if sprs_mat.m != n || sprs_mat.n != n {
        return Err("Sprs matrix conversion failed (dimension mismatch)".to_string());
    }
    if n > 0 && sprs_mat.p.len() != n + 1 {
        return Err(format!(
            "Sprs matrix conversion failed (column pointer 'p' has wrong length: {} expected {})",
            sprs_mat.p.len(),
            n + 1
        ));
    }
    let expected_nnz = if n > 0 { sprs_mat.p[n] as usize } else { 0 };
    if sprs_mat.x.len() != expected_nnz || sprs_mat.i.len() != expected_nnz {
        println!("Warning: Mismatch between expected nnz ({}) and actual vector lengths (i: {}, x: {}) in Sprs after conversion. Duplicates might have been summed.", expected_nnz, sprs_mat.i.len(), sprs_mat.x.len());
    }

    Ok(sprs_mat)
}

pub fn solve_poisson_equation_direct(
    b: &DMatrix<f64>,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    verify: bool,
) -> Result<DMatrix<f64>, String> {
    if b.nrows() != nx || b.ncols() != ny {
        return Err(format!(
            "Input matrix b has dimensions {}x{}, expected {}x{}",
            b.nrows(),
            b.ncols(),
            nx,
            ny
        ));
    }
    if nx == 0 || ny == 0 {
        return Ok(DMatrix::<f64>::zeros(nx, ny));
    }

    let a_sparse = construct_poisson_matrix_sparse(nx, ny, dx, dy)?;

    let n = nx * ny;
    let mut f: Vec<f64> = vec![0.0; n];
    f.copy_from_slice(b.as_slice());

    if n > 0 {
        f[0] = 0.0;
    }

    match lusol(&a_sparse, &mut f, 1, 1e-10) {
        Ok(()) => {}
        Err(error_code) => {
            return Err(format!(
                "Sparse LU solver failed with error code: {}",
                error_code
            ));
        }
    }

    if verify {
        println!(
            "Verification requested. Note: Internal residual check requires sparse*dense multiply."
        );
        if n > 0 && f.iter().any(|&val| val.is_nan() || val.is_infinite()) {
            println!("Warning: Solution vector contains NaN or Inf values.");
        }
    }

    let dp = DMatrix::from_vec(nx, ny, f);

    Ok(dp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;

    #[test]
    fn test_solve_poisson_equation_direct() {
        let b = dmatrix![
            20.5000, -102.0000,  17.0000,  20.5000,  20.5000;
            -102.0000,   3.0000,  20.5000,  20.5000,  20.5000;
             20.5000,  17.0000,  20.5000,  20.5000,   3.0000;
              3.0000,  20.5000,  20.5000,  20.5000,  17.0000;
             17.0000,  20.5000,  20.5000,   3.0000, -102.0000
        ];

        let nx = 5;
        let ny = 5;
        let dx = 0.5;
        let dy = 2.0;

        let dp = solve_poisson_equation_direct(&b, nx, ny, dx, dy, true).unwrap();

        let expected_dp = dmatrix![
            0.000, -74.918, -219.534, -263.699, -248.442;
            -5.443, -96.062, -221.562, -262.288, -242.364;
            -30.722, -114.275, -223.763, -259.543, -229.915;
            -45.653, -126.618, -225.447, -255.760, -214.864;
            -54.775, -132.720, -226.287, -251.303, -193.008
        ];

        println!("dp: {}", dp);
        println!("expected_dp: {}", expected_dp);
        assert_eq!(dp.nrows(), 5);
        assert_eq!(dp.ncols(), 5);
        assert_relative_eq!(dp, expected_dp, epsilon = 1e-3);
    }

    #[test]
    fn test_larger_grid_performance() {
        let nx = 80;
        let ny = 80;
        let dx = 1.0 / (nx as f64);
        let dy = 1.0 / (ny as f64);

        let b = DMatrix::<f64>::from_element(nx, ny, 1.0);

        println!("Starting sparse solve for {}x{} grid...", nx, ny);
        let start = std::time::Instant::now();
        let result = solve_poisson_equation_direct(&b, nx, ny, dx, dy, false);
        let duration = start.elapsed();
        println!("Sparse solve for {}x{} took: {:?}", nx, ny, duration);

        assert!(
            result.is_ok(),
            "Solver failed for {}x{} grid: {:?}",
            nx,
            ny,
            result.err()
        );
        assert!(
            duration.as_secs_f64() < 10.0,
            "Solve took longer than expected limit (10s)"
        );

        if let Ok(dp) = result {
            assert_eq!(dp.nrows(), nx);
            assert_eq!(dp.ncols(), ny);
            if nx > 0 && ny > 0 {
                assert_relative_eq!(dp[(0, 0)], 0.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_zero_dimension_ok() {
        let b = DMatrix::<f64>::zeros(0, 5);
        let result = solve_poisson_equation_direct(&b, 0, 5, 1.0, 1.0, false);
        assert!(result.is_ok());
        let dp = result.unwrap();
        assert_eq!(dp.nrows(), 0);
        assert_eq!(dp.ncols(), 5);

        let b2 = DMatrix::<f64>::zeros(5, 0);
        let result2 = solve_poisson_equation_direct(&b2, 5, 0, 1.0, 1.0, false);
        assert!(result2.is_ok());
        let dp2 = result2.unwrap();
        assert_eq!(dp2.nrows(), 5);
        assert_eq!(dp2.ncols(), 0);

        let b3 = DMatrix::<f64>::zeros(0, 0);
        let result3 = solve_poisson_equation_direct(&b3, 0, 0, 1.0, 1.0, false);
        assert!(result3.is_ok());
        let dp3 = result3.unwrap();
        assert_eq!(dp3.nrows(), 0);
        assert_eq!(dp3.ncols(), 0);
        assert_eq!(dp3.len(), 0);
    }

    #[test]
    fn test_invalid_step_size() {
        let b = DMatrix::<f64>::zeros(2, 2);
        let result = solve_poisson_equation_direct(&b, 2, 2, 0.0, 1.0, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dx or dy is too small"));

        let result2 = solve_poisson_equation_direct(&b, 2, 2, 1.0, 0.0, false);
        assert!(result2.is_err());
        assert!(result2.unwrap_err().contains("dx or dy is too small"));
    }
}
