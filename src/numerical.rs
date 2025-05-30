#![allow(dead_code)]

use nalgebra::DMatrix;

pub fn interpolate_u_to_cell_centers(u: &DMatrix<f64>) -> DMatrix<f64> {
    let target_nx = u.nrows() - 1;
    let target_ny = u.ncols() - 2;
    if target_nx == 0 || target_ny == 0 {
        return DMatrix::<f64>::zeros(target_nx, target_ny);
    }

    let mut uce = DMatrix::<f64>::zeros(target_nx, target_ny);
    for r in 0..target_nx {
        for c in 0..target_ny {
            uce[(r, c)] = 0.5 * (u[(r, c + 1)] + u[(r + 1, c + 1)]);
        }
    }
    uce
}

pub fn interpolate_v_to_cell_centers(v: &DMatrix<f64>) -> DMatrix<f64> {
    let target_nx = v.nrows() - 2;
    let target_ny = v.ncols() - 1;
    if target_nx == 0 || target_ny == 0 {
        return DMatrix::<f64>::zeros(target_nx, target_ny);
    }

    let mut vce = DMatrix::<f64>::zeros(target_nx, target_ny);
    for r in 0..target_nx {
        for c in 0..target_ny {
            vce[(r, c)] = 0.5 * (v[(r + 1, c)] + v[(r + 1, c + 1)]);
        }
    }
    vce
}

pub fn interpolate_u_to_cell_edges(u: &DMatrix<f64>) -> DMatrix<f64> {
    let target_nx = u.nrows();
    let target_ny = u.ncols() - 1;
    if target_nx == 0 || target_ny == 0 {
        return DMatrix::<f64>::zeros(target_nx, target_ny);
    }

    let mut uco = DMatrix::<f64>::zeros(target_nx, target_ny);
    for r in 0..target_nx {
        for c in 0..target_ny {
            uco[(r, c)] = 0.5 * (u[(r, c)] + u[(r, c + 1)]);
        }
    }
    uco
}

pub fn interpolate_v_to_cell_edges(v: &DMatrix<f64>) -> DMatrix<f64> {
    let target_nx = v.nrows() - 1;
    let target_ny = v.ncols();
    if target_nx == 0 || target_ny == 0 {
        return DMatrix::<f64>::zeros(target_nx, target_ny);
    }

    let mut vco = DMatrix::<f64>::zeros(target_nx, target_ny);
    for r in 0..target_nx {
        for c in 0..target_ny {
            vco[(r, c)] = 0.5 * (v[(r, c)] + v[(r + 1, c)]);
        }
    }
    vco
}

pub fn calculate_viscous_x(u: &DMatrix<f64>, dx: f64) -> DMatrix<f64> {
    let nx = u.nrows() - 2;
    let ny = u.ncols() - 2;
    if nx == 0 || ny == 0 {
        return DMatrix::<f64>::zeros(nx, ny);
    }

    let mut lux = DMatrix::<f64>::zeros(nx, ny);
    for r in 0..nx {
        for c in 0..ny {
            lux[(r, c)] = (u[(r, c + 1)] - 2.0 * u[(r + 1, c + 1)] + u[(r + 2, c + 1)]) / (dx * dx);
        }
    }
    lux
}

pub fn calculate_viscous_y(u: &DMatrix<f64>, dy: f64) -> DMatrix<f64> {
    let nx = u.nrows() - 2;
    let ny = u.ncols() - 2;
    if nx == 0 || ny == 0 {
        return DMatrix::<f64>::zeros(nx, ny);
    }

    let mut luy = DMatrix::<f64>::zeros(nx, ny);
    for r in 0..nx {
        for c in 0..ny {
            luy[(r, c)] = (u[(r + 1, c)] - 2.0 * u[(r + 1, c + 1)] + u[(r + 1, c + 2)]) / (dy * dy);
        }
    }
    luy
}

pub fn calculate_viscous_terms(
    u: &DMatrix<f64>,
    v: &DMatrix<f64>,
    dx: f64,
    dy: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let lux = calculate_viscous_x(u, dx);
    let luy = calculate_viscous_y(u, dy);
    let lu = lux + luy;

    let lvx = calculate_viscous_x(v, dx);
    let lvy = calculate_viscous_y(v, dy);
    let lv = lvx + lvy;

    (lu, lv)
}

pub fn element_wise_multiply(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    assert_eq!(
        a.nrows(),
        b.nrows(),
        "Matrices must have the same number of rows for element-wise multiplication"
    );
    assert_eq!(
        a.ncols(),
        b.ncols(),
        "Matrices must have the same number of columns for element-wise multiplication"
    );

    let mut result = DMatrix::<f64>::zeros(a.nrows(), a.ncols());
    for r in 0..a.nrows() {
        for c in 0..a.ncols() {
            result[(r, c)] = a[(r, c)] * b[(r, c)];
        }
    }
    result
}

pub fn calculate_divergence(u: &DMatrix<f64>, v: &DMatrix<f64>, dx: f64, dy: f64) -> DMatrix<f64> {
    let n_true = u.nrows() - 1;
    let m_true = u.ncols() - 2;
    assert_eq!(
        n_true,
        v.nrows() - 2,
        "Inconsistent N derived from u and v shapes"
    );
    assert_eq!(
        m_true,
        v.ncols() - 1,
        "Inconsistent M derived from u and v shapes"
    );
    if n_true == 0 || m_true == 0 {
        return DMatrix::zeros(n_true, m_true);
    }
    let u_right_div = u.view((1, 1), (n_true, m_true));
    let u_left_div = u.view((0, 1), (n_true, m_true));
    let dudx = (u_right_div - u_left_div) / dx;
    let v_top_div = v.view((1, 1), (n_true, m_true));
    let v_bottom_div = v.view((1, 0), (n_true, m_true));
    let dvdy = (v_top_div - v_bottom_div) / dy;
    dudx + dvdy
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
    println!("shape of b: {:?}", b.shape());

    let mut abig = construct_poisson_matrix(nx, ny, dx, dy);

    for j in 0..nx * ny {
        abig[(0, j)] = 0.0;
    }
    abig[(0, 0)] = 1.0;

    let mut f = DMatrix::<f64>::zeros(nx * ny, 1);
    for r in 0..nx {
        for c in 0..ny {
            f[(r + c * nx, 0)] = b[(r, c)];
        }
    }

    f[(0, 0)] = 0.0;

    let abig_for_verify = if verify { Some(abig.clone()) } else { None };

    let lu = abig.lu();

    if verify {
        if !lu.is_invertible() {
            println!("Warning: The coefficient matrix is not invertible.");
        }

        let det = lu.determinant();
        println!(
            ">>> LU Check: Determinant of the coefficient matrix = {:.6e}",
            det
        );
        if det.abs() < 1e-10 {
            println!(
                "Warning: The coefficient matrix is nearly singular (determinant = {:.6e})",
                det
            );
        }
    }

    let dp_vec = match lu.solve(&f) {
        Some(solution) => solution,
        None => return Err("Failed to solve the linear system".to_string()),
    };

    if verify {
        if let Some(abig) = abig_for_verify {
            let residual = &abig * &dp_vec - &f;
            let norm_residual = residual.norm();
            println!(
                ">>> Solution Check: Norm of residual (A*dp - f) = {:.6e}",
                norm_residual
            );
            if norm_residual > 1e-8 {
                println!(
                    "Warning: Solution residual norm is high ({:.6e}), result might be inaccurate.",
                    norm_residual
                );
            }
        }
    }

    let mut dp = DMatrix::<f64>::zeros(nx, ny);
    for r in 0..nx {
        for c in 0..ny {
            dp[(r, c)] = dp_vec[(r + c * nx, 0)];
        }
    }

    Ok(dp)
}

fn construct_poisson_matrix(nx: usize, ny: usize, dx: f64, dy: f64) -> DMatrix<f64> {
    let n = nx * ny;
    let mut abig = DMatrix::<f64>::zeros(n, n);

    let mut tmp = vec![-2.0; nx];
    tmp[0] = -1.0;
    tmp[nx - 1] = -1.0;

    let mut ax = DMatrix::<f64>::zeros(nx, nx);
    for i in 0..nx {
        ax[(i, i)] = tmp[i];
        if i > 0 {
            ax[(i, i - 1)] = 1.0;
        }
        if i < nx - 1 {
            ax[(i, i + 1)] = 1.0;
        }
    }

    ax /= dx * dx;

    let id = DMatrix::<f64>::identity(nx, nx);

    for i in 0..ny {
        let start_row = i * nx;
        let start_col = i * nx;

        let scale = if i == 0 || i == ny - 1 { 1.0 } else { 2.0 };
        let block = &ax - &(scale * &id / (dy * dy));

        for r in 0..nx {
            for c in 0..nx {
                abig[(start_row + r, start_col + c)] = block[(r, c)];
            }
        }
    }

    if ny > 1 {
        for i in 0..ny - 1 {
            let start_row = i * nx;
            let start_col = (i + 1) * nx;

            for r in 0..nx {
                abig[(start_row + r, start_col + r)] = 1.0 / (dy * dy);
                abig[(start_col + r, start_row + r)] = 1.0 / (dy * dy);
            }
        }
    }

    abig
}

pub fn calculate_nu(uuce: &DMatrix<f64>, uvco: &DMatrix<f64>, dx: f64, dy: f64) -> DMatrix<f64> {
    let mut nu = DMatrix::<f64>::zeros(uuce.nrows() - 1, uuce.ncols());
    for r in 0..nu.nrows() {
        for c in 0..nu.ncols() {
            nu[(r, c)] = (uuce[(r + 1, c)] - uuce[(r, c)]) / dx;
        }
    }

    let mut nu_y = DMatrix::<f64>::zeros(nu.nrows(), nu.ncols());
    for r in 0..nu_y.nrows() {
        for c in 0..nu_y.ncols() {
            nu_y[(r, c)] = (uvco[(r + 1, c + 1)] - uvco[(r + 1, c)]) / dy;
        }
    }

    nu + nu_y
}

pub fn calculate_nv(vvce: &DMatrix<f64>, uvco: &DMatrix<f64>, dx: f64, dy: f64) -> DMatrix<f64> {
    let mut nv = DMatrix::<f64>::zeros(vvce.nrows(), vvce.ncols() - 1);
    for r in 0..nv.nrows() {
        for c in 0..nv.ncols() {
            nv[(r, c)] = (vvce[(r, c + 1)] - vvce[(r, c)]) / dy;
        }
    }

    let mut nv_x = DMatrix::<f64>::zeros(nv.nrows(), nv.ncols());
    for r in 0..nv_x.nrows() {
        for c in 0..nv_x.ncols() {
            nv_x[(r, c)] = (uvco[(r + 1, c + 1)] - uvco[(r, c + 1)]) / dx;
        }
    }

    nv + nv_x
}

pub fn update_velocity_field(
    velocity: &mut DMatrix<f64>,
    convective_term: &DMatrix<f64>,
    viscous_x: &DMatrix<f64>,
    viscous_y: &DMatrix<f64>,
    dt: f64,
    re: f64,
) {
    let conv_rows = convective_term.nrows();
    let conv_cols = convective_term.ncols();

    assert_eq!(
        viscous_x.nrows(),
        conv_rows,
        "Viscous x term has different number of rows than convective term"
    );
    assert_eq!(
        viscous_x.ncols(),
        conv_cols,
        "Viscous x term has different number of columns than convective term"
    );
    assert_eq!(
        viscous_y.nrows(),
        conv_rows,
        "Viscous y term has different number of rows than convective term"
    );
    assert_eq!(
        viscous_y.ncols(),
        conv_cols,
        "Viscous y term has different number of columns than convective term"
    );

    for r in 0..conv_rows {
        for c in 0..conv_cols {
            let vel_r = r + 1;
            let vel_c = c + 1;

            velocity[(vel_r, vel_c)] +=
                dt * (-convective_term[(r, c)] + (viscous_x[(r, c)] + viscous_y[(r, c)]) / re);
        }
    }
}

pub fn calculate_divergence_term(
    u: &DMatrix<f64>,
    v: &DMatrix<f64>,
    dx: f64,
    dy: f64,
) -> DMatrix<f64> {
    let nx = u.nrows() - 1;
    let ny = u.ncols() - 2;

    assert_eq!(v.nrows() - 2, nx, "Incompatible dimensions between u and v");
    assert_eq!(v.ncols() - 1, ny, "Incompatible dimensions between u and v");

    let mut b = DMatrix::<f64>::zeros(nx, ny);

    for r in 0..nx {
        for c in 0..ny {
            let component1 = (u[(r + 1, c + 1)] - u[(r, c + 1)]) / dx;
            let component2 = (v[(r + 1, c + 1)] - v[(r + 1, c)]) / dy;
            b[(r, c)] = component1 + component2;
        }
    }

    b
}

pub fn apply_pressure_corrections(
    u: &mut DMatrix<f64>,
    v: &mut DMatrix<f64>,
    p: &DMatrix<f64>,
    dx: f64,
    dy: f64,
) {
    let p_rows = p.nrows();
    let p_cols = p.ncols();

    let u_rows_loop_end = u.nrows() - 1;
    let u_cols_loop_end = u.ncols() - 1;
    for r in 1..u_rows_loop_end {
        for c in 1..u_cols_loop_end {
            let p_idx_r = r;
            let p_idx_r_minus_1 = r - 1;
            let p_idx_c = c - 1;

            if p_idx_r < p_rows && p_idx_r_minus_1 < p_rows && p_idx_c < p_cols {
                let dp_dx = (p[(p_idx_r, p_idx_c)] - p[(p_idx_r_minus_1, p_idx_c)]) / dx;
                u[(r, c)] -= dp_dx;
            }
        }
    }

    let v_rows_loop_end = v.nrows() - 1;
    let v_cols_loop_end = v.ncols() - 1;
    for r in 1..v_rows_loop_end {
        for c in 1..v_cols_loop_end {
            let p_idx_r = r - 1;
            let p_idx_c = c;
            let p_idx_c_minus_1 = c - 1;

            if p_idx_r < p_rows && p_idx_c < p_cols && p_idx_c_minus_1 < p_cols {
                let dp_dy = (p[(p_idx_r, p_idx_c)] - p[(p_idx_r, p_idx_c_minus_1)]) / dy;
                v[(r, c)] -= dp_dy;
            }
        }
    }
}

pub fn calculate_intermediate_velocity(
    u_old: &DMatrix<f64>,
    v_old: &DMatrix<f64>,
    nu: &DMatrix<f64>,
    nv: &DMatrix<f64>,
    lux: &DMatrix<f64>,
    luy: &DMatrix<f64>,
    lvx: &DMatrix<f64>,
    lvy: &DMatrix<f64>,
    dt: f64,
    re: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut u_star = u_old.clone();
    let mut v_star = v_old.clone();

    for r in 0..nu.nrows() {
        for c in 0..nu.ncols() {
            u_star[(r + 1, c + 1)] =
                u_old[(r + 1, c + 1)] + dt * (-nu[(r, c)] + (lux[(r, c)] + luy[(r, c)]) / re);
        }
    }
    for r in 0..nv.nrows() {
        for c in 0..nv.ncols() {
            v_star[(r + 1, c + 1)] =
                v_old[(r + 1, c + 1)] + dt * (-nv[(r, c)] + (lvx[(r, c)] + lvy[(r, c)]) / re);
        }
    }
    (u_star, v_star)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;

    #[test]
    fn test_interpolate_u_to_cell_centers() {
        let u = dmatrix![
            17.0, 24.0,  1.0,  8.0, 15.0;
            23.0,  5.0,  7.0, 14.0, 16.0;
             4.0,  6.0, 13.0, 20.0, 22.0;
            10.0, 12.0, 19.0, 21.0,  3.0;
            11.0, 18.0, 25.0,  2.0,  9.0
        ];

        let uce = interpolate_u_to_cell_centers(&u);

        let expected_uce = dmatrix![
            14.5,  4.0, 11.0;
             5.5, 10.0, 17.0;
             9.0, 16.0, 20.5;
            15.0, 22.0, 11.5
        ];

        println!("uce: {}", uce);
        println!("expected_uce: {}", expected_uce);
        assert_eq!(uce.nrows(), 4);
        assert_eq!(uce.ncols(), 3);
        assert_relative_eq!(uce, expected_uce, epsilon = 1e-9);
    }

    #[test]
    fn test_interpolate_v_to_cell_centers() {
        let v = dmatrix![
             1.0,  2.0,  3.0,  4.0,  5.0;
             6.0,  7.0,  8.0,  9.0, 10.0;
            11.0, 12.0, 13.0, 14.0, 15.0;
            16.0, 17.0, 18.0, 19.0, 20.0;
            21.0, 22.0, 23.0, 24.0, 25.0
        ];

        let vce = interpolate_v_to_cell_centers(&v);

        let expected_vce = dmatrix![
             6.5,  7.5,  8.5,  9.5;
            11.5, 12.5, 13.5, 14.5;
            16.5, 17.5, 18.5, 19.5
        ];

        println!("vce: {}", vce);
        println!("expected_vce: {}", expected_vce);
        assert_eq!(vce.nrows(), 3);
        assert_eq!(vce.ncols(), 4);
        assert_relative_eq!(vce, expected_vce, epsilon = 1e-9);
    }

    #[test]
    fn test_interpolate_u_to_cell_edges() {
        let u = dmatrix![
            17.0, 24.0,  1.0,  8.0, 15.0;
            23.0,  5.0,  7.0, 14.0, 16.0;
             4.0,  6.0, 13.0, 20.0, 22.0;
            10.0, 12.0, 19.0, 21.0,  3.0;
            11.0, 18.0, 25.0,  2.0,  9.0
        ];

        let uco = interpolate_u_to_cell_edges(&u);

        let expected_uco = dmatrix![
            20.5, 12.5,  4.5, 11.5;
            14.0,  6.0, 10.5, 15.0;
             5.0,  9.5, 16.5, 21.0;
            11.0, 15.5, 20.0, 12.0;
            14.5, 21.5, 13.5,  5.5
        ];

        println!("uco: {}", uco);
        println!("expected_uco: {}", expected_uco);
        assert_eq!(uco.nrows(), 5);
        assert_eq!(uco.ncols(), 4);
        assert_relative_eq!(uco, expected_uco, epsilon = 1e-9);
    }

    #[test]
    fn test_interpolate_v_to_cell_edges() {
        let v = dmatrix![
             1.0,  2.0,  3.0,  4.0,  5.0;
             6.0,  7.0,  8.0,  9.0, 10.0;
            11.0, 12.0, 13.0, 14.0, 15.0;
            16.0, 17.0, 18.0, 19.0, 20.0;
            21.0, 22.0, 23.0, 24.0, 25.0
        ];

        let vco = interpolate_v_to_cell_edges(&v);

        let expected_vco = dmatrix![
             3.5,  4.5,  5.5,  6.5,  7.5;
             8.5,  9.5, 10.5, 11.5, 12.5;
            13.5, 14.5, 15.5, 16.5, 17.5;
            18.5, 19.5, 20.5, 21.5, 22.5
        ];

        println!("vco: {}", vco);
        println!("expected_vco: {}", expected_vco);
        assert_eq!(vco.nrows(), 4);
        assert_eq!(vco.ncols(), 5);
        assert_relative_eq!(vco, expected_vco, epsilon = 1e-9);
    }

    #[test]
    fn test_calculate_viscous_x() {
        let u = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0, 29.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0, 37.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0, 45.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0,  4.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0;
            22.0, 31.0, 40.0, 49.0,  2.0, 11.0, 20.0
        ];

        let lux = calculate_viscous_x(&u, 1.0);

        let expected_lux = dmatrix![
            -49.0,  42.0,   0.0,   0.0,   0.0;
             49.0,   7.0,   0.0,   0.0,  -7.0;
             -7.0,   0.0,   0.0,   0.0,   7.0;
              7.0,   0.0,   0.0,  -7.0, -49.0;
              0.0,   0.0,   0.0, -42.0,  49.0
        ];

        println!("lux: {}", lux);
        println!("expected_lux: {}", expected_lux);
        assert_eq!(lux.nrows(), 5);
        assert_eq!(lux.ncols(), 5);
        assert_relative_eq!(lux, expected_lux, epsilon = 1e-9);
    }

    #[test]
    fn test_calculate_viscous_y() {
        let u = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0, 29.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0, 37.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0, 45.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0,  4.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0;
            22.0, 31.0, 40.0, 49.0,  2.0, 11.0, 20.0
        ];

        let luy = calculate_viscous_y(&u, 1.0);

        let expected_luy = dmatrix![
            -49.0,  42.0,   7.0,   0.0,  -7.0;
             42.0,   7.0,   0.0,   0.0,  -7.0;
             -7.0,   7.0,   0.0,  -7.0,   7.0;
              7.0,   0.0,   0.0,  -7.0, -42.0;
              7.0,   0.0,  -7.0, -42.0,  49.0
        ];

        println!("luy: {}", luy);
        println!("expected_luy: {}", expected_luy);
        assert_eq!(luy.nrows(), 5);
        assert_eq!(luy.ncols(), 5);
        assert_relative_eq!(luy, expected_luy, epsilon = 1e-9);
    }

    #[test]
    fn test_calculate_viscous_terms() {
        let u = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0, 29.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0, 37.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0, 45.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0,  4.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0;
            22.0, 31.0, 40.0, 49.0,  2.0, 11.0, 20.0
        ];

        let v = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0, 29.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0, 37.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0, 45.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0,  4.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0;
            22.0, 31.0, 40.0, 49.0,  2.0, 11.0, 20.0
        ];

        let (lu, lv) = calculate_viscous_terms(&u, &v, 1.0, 1.0);

        let expected_lux = dmatrix![
            -49.0,  42.0,   0.0,   0.0,   0.0;
             49.0,   7.0,   0.0,   0.0,  -7.0;
             -7.0,   0.0,   0.0,   0.0,   7.0;
              7.0,   0.0,   0.0,  -7.0, -49.0;
              0.0,   0.0,   0.0, -42.0,  49.0
        ];

        let expected_luy = dmatrix![
            -49.0,  42.0,   7.0,   0.0,  -7.0;
             42.0,   7.0,   0.0,   0.0,  -7.0;
             -7.0,   7.0,   0.0,  -7.0,   7.0;
              7.0,   0.0,   0.0,  -7.0, -42.0;
              7.0,   0.0,  -7.0, -42.0,  49.0
        ];

        let expected_lu = expected_lux.clone() + expected_luy.clone();
        let expected_lv = expected_lux + expected_luy;

        println!("lu: {}", lu);
        println!("expected_lu: {}", expected_lu);
        assert_eq!(lu.nrows(), 5);
        assert_eq!(lu.ncols(), 5);
        assert_relative_eq!(lu, expected_lu, epsilon = 1e-9);

        println!("lv: {}", lv);
        println!("expected_lv: {}", expected_lv);
        assert_eq!(lv.nrows(), 5);
        assert_eq!(lv.ncols(), 5);
        assert_relative_eq!(lv, expected_lv, epsilon = 1e-9);
    }

    #[test]
    fn test_element_wise_multiply() {
        let a = dmatrix![
            1.0, 2.0;
            3.0, 4.0
        ];
        let b = dmatrix![
            5.0, 6.0;
            7.0, 8.0
        ];
        let expected = dmatrix![
            5.0, 12.0;
            21.0, 32.0
        ];
        let result = element_wise_multiply(&a, &b);
        assert_relative_eq!(result, expected, epsilon = 1e-9);
    }

    #[test]
    fn test_calculate_nu_nv() {
        let uuce = dmatrix![
            1849.000, 756.250,  25.000, 196.000, 529.000;
            702.250,  56.250, 169.000, 484.000, 961.000;
            100.000, 144.000, 441.000, 900.000, 1260.250;
            210.250, 400.000, 841.000, 1444.000, 1600.000;
            361.000, 784.000, 1369.000, 1806.250, 552.250
        ];

        let uvco = dmatrix![
            1173.000, 1870.500, 673.750,  27.500, 203.000, 540.500;
            1785.000,  715.500,  60.000, 175.500, 495.000, 868.000;
            663.000,   70.000, 150.000, 451.500, 915.000, 1278.000;
             85.500,  217.500, 410.000, 855.500, 1330.000, 1620.000;
            238.000,  370.500, 798.000, 1387.500, 1827.500, 564.000;
            473.000,  742.500, 1314.000, 1890.000, 517.500,  52.500
        ];

        let vvce = dmatrix![
            1806.250, 729.000,  64.000, 182.250, 506.250;
            676.000,  49.000, 156.250, 462.250, 930.250;
             90.250, 225.000, 420.250, 870.250, 1225.000;
            196.000, 380.250, 812.250, 1406.250, 1849.000;
            484.000, 756.250, 1332.250, 1764.000, 529.000
        ];

        let nu = calculate_nu(&uuce, &uvco, 2.0, 0.5);
        let nv = calculate_nv(&vvce, &uvco, 2.0, 0.5);

        let expected_nu = dmatrix![
            -2712.375, -1661.00,  303.00,  783.00,  962.00;
            -1487.125,   203.875,  739.00, 1135.00,  875.625;
              319.125,   513.00, 1091.00, 1221.00,  749.875;
              340.375,  1047.00, 1443.00, 1061.125, -3050.875
        ];

        let expected_nv = dmatrix![
            -2732.00, -1636.875,  310.50,  794.00;
            -1576.75,   259.50,  750.00, 1146.00;
              343.25,   520.50, 1102.00,  917.00;
              445.00,  1058.00, 1454.00, 1134.25;
              730.50,  1410.00, 1114.75, -3125.00
        ];

        println!("nu: {}", nu);
        println!("expected_nu: {}", expected_nu);
        assert_eq!(nu.nrows(), 4);
        assert_eq!(nu.ncols(), 5);
        assert_relative_eq!(nu, expected_nu, epsilon = 1e-9);

        println!("nv: {}", nv);
        println!("expected_nv: {}", expected_nv);
        assert_eq!(nv.nrows(), 5);
        assert_eq!(nv.ncols(), 4);
        assert_relative_eq!(nv, expected_nv, epsilon = 1e-9);
    }

    #[test]
    fn test_update_velocity_field() {
        let mut u = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0, 29.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0, 37.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0, 45.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0,  4.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0
        ];

        let nu = dmatrix![
            -2216.25, -1355.50,  259.50,  607.50,  805.00;
            -1195.25,   167.75,  573.50,  879.50,  662.25;
              242.25,   448.50,  845.50, 1018.50,  629.75;
              283.25,   811.50, 1117.50,  802.25, -2311.25
        ];

        let lux = dmatrix![
            -49.0,  42.0,   0.0,   0.0,   0.0;
             49.0,   7.0,   0.0,   0.0,  -7.0;
             -7.0,   0.0,   0.0,   0.0,   7.0;
              7.0,   0.0,   0.0,  -7.0, -49.0
        ];

        let luy = dmatrix![
            -49.0,  42.0,   7.0,   0.0,  -7.0;
             42.0,   7.0,   0.0,   0.0,  -7.0;
             -7.0,   7.0,   0.0,  -7.0,   7.0;
              7.0,   0.0,   0.0,  -7.0, -42.0
        ];

        let dt = 0.1;
        let re = 100.0;
        update_velocity_field(&mut u, &nu, &lux, &luy, dt, re);

        let expected_u = dmatrix![
            30.0000, 39.0000, 48.0000,  1.0000, 10.0000, 19.0000, 28.0000;
            38.0000, 268.5270, 142.6340, -16.9430, -42.7500, -53.5070, 29.0000;
            46.0000, 125.6160, -8.7610, -40.3500, -61.9500, -31.2390, 37.0000;
             5.0000, -10.2390, -28.8430, -59.5500, -67.8570, -26.9610, 45.0000;
            13.0000, -13.3110, -57.1500, -78.7500, -38.2390, 275.0340, 4.0000;
            21.0000, 23.0000, 32.0000, 41.0000, 43.0000, 3.0000, 12.0000
        ];

        println!("Updated u: {}", u);
        println!("Expected u: {}", expected_u);
        assert_relative_eq!(u, expected_u, epsilon = 1e-3);
    }

    #[test]
    fn test_calculate_divergence_term() {
        let u = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0, 29.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0, 37.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0, 45.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0,  4.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0
        ];

        let v = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0;
            22.0, 31.0, 40.0, 49.0,  2.0, 11.0
        ];

        let dx = 0.5;
        let dy = 2.0;

        let b = calculate_divergence_term(&u, &v, dx, dy);

        let expected_b = dmatrix![
            20.5000, -102.0000,  17.0000,  20.5000,  20.5000;
            -102.0000,   3.0000,  20.5000,  20.5000,  20.5000;
             20.5000,  17.0000,  20.5000,  20.5000,   3.0000;
              3.0000,  20.5000,  20.5000,  20.5000,  17.0000;
             17.0000,  20.5000,  20.5000,   3.0000, -102.0000
        ];

        println!("b: {}", b);
        println!("expected_b: {}", expected_b);
        assert_eq!(b.nrows(), 5);
        assert_eq!(b.ncols(), 5);
        assert_relative_eq!(b, expected_b, epsilon = 1e-3);
    }

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
    fn test_apply_pressure_corrections() {
        let mut u = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0, 29.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0, 37.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0, 45.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0,  4.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0
        ];

        let mut v = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0;
            22.0, 31.0, 40.0, 49.0,  2.0, 11.0
        ];

        let p = dmatrix![
             0.0000, -20.9977, -97.3091, -138.6966, -155.9034;
            -19.5023, -67.6841, -115.2330, -142.3773, -152.6102;
            -92.8227, -112.0034, -133.0614, -142.4693, -139.0500;
           -126.4625, -137.4455, -142.0398, -134.8886, -119.0705;
           -146.1193, -148.7761, -142.2636, -115.4750,  -66.2727
        ];

        let dx = 1.0;
        let dy = 1.0;

        apply_pressure_corrections(&mut u, &mut v, &p, dx, dy);

        let expected_u = dmatrix![
            30.0000, 39.0000, 48.0000,  1.0000, 10.0000, 19.0000, 28.0000;
            38.0000, 66.5023, 53.6864, 26.9239, 21.6807, 23.7068, 29.0000;
            46.0000, 79.3205, 52.3193, 34.8284, 26.0920, 21.4398, 37.0000;
             5.0000, 47.6398, 41.4420, 33.9784, 26.4193, 16.0205, 45.0000;
            13.0000, 34.6568, 35.3307, 33.2239, 22.5864, -8.7977,  4.0000;
            21.0000, 23.0000, 32.0000, 41.0000, 43.0000,  3.0000, 12.0000
        ];

        let expected_v = dmatrix![
            30.0000, 39.0000, 48.0000,  1.0000, 10.0000, 19.0000;
            38.0000, 67.9977, 83.3114, 50.3875, 35.2068, 27.0000;
            46.0000, 54.1818, 55.5489, 44.1443, 36.2330, 35.0000;
             5.0000, 33.1807, 37.0580, 34.4080, 30.5807, 36.0000;
            13.0000, 25.9830, 28.5943, 25.8489, 26.1818, 44.0000;
            21.0000, 25.6568, 25.4875, 14.2114, -6.2023,  3.0000;
            22.0000, 31.0000, 40.0000, 49.0000,  2.0000, 11.0000
        ];

        println!("u after correction: {}", u);
        println!("expected u: {}", expected_u);
        assert_relative_eq!(u, expected_u, epsilon = 1e-3);

        println!("v after correction: {}", v);
        println!("expected v: {}", expected_v);
        assert_relative_eq!(v, expected_v, epsilon = 1e-3);
    }

    #[test]
    fn test_full_simulation_step_function_calls() {
        let nx = 5;
        let ny = 5;
        let dx = 1.0;
        let dy = 1.0;
        let dt = 0.1;
        let re = 100.0;

        let u_initial = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0, 29.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0, 37.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0, 45.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0,  4.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0
        ];

        let v_initial = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0;
            38.0, 47.0,  7.0,  9.0, 18.0, 27.0;
            46.0,  6.0,  8.0, 17.0, 26.0, 35.0;
             5.0, 14.0, 16.0, 25.0, 34.0, 36.0;
            13.0, 15.0, 24.0, 33.0, 42.0, 44.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0;
            22.0, 31.0, 40.0, 49.0,  2.0, 11.0
        ];

        println!("INITIAL u matrix:\n{}", u_initial);
        println!("INITIAL v matrix:\n{}", v_initial);

        let u = u_initial.clone();
        let v = v_initial.clone();

        let lux = calculate_viscous_x(&u, dx);
        let luy = calculate_viscous_y(&u, dy);
        let lvx = calculate_viscous_x(&v, dx);
        let lvy = calculate_viscous_y(&v, dy);

        let uce = interpolate_u_to_cell_centers(&u);
        let uco = interpolate_u_to_cell_edges(&u);
        let vco = interpolate_v_to_cell_edges(&v);
        let vce = interpolate_v_to_cell_centers(&v);

        let uuce = uce.component_mul(&uce);
        let uvco = uco.component_mul(&vco);
        let vvce = vce.component_mul(&vce);

        let nu = calculate_nu(&uuce, &uvco, dx, dy);
        let nv = calculate_nv(&vvce, &uvco, dx, dy);

        let (mut u_star, mut v_star) =
            calculate_intermediate_velocity(&u, &v, &nu, &nv, &lux, &luy, &lvx, &lvy, dt, re);

        let b = calculate_divergence_term(&u_star, &v_star, dx, dy);

        let p = solve_poisson_equation_direct(&b, nx, ny, dx, dy, true)
            .expect("Poisson solve failed in test");

        apply_pressure_corrections(&mut u_star, &mut v_star, &p, dx, dy);

        let u_final = u_star;
        let v_final = v_star;

        println!("FINAL u matrix (Rust):\n{}", u_final);
        println!("FINAL v matrix (Rust):\n{}", v_final);

        let expected_u_final = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0, 28.0;
            38.0, 59.254, 50.174, 20.644, 16.916, 28.012, 29.0;
            46.0, 60.48, 49.0, 25.167, 19.677, 31.677, 37.0;
             5.0, 45.389, 33.193, 22.728, 18.241, 35.45, 45.0;
            13.0, 36.361, 27.563, 27.429, 34.168, -1.5212, 4.0;
            21.0, 23.0, 32.0, 41.0, 43.0,  3.0, 12.0
        ];
        let expected_v_final = dmatrix![
            30.0, 39.0, 48.0,  1.0, 10.0, 19.0;
            38.0, 64.746, 62.572, 42.928, 36.012, 27.0;
            46.0, 44.774, 45.949, 41.426, 38.665, 35.0;
             5.0, 20.091, 35.898, 38.336, 39.773, 36.0;
            13.0, 22.027, 27.658, 22.956,  7.0288, 44.0;
            21.0, 34.361, 29.924, 16.353,  7.5212, 3.0;
            22.0, 31.0, 40.0, 49.0,  2.0, 11.0
        ];

        println!(
            "Expected FINAL u matrix (from MATLAB):\n{}",
            expected_u_final
        );
        println!(
            "Expected FINAL v matrix (from MATLAB):\n{}",
            expected_v_final
        );
        assert_relative_eq!(u_final, expected_u_final, epsilon = 1e-3);
        assert_relative_eq!(v_final, expected_v_final, epsilon = 1e-3);

        let b_final_rust = calculate_divergence_term(&u_final, &v_final, dx, dy);
        let final_divergence_norm_rust = b_final_rust.norm();
        let expected_divergence_norm_from_matlab = 47.0;
        assert!(
            (final_divergence_norm_rust - expected_divergence_norm_from_matlab).abs() < 1.0,
            "Rust final divergence norm ({:.6e}) is not close to MATLAB's reported norm ({:.1})",
            final_divergence_norm_rust,
            expected_divergence_norm_from_matlab
        );
    }
}
