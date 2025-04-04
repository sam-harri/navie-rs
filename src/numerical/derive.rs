use ndarray::{Array2, s};

/// Computes the Laplace operator of a 2D field with grid spacings `dx` and `dy`.
/// 
/// The interior is computed with the standard five‐point stencil:
/// 
/// \[
/// \nabla^2 f_{ij} = \frac{f_{i,j+1} - 2f_{ij} + f_{i,j-1}}{dx^2} + \frac{f_{i+1,j} - 2f_{ij} + f_{i-1,j}}{dy^2},
/// \]
/// 
/// while the boundaries are approximated using one-sided differences (with second‑order accuracy)
/// implemented via slicing.
/// 
/// # Arguments
/// - `field`: The input 2D array.
/// - `dx`: The grid spacing in the x‑direction.
/// - `dy`: The grid spacing in the y‑direction.
/// 
/// # Panics
/// The function asserts that `field` has at least 3 rows and 3 columns.
/// 
/// # Example
/// ```
/// use ndarray::array;
/// use your_module::laplace_operator;
///
/// // Build a 5x5 field (for example, representing a quadratic function).
/// let field = array![
///     [ 0.0,  1.0,  4.0,  9.0, 16.0],
///     [ 1.0,  2.0,  5.0, 10.0, 17.0],
///     [ 4.0,  5.0,  8.0, 13.0, 20.0],
///     [ 9.0, 10.0, 13.0, 18.0, 25.0],
///     [16.0, 17.0, 20.0, 25.0, 32.0],
/// ];
/// let lap = laplace_operator(&field, 1.0, 1.0);
/// // For a quadratic function f(x,y)=x^2+y^2 the Laplacian should be constant (4.0 in this case).
/// ```
/// 
pub fn laplace_operator(field: &Array2<f64>, dx: f64, dy: f64) -> Array2<f64> {
    let (nrows, ncols) = field.dim();
    assert!(nrows >= 3 && ncols >= 3, "Field must have at least 3 rows and 3 columns.");
    let mut lap = Array2::<f64>::zeros((nrows, ncols));
    let dx2 = dx * dx;
    let dy2 = dy * dy;

    // --- Interior (using slicing) ---
    {
        let center = field.slice(s![1..nrows-1, 1..ncols-1]);
        let left   = field.slice(s![1..nrows-1, 0..ncols-2]);
        let right  = field.slice(s![1..nrows-1, 2..]);
        let top    = field.slice(s![0..nrows-2, 1..ncols-1]);
        let bottom = field.slice(s![2.., 1..ncols-1]);

        let lap_interior = ((&left + &right) - 2.0 * &center) / dx2 +
                           ((&top  + &bottom) - 2.0 * &center) / dy2;
        lap.slice_mut(s![1..nrows-1, 1..ncols-1]).assign(&lap_interior);
    }

    // --- Left Boundary (j = 0) ---
    // f'' ≈ (-f[i,2] + 4 f[i,1] - 3 f[i,0]) / dx^2
    {
        let f0 = field.slice(s![.., 0]);
        let f1 = field.slice(s![.., 1]);
        let f2 = field.slice(s![.., 2]);
        lap.slice_mut(s![.., 0]).assign(&((-&f2 + 4.0 * &f1 - 3.0 * &f0) / dx2));
    }

    // --- Right Boundary (j = ncols-1) ---
    // f'' ≈ (3 f[i,ncols-1] - 4 f[i,ncols-2] + f[i,ncols-3]) / dx^2
    {
        let f_nm1 = field.slice(s![.., ncols-1]);
        let f_nm2 = field.slice(s![.., ncols-2]);
        let f_nm3 = field.slice(s![.., ncols-3]);
        lap.slice_mut(s![.., ncols-1]).assign(
            &((3.0 * &f_nm1 - 4.0 * &f_nm2 + &f_nm3) / dx2)
        );
    }

    // --- Top Boundary (i = 0) ---
    // f'' ≈ (-f[2,j] + 4 f[1,j] - 3 f[0,j]) / dy^2
    {
        let f0 = field.slice(s![0, ..]);
        let f1 = field.slice(s![1, ..]);
        let f2 = field.slice(s![2, ..]);
        lap.slice_mut(s![0, ..]).assign(&((-&f2 + 4.0 * &f1 - 3.0 * &f0) / dy2));
    }

    // --- Bottom Boundary (i = nrows-1) ---
    // f'' ≈ (3 f[nrows-1,j] - 4 f[nrows-2,j] + f[nrows-3,j]) / dy^2
    {
        let f_nm1 = field.slice(s![nrows-1, ..]);
        let f_nm2 = field.slice(s![nrows-2, ..]);
        let f_nm3 = field.slice(s![nrows-3, ..]);
        lap.slice_mut(s![nrows-1, ..]).assign(
            &((3.0 * &f_nm1 - 4.0 * &f_nm2 + &f_nm3) / dy2)
        );
    }

    lap
}

/// Specifies the direction in which to take the derivative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeDirection {
    X,
    Y,
}

/// Computes the derivative of a 2D field and returns an array of the same shape.
///
/// Uses central differences for interior points and one-sided differences at the boundaries.
///
/// # Arguments
/// - `field`: The input 2D array.
/// - `d`: The spacing (step size) in the derivative direction.
/// - `dir`: The direction (`X` or `Y`) in which to compute the derivative.
///
/// # Example
/// ```
/// use ndarray::array;
/// let field = array![
///     [0.0, 1.0, 4.0, 9.0, 16.0],
///     [0.0, 1.0, 4.0, 9.0, 16.0],
///     [0.0, 1.0, 4.0, 9.0, 16.0],
///     [0.0, 1.0, 4.0, 9.0, 16.0],
///     [0.0, 1.0, 4.0, 9.0, 16.0],
/// ];
/// let deriv = derivative_same_shape(&field, 1.0, DerivativeDirection::X);
/// // 'deriv' now contains the x-derivative computed with centered differences in the interior.
/// ```
pub fn derivative_same_shape(field: &Array2<f64>, d: f64, dir: DerivativeDirection) -> Array2<f64> {
    let (nrows, ncols) = field.dim();

    let mut deriv = Array2::<f64>::zeros((nrows, ncols));
    match dir {
        DerivativeDirection::X => {
            assert!(ncols > 2, "Field must have more than 2 columns for X derivative");
            let mid = (&field.slice(s![.., 2..]) - &field.slice(s![.., ..ncols-2])) / (2.0 * d);
            deriv.slice_mut(s![.., 1..ncols-1]).assign(&mid);
            let left = (&field.slice(s![.., 1]) - &field.slice(s![.., 0])) / d;
            deriv.slice_mut(s![.., 0]).assign(&left);
            let right = (&field.slice(s![.., ncols-1]) - &field.slice(s![.., ncols-2])) / d;
            deriv.slice_mut(s![.., ncols-1]).assign(&right);
        }
        DerivativeDirection::Y => {
            assert!(nrows > 2, "Field must have more than 2 rows for Y derivative");
            let mid = (&field.slice(s![2.., ..]) - &field.slice(s![..nrows-2, ..])) / (2.0 * d);
            deriv.slice_mut(s![1..nrows-1, ..]).assign(&mid);
            let top = (&field.slice(s![1, ..]) - &field.slice(s![0, ..])) / d;
            deriv.slice_mut(s![0, ..]).assign(&top);
            let bottom = (&field.slice(s![nrows-1, ..]) - &field.slice(s![nrows-2, ..])) / d;
            deriv.slice_mut(s![nrows-1, ..]).assign(&bottom);
        }
    }
    deriv
}

/// Computes the derivative of a 2D field using central differences and returns an array
/// of the same shape.
///
/// Uses central differences for the interior points and one-sided differences at the boundaries.
///
/// # Arguments
/// - `field`: The input 2D array.
/// - `d`: The spacing (step size) in the derivative direction.
/// - `dir`: The direction (`X` or `Y`) to compute the derivative.
///
/// # Example
/// ```
/// use ndarray::array;
/// let field = array![
///     [0.0, 1.0, 2.0, 3.0, 4.0],
///     [0.0, 1.0, 2.0, 3.0, 4.0],
///     [0.0, 1.0, 2.0, 3.0, 4.0],
///     [0.0, 1.0, 2.0, 3.0, 4.0],
///     [0.0, 1.0, 2.0, 3.0, 4.0],
/// ];
/// let deriv = central_difference(&field, 1.0, DerivativeDirection::X);
/// // 'deriv' will have the derivative values computed across x.
/// ```
pub fn central_difference(field: &Array2<f64>, d: f64, dir: DerivativeDirection) -> Array2<f64> {
    let (nrows, ncols) = field.dim();
    let mut deriv = Array2::<f64>::zeros((nrows, ncols));

    match dir {
        DerivativeDirection::X => {
            assert!(ncols >= 2, "Need at least 2 columns for x-derivative.");
            if ncols >= 3 {
                let right = field.slice(s![.., 2..]);
                let left  = field.slice(s![.., ..ncols-2]);
                let mut mid = deriv.slice_mut(s![.., 1..ncols-1]);
                mid.assign(&((&right - &left) / (2.0 * d)));
            }
            for i in 0..nrows {
                deriv[[i, 0]] = (field[[i, 1]] - field[[i, 0]]) / d;
                deriv[[i, ncols - 1]] = (field[[i, ncols - 1]] - field[[i, ncols - 2]]) / d;
            }
        }
        DerivativeDirection::Y => {
            assert!(nrows >= 2, "Need at least 2 rows for y-derivative.");
            if nrows >= 3 {
                let bottom = field.slice(s![2.., ..]);
                let top    = field.slice(s![..nrows-2, ..]);
                let mut mid = deriv.slice_mut(s![1..nrows-1, ..]);
                mid.assign(&((&bottom - &top) / (2.0 * d)));
            }
            for j in 0..ncols {
                deriv[[0, j]] = (field[[1, j]] - field[[0, j]]) / d;
                deriv[[nrows - 1, j]] = (field[[nrows - 1, j]] - field[[nrows - 2, j]]) / d;
            }
        }
    }

    deriv
}

/// Computes the difference between adjacent cells (face differences) along a specified direction.
///
/// This function returns an array with a shape reduced by one in the differentiation direction.
/// For example, for `X` the result shape is `(rows, cols-1)`.
///
/// # Arguments
/// - `field`: The input 2D array.
/// - `d`: The spacing (step size) between points.
/// - `dir`: The direction (`X` or `Y`) to compute the face difference.
///
/// # Example
/// ```
/// use ndarray::array;
/// let field = array![
///     [0.0,  1.0,  4.0,  9.0, 16.0],
///     [0.0,  1.0,  4.0,  9.0, 16.0],
/// ];
/// let face_diff = face_difference(&field, 2.0, DerivativeDirection::X);
/// // 'face_diff' will be an array of shape (2, 4) with differences computed between adjacent x-values.
/// ```
pub fn face_difference(field: &Array2<f64>, d: f64, dir: DerivativeDirection) -> Array2<f64> {
    let (rows, cols) = field.dim();
    match dir {
        DerivativeDirection::X => {
            let right = field.slice(s![.., 1..]);
            let left  = field.slice(s![.., ..cols-1]);
            (&right - &left) / d
        }
        DerivativeDirection::Y => {
            let bottom = field.slice(s![1.., ..]);
            let top    = field.slice(s![..rows-1, ..]);
            (&bottom - &top) / d
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_laplace_operator_interior() {
        // For f(x,y) = x^2 + y^2, the Laplacian should be 2 + 2 = 4.
        // Create a 7x7 grid with f(i,j) = i^2 + j^2.
        let n = 7;
        let mut field = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                field[[i, j]] = (i * i + j * j) as f64;
            }
        }
        let lap = laplace_operator(&field, 1.0, 1.0);
        // Check the interior points.
        for i in 1..n-1 {
            for j in 1..n-1 {
                assert!((lap[[i, j]] - 4.0).abs() < 1e-12,
                    "lap[{}, {}] = {} != 4.0", i, j, lap[[i, j]]);
            }
        }
        assert!(field.dim() == lap.dim());
    }

    #[test]
    fn test_central_difference_x() {
        let mut field = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                field[[i, j]] = j as f64;
            }
        }
        let deriv = central_difference(&field, 1.0, DerivativeDirection::X);
        for i in 0..5 {
            for j in 0..5 {
                assert!((deriv[[i, j]] - 1.0).abs() < 1e-12, "deriv[{}, {}] = {}", i, j, deriv[[i, j]]);
            }
        }
    }

    #[test]
    fn test_central_difference_y() {
        let mut field = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                field[[i, j]] = i as f64;
            }
        }
        let deriv = central_difference(&field, 1.0, DerivativeDirection::Y);
        for i in 0..5 {
            for j in 0..5 {
                assert!((deriv[[i, j]] - 1.0).abs() < 1e-12, "deriv[{}, {}] = {}", i, j, deriv[[i, j]]);
            }
        }
    }
    #[test]
    fn test_face_difference_x() {
        let p = array![
            [0.0,  1.0,  4.0,  9.0, 16.0],
            [0.0,  1.0,  4.0,  9.0, 16.0],
            [0.0,  1.0,  4.0,  9.0, 16.0],
            [0.0,  1.0,  4.0,  9.0, 16.0],
            [0.0,  1.0,  4.0,  9.0, 16.0],
        ];
        let dpdx = face_difference(&p, 2.0, DerivativeDirection::X);
        let expected = array![
            [0.5, 1.5, 2.5, 3.5],
            [0.5, 1.5, 2.5, 3.5],
            [0.5, 1.5, 2.5, 3.5],
            [0.5, 1.5, 2.5, 3.5],
            [0.5, 1.5, 2.5, 3.5],
        ];
        assert_eq!(dpdx, expected);
    }

    #[test]
    fn test_face_difference_y() {
        let p = array![
            [0.0,  0.0,  0.0,  0.0,  0.0],
            [1.0,  1.0,  1.0,  1.0,  1.0],
            [4.0,  4.0,  4.0,  4.0,  4.0],
            [9.0,  9.0,  9.0,  9.0,  9.0],
            [16.0, 16.0, 16.0, 16.0, 16.0],
        ];
        let dpdy = face_difference(&p, 0.5, DerivativeDirection::Y);
        let expected = array![
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [6.0, 6.0, 6.0, 6.0, 6.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [14.0, 14.0, 14.0, 14.0, 14.0],
        ];
        assert_eq!(dpdy, expected);
    }

    #[test]
    fn test_derivative_same_shape_x() {
        let field = array![
            [0.0,  1.0,  4.0,  9.0, 16.0],
            [0.0,  1.0,  4.0,  9.0, 16.0],
            [0.0,  1.0,  4.0,  9.0, 16.0],
            [0.0,  1.0,  4.0,  9.0, 16.0],
            [0.0,  1.0,  4.0,  9.0, 16.0],
        ];
        let deriv = derivative_same_shape(&field, 1.0, DerivativeDirection::X);
        let expected = array![
            [1.0, 2.0, 4.0, 6.0, 7.0],
            [1.0, 2.0, 4.0, 6.0, 7.0],
            [1.0, 2.0, 4.0, 6.0, 7.0],
            [1.0, 2.0, 4.0, 6.0, 7.0],
            [1.0, 2.0, 4.0, 6.0, 7.0],
        ];
        assert_eq!(deriv, expected);
    }

    #[test]
    fn test_derivative_same_shape_y() {
        let field = array![
            [0.0,  0.0,  0.0],
            [1.0,  1.0,  1.0],
            [4.0,  4.0,  4.0],
            [9.0,  9.0,  9.0],
            [16.0, 16.0, 16.0],
        ];
        let deriv = derivative_same_shape(&field, 1.0, DerivativeDirection::Y);
        let expected = array![
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
        ];
        assert_eq!(deriv, expected);
    }
}
