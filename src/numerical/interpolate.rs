use ndarray::{Array2, s};

/// Specifies the axis along which to perform 1D (simple) interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpAxis {
    /// Interpolate horizontally (i.e. average adjacent values along each row).
    Horizontal,
    /// Interpolate vertically (i.e. average adjacent values along each column).
    Vertical,
}

/// Performs simple linear interpolation along a specified axis of a 2D field using slicing.
///
/// The routine returns an array whose dimensions are reduced by one along the chosen axis.
/// It computes the midpoint between adjacent grid points along that axis using a weighted average.
/// Since the target point is assumed to lie exactly midway, the weight is 0.5 (i.e. the arithmetic mean).
///
/// # Arguments
/// - `field`: The input 2D array.
/// - `axis`: The axis along which to interpolate (Horizontal or Vertical).
///
/// # Example
/// ```
/// use ndarray::array;
/// use your_module::{InterpAxis, interpolate_simple};
///
/// let field = array![ [1.0, 2.0, 3.0],
///                     [4.0, 5.0, 6.0] ];
/// // Horizontal interpolation returns an array of shape (2, 2):
/// // [ [(1+2)/2, (2+3)/2], [(4+5)/2, (5+6)/2 ] ] = [ [1.5, 2.5], [4.5, 5.5] ]
/// let horiz = interpolate_simple(&field, InterpAxis::Horizontal, 1.0);
/// assert_eq!(horiz, ndarray::array![ [1.5, 2.5], [4.5, 5.5] ]);
///
/// // Vertical interpolation returns an array of shape (1, 3):
/// // [ [(1+4)/2, (2+5)/2, (3+6)/2] ] = [ [2.5, 3.5, 4.5] ]
/// let vert = interpolate_simple(&field, InterpAxis::Vertical, 1.0);
/// assert_eq!(vert, ndarray::array![ [2.5, 3.5, 4.5] ]);
/// ```
pub fn interpolate_simple(field: &Array2<f64>, axis: InterpAxis) -> Array2<f64> {
    match axis {
        InterpAxis::Horizontal => {
            let (_, ncols) = field.dim();
            // Use slicing to obtain adjacent pairs along each row.
            let left = field.slice(s![.., ..ncols - 1]);
            let right = field.slice(s![.., 1..]);
            (&left + &right) * 0.5
        }
        InterpAxis::Vertical => {
            let (nrows, _) = field.dim();
            // Use slicing to obtain adjacent pairs along each column.
            let top = field.slice(s![..nrows - 1, ..]);
            let bottom = field.slice(s![1.., ..]);
            (&top + &bottom) * 0.5
        }
    }
}

/// Performs bilinear interpolation over each 2×2 block of a 2D field using slicing.
///
/// This routine computes the interpolated value at the cell’s center (i.e. the midpoint in both x and y)
/// by averaging the four corner values. For midpoint interpolation, each corner contributes equally (0.25).
///
/// # Arguments
/// - `field`: The input 2D array.
///
/// # Example
/// ```
/// use ndarray::array;
/// use your_module::centered_bilinear_interpolation;
///
/// let field = array![ [1.0, 2.0, 3.0],
///                     [4.0, 5.0, 6.0],
///                     [7.0, 8.0, 9.0] ];
/// // Bilinear interpolation returns an array of shape (2, 2).
/// // For the top-left block, the interpolated value is (1+2+4+5)/4 = 3.0.
/// let bilin = centered_bilinear_interpolation(&field, 1.0, 1.0);
/// assert_eq!(bilin, ndarray::array![ [3.0, 4.0], [6.0, 7.0] ]);
/// ```
pub fn centered_bilinear_interpolation(field: &Array2<f64>) -> Array2<f64> {
    let (nrows, ncols) = field.dim();
    // Use slicing to grab the four neighboring blocks:
    let f00 = field.slice(s![..nrows - 1, ..ncols - 1]);
    let f10 = field.slice(s![..nrows - 1, 1..]);
    let f01 = field.slice(s![1.., ..ncols - 1]);
    let f11 = field.slice(s![1.., 1..]);
    (&f00 + &f10 + &f01 + &f11) * 0.25
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_interpolate_simple_horizontal() {
        let field = array![
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0]
        ];
        let expected = array![
            [1.5, 2.5, 3.5],
            [15.0, 25.0, 35.0],
            [150.0, 250.0, 350.0]
        ];
        assert_eq!(interpolate_simple(&field, InterpAxis::Horizontal), expected);
    }

    #[test]
    fn test_interpolate_simple_vertical() {
        let field = array![
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
            [1000.0, 2000.0, 3000.0]
        ];
        let expected = array![
            [5.5, 11.0, 16.5],
            [55.0, 110.0, 165.0],
            [550.0, 1100.0, 1650.0]
        ];
        assert_eq!(interpolate_simple(&field, InterpAxis::Vertical), expected);
    }

    #[test]
    fn test_centered_bilinear_interpolation() {
        let field = array![
            [ 1.0,  2.0,  3.0,  4.0],
            [ 5.0,  6.0,  7.0,  8.0],
            [ 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];
        let expected = array![
            [ (1.0+2.0+5.0+6.0)/4.0, (2.0+3.0+6.0+7.0)/4.0, (3.0+4.0+7.0+8.0)/4.0 ],
            [ (5.0+6.0+9.0+10.0)/4.0, (6.0+7.0+10.0+11.0)/4.0, (7.0+8.0+11.0+12.0)/4.0 ],
            [ (9.0+10.0+13.0+14.0)/4.0, (10.0+11.0+14.0+15.0)/4.0, (11.0+12.0+15.0+16.0)/4.0 ],
        ];
        assert_eq!(centered_bilinear_interpolation(&field), expected);
    }
}
