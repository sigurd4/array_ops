use crate::Array2dOps;

pub trait SquareArray2dOps<T, const N: usize>: Array2dOps<T, N, N>
{
    /// Transposes a square matrix in-place.
    /// 
    /// # Examples
    /// ```rust
    /// use array__ops::*;
    /// 
    /// let mut a = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ];
    /// 
    /// a.transpose_assign();
    /// 
    /// assert_eq!(a, [
    ///     [1, 4, 7],
    ///     [2, 5, 8],
    ///     [3, 6, 9]
    /// ]);
    /// ```
    fn transpose_assign(&mut self);
}

pub const fn transpose_assign<T, const N: usize>(matrix: &mut [[T; N]; N])
{
    let ptr: *mut T = matrix.as_mut_ptr().cast();
    let mut r = 0;
    while r < N
    {
        let mut c = r + 1;
        while c < N
        {
            let i = r*N + c;
            let j = r + c*N;
            unsafe {
                core::ptr::swap_nonoverlapping(ptr.add(i), ptr.add(j), 1);
            }
            c += 1;
        }
        r += 1;
    }
}

impl<T, const N: usize> SquareArray2dOps<T, N> for [[T; N]; N]
{
    fn transpose_assign(&mut self)
    {
        crate::transpose_assign(self)
    }
}