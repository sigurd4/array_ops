use core::{ops::{Mul, AddAssign}, borrow::Borrow, marker::Destruct};

use super::*;

#[const_trait]
pub trait Array2dOps<T, const M: usize, const N: usize>: ArrayOps<[T; N], M>
{
    /// Transposes a two-dimensional array (as if it were a matrix)
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array__ops::*;
    /// 
    /// let matrix: [[u8; 5]; 3] = [
    ///     [1,   2,  3,  4,  5],
    ///     [6,   7,  8,  9, 10],
    ///     [11, 12, 13, 14, 15]
    /// ];
    /// 
    /// assert_eq!(matrix.transpose(), [
    ///     [1,  6, 11],
    ///     [2,  7, 12],
    ///     [3,  8, 13],
    ///     [4,  9, 14],
    ///     [5, 10, 15]
    /// ]);
    /// ```
    fn transpose(self) -> [[T; M]; N];
    
    fn mul_kronecker<Rhs, const H: usize, const W: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<T as Mul<Rhs>>::Output; W*N]; H*N]
    where
        T: Mul<Rhs> + Copy,
        Rhs: Copy;

    fn into_diagonal(self) -> [T; crate::min_len(M, N)]
    where
        T: ~const Destruct;
    fn diagonal_ref(&self) -> [&T; crate::min_len(M, N)];
    fn diagonal_mut(&mut self) -> [&mut T; crate::min_len(M, N)];
}

pub const fn transpose<T, const M: usize, const N: usize>(matrix: [[T; N]; M]) -> [[T; M]; N]
{
    // Alternative 1: (dirtier)
    let mut this_t: [[T; M]; N] = unsafe {private::uninit()};
    let mut i = 0;
    while i != M
    {
        let mut j = 0;
        while j != N
        {
            unsafe {core::ptr::copy_nonoverlapping(
                &matrix[i][j] as *const T,
                &mut this_t[j][i] as *mut T,
                1
            )};
            j += 1;
        }
        i += 1;
    }

    core::mem::forget(matrix);

    this_t

    // Alternative 2: (cleaner)
    /*ArrayOps::fill(const |i| ArrayOps::fill(const |j| unsafe {
        core::ptr::read(&this.borrow()[j][i] as *const T)
    }))*/
}

pub const fn diagonal_ref<T, const M: usize, const N: usize>(array: &[[T; N]; M]) -> [&T; crate::min_len(M, N)]
{
    let mut dst: [&T; crate::min_len(M, N)] = unsafe {private::uninit()};
    
    let mut n = 0;
    while n != crate::min_len(M, N)
    {
        dst[n] = &array[n][n];
        n += 1;
    }

    dst
}
pub const fn diagonal_mut<T, const M: usize, const N: usize>(array: &mut [[T; N]; M]) -> [&mut T; crate::min_len(M, N)]
{
    let mut dst: [&mut T; crate::min_len(M, N)] = unsafe {private::uninit()};
    
    let mut n = 0;
    while n != crate::min_len(M, N)
    {
        dst[n] = unsafe {core::mem::transmute(&mut array[n][n])};
        n += 1;
    }

    dst
}

impl<T, const M: usize, const N: usize> Array2dOps<T, M, N> for [[T; N]; M]
{
    fn transpose(self) -> [[T; M]; N]
    {
        crate::transpose(self)
    }
    
    fn mul_kronecker<Rhs, const H: usize, const W: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<T as Mul<Rhs>>::Output; W*N]; H*N]
    where
        T: Mul<Rhs> + Copy,
        Rhs: Copy
    {
        ArrayOps::fill(|r| ArrayOps::fill(|c| self[r % M][c % N]*rhs[r / M][c / N]))
    }
    
    fn into_diagonal(self) -> [T; crate::min_len(M, N)]
    where
        T: /*~const*/ Destruct
    {
        let mut dst: [T; crate::min_len(M, N)] = unsafe {private::uninit()};
        
        let mut m = 0;
        while m != M
        {
            let mut n = 0;
            while n != N
            {
                unsafe {
                    let src = self[m].as_ptr().add(n);
                    if m == n
                    {
                        core::ptr::copy_nonoverlapping(src, dst.as_mut_ptr().add(n), 1)
                    }
                    else
                    {
                        let _ = core::ptr::read(src);
                    }
                }
                n += 1;
            }
            m += 1;
        }
        core::mem::forget(self);
    
        dst
    }
    fn diagonal_ref(&self) -> [&T; crate::min_len(M, N)]
    {
        crate::diagonal_ref(self)
    }
    fn diagonal_mut(&mut self) -> [&mut T; crate::min_len(M, N)]
    {
        crate::diagonal_mut(self)
    }
}