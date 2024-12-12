use crate::{private, Array2dOps};

pub trait CollumnArrayOps<T, const N: usize>: Array2dOps<T, N, 1>
{
    fn into_uncollumn(self) -> [T; N];
    fn as_uncollumn(&self) -> &[T; N];
    fn as_uncollumn_mut(&mut self) -> &mut [T; N];
}

pub const fn into_uncollumn<T, const N: usize>(array: [[T; 1]; N]) -> [T; N]
{
    unsafe {
        private::transmute_unchecked_size(array)
    }
}

pub const fn as_uncollumn<T, const N: usize>(array: &[[T; 1]; N]) -> &[T; N]
{
    unsafe {
        &*array.as_ptr().cast()
    }
}

pub const fn as_uncollumn_mut<T, const N: usize>(array: &mut [[T; 1]; N]) -> &mut [T; N]
{
    unsafe {
        &mut *array.as_mut_ptr().cast()
    }
}

impl<T, const N: usize> CollumnArrayOps<T, N> for [[T; 1]; N]
{
    fn into_uncollumn(self) -> [T; N]
    {
        crate::into_uncollumn(self)
    }

    fn as_uncollumn(&self) -> &[T; N]
    {
        crate::as_uncollumn(self)
    }

    fn as_uncollumn_mut(&mut self) -> &mut [T; N]
    {
        crate::as_uncollumn_mut(self)
    }
}