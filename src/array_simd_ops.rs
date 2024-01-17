use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

use crate::{private, ArrayOps};

pub trait ArraySimdOps<T, const N: usize, const M: usize>: ArrayOps<Simd<T, M>, N>
where
    T: SimdElement,
    LaneCount<M>: SupportedLaneCount
{
    fn array_unsimd(self) -> [T; N*M];
    fn array_unsimd_ref(&self) -> &[T; N*M];
    fn array_unsimd_mut(&mut self) -> &mut [T; N*M];
}

impl<T, const N: usize, const M: usize> ArraySimdOps<T, N, M> for [Simd<T, M>; N]
where
    T: SimdElement,
    LaneCount<M>: SupportedLaneCount
{
    fn array_unsimd(self) -> [T; N*M]
    {
        unsafe {private::transmute_unchecked_size(self)}
    }

    fn array_unsimd_ref(&self) -> &[T; N*M]
    {
        unsafe {&*self.as_ptr().cast()}
    }

    fn array_unsimd_mut(&mut self) -> &mut [T; N*M]
    {
        unsafe {&mut *self.as_mut_ptr().cast()}
    }
}