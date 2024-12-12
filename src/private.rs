#[repr(C)]
pub(crate) struct Pair<L, R>
{
    pub left: L,
    pub right: R
}

impl<L, R> Pair<L, R>
{
    pub const fn new(left: L, right: R) -> Self
    {
        Self {left, right}
    }

    pub const fn unpack(self) -> (L, R)
    {
        unsafe {
            let mut left_right: (L, R) = uninit();

            core::ptr::copy_nonoverlapping(&self.left as *const L, &mut left_right.0 as *mut L, 1);
            core::ptr::copy_nonoverlapping(&self.right as *const R, &mut left_right.1 as *mut R, 1);

            core::mem::forget(self);

            left_right
        }
    }

    pub const fn pack(left_right: (L, R)) -> Self
    {
        unsafe {
            let mut pair: Self = uninit();

            core::ptr::copy_nonoverlapping(&left_right.0 as *const L, &mut pair.left as *mut L, 1);
            core::ptr::copy_nonoverlapping(&left_right.1 as *const R, &mut pair.right as *mut R, 1);

            core::mem::forget(left_right);

            pair
        }
    }
    
    pub const fn unpack_mandrop(self) -> (ManuallyDrop<L>, ManuallyDrop<R>)
    {
        unsafe {
            let mut left_right: (ManuallyDrop<L>, ManuallyDrop<R>) = uninit();

            core::ptr::copy_nonoverlapping(&self.left as *const L, (&mut left_right.0 as *mut ManuallyDrop<L>).cast(), 1);
            core::ptr::copy_nonoverlapping(&self.right as *const R, (&mut left_right.1 as *mut ManuallyDrop<R>).cast(), 1);

            core::mem::forget(self);

            left_right
        }
    }
}

impl<L, R> From<(L, R)> for Pair<L, R>
{
    fn from(left_right: (L, R)) -> Self
    {
        Self::pack(left_right)
    }
}

impl<L, R> Into<(L, R)> for Pair<L, R>
{
    fn into(self) -> (L, R)
    {
        self.unpack()
    }
}

use core::{mem::{ManuallyDrop, MaybeUninit}, ops::DerefMut};

/*impl<T, const P: &'static [usize]> NotTuple for PartitionedArray<T, P>
where
[(); crate::sum_len::<{P}>()]: {}*/

pub(crate) const unsafe fn uninit<T>() -> T
{
    MaybeUninit::assume_init(MaybeUninit::uninit())
}

pub(crate) const unsafe fn split_transmute<A, B, C>(a: A) -> (B, C)
{
    transmute_unchecked_size::<_, Pair<_, _>>(a).unpack()
}

pub(crate) const unsafe fn merge_transmute<A, B, C>(a: A, b: B) -> C
{
    transmute_unchecked_size(Pair::new(a, b))
}

pub(crate) const unsafe fn overlap_swap_transmute<A, B>(a: A, b: B) -> (B, A)
{
    split_transmute(Pair::new(a, b))
}

pub(crate) const unsafe fn transmute_unchecked_size<A, B>(from: A) -> B
{
    /*#[cfg(test)]
    if core::mem::size_of::<A>() != core::mem::size_of::<B>() && core::mem::align_of::<A>() != core::mem::align_of::<B>()
    {
        panic!("Cannot transmute due to unequal size or alignment")
    }*/
    
    let b = unsafe {core::mem::transmute_copy(&from)};
    core::mem::forget(from);
    b

    //core::ptr::read(core::mem::transmute(&ManuallyDrop::new(from)))
    
    /*union Transmutation<A, B>
    {
        a: ManuallyDrop<A>,
        b: ManuallyDrop<B>
    }

    unsafe {ManuallyDrop::into_inner(Transmutation {a: ManuallyDrop::new(a)}.b)}*/
}