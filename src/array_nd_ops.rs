use core::{borrow::{Borrow, BorrowMut}, ops::{Deref, DerefMut}, marker::Destruct, mem::{MaybeUninit, ManuallyDrop}};

use array_trait::{ArrayNd, Array};

use super::*;

/// A trait for N-dimensional arrays
pub trait ArrayNdOps<const D: usize, T, const L: usize>: Array + ArrayNd<D, ItemNd = T, /*FLAT_LENGTH = {L}*/>
{
    type Mapped<M>: /*~const*/ ArrayNdOps<D, M, L>;

    fn as_ptr_nd(&self) -> *const T;
    fn as_mut_ptr_nd(&mut self) -> *mut T;

    /// Fills an N-dimensional array. Indices passed to fill-function are sorted from outermost to innermost.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(const_trait_impl)]
    /// #![feature(const_closures)]
    /// #![feature(const_mut_refs)]
    /// 
    /// use array__ops::*;
    /// 
    /// type T = u8;
    /// 
    /// let nd: [[T; 3]; 3] = ArrayNdOps::fill_nd(|[i, j]| 1 + 3*i as T + j as T);
    /// 
    /// assert_eq!(nd, [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ]);
    /// ```
    fn fill_nd<F>(fill: F) -> Self
    where
        F: /*~const*/ FnMut([usize; D]) -> T + /*~const*/ Destruct;

    /// Maps each element in the N-dimensional array.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(const_trait_impl)]
    /// #![feature(const_closures)]
    /// #![feature(const_mut_refs)]
    /// 
    /// use array__ops::*;
    /// 
    /// const ND: [[u8; 3]; 3] = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ];
    /// 
    /// let nd_mapped: [[i8; 3]; 3] = ND.map_nd(const |x: u8| -(x as i8));
    /// 
    /// assert_eq!(nd_mapped, [
    ///     [-1, -2, -3],
    ///     [-4, -5, -6],
    ///     [-7, -8, -9]
    /// ]);
    /// ```
    fn map_nd<M>(self, map: M) -> Self::Mapped<<M as FnOnce<(T,)>>::Output>
    where
        M: /*~const*/ FnMut<(T,)> + /*~const*/ Destruct;

    /// Enumerates each element of an N-dimensional array. Indices are sorted from outermost to innermost.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(const_trait_impl)]
    /// #![feature(const_closures)]
    /// #![feature(const_mut_refs)]
    /// #![feature(generic_arg_infer)]
    /// 
    /// use array__ops::*;
    /// 
    /// type T = u8;
    /// 
    /// const ND: [[T; 3]; 3] = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ];
    /// 
    /// // For now, the compiler cannot infer the type, so type annotations are needed.
    /// let nd_enum: [[([usize; 2], T); 3]; 3] = <[[T; 3]; 3] as ArrayNdOps<2, _, _>>::enumerate_nd(ND);
    /// 
    /// assert_eq!(nd_enum, [
    ///     [([0, 0], 1), ([0, 1], 2), ([0, 2], 3)],
    ///     [([1, 0], 4), ([1, 1], 5), ([1, 2], 6)],
    ///     [([2, 0], 7), ([2, 1], 8), ([2, 2], 9)]
    /// ]);
    /// ```
    fn enumerate_nd(self) -> Self::Mapped<([usize; D], T)>;

    /// Flattens one or multiple dimensions of an N-dimensional array.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(const_trait_impl)]
    /// 
    /// use array__ops::*;
    /// 
    /// type T = u8;
    /// 
    /// const ND: [[T; 3]; 3] = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ];
    /// let flat: [T; 9] = ND.flatten_nd_array();
    /// assert_eq!(flat, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// ```
    fn flatten_nd_array(self) -> [T; L]
    where
        [(); L]:;

    /// Flattens one or multiple dimensions of an N-dimensional array-slice.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(const_trait_impl)]
    /// 
    /// use array__ops::*;
    /// 
    /// type T = u8;
    /// 
    /// const ND: [[T; 3]; 3] = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ];
    /// let flat: &[T; 9] = ND.flatten_nd_array_ref();
    /// assert_eq!(flat, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// ```
    fn flatten_nd_array_ref(&self) -> &[T; L]
    where
        [(); L]:;
    
    /// Flattens one or multiple dimensions of an N-dimensional array-slice
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array__ops::*;
    /// 
    /// type T = u8;
    /// 
    /// let mut nd: [[T; 3]; 3] = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ];
    /// let flat: &mut [T; 9] = nd.flatten_nd_array_mut();
    /// 
    /// for x in flat.into_iter()
    /// {
    ///     *x = 10 - *x;
    /// }
    /// 
    /// assert_eq!(nd, [
    ///     [9, 8, 7],
    ///     [6, 5, 4],
    ///     [3, 2, 1]
    /// ]);
    /// ```
    fn flatten_nd_array_mut(&mut self) -> &mut [T; L]
    where
        [(); L]:;

    fn each_ref_nd(&self) -> Self::Mapped<&T>;
    fn each_mut_nd(&mut self) -> Self::Mapped<&mut T>;

    /// Reduces inner elements in N-dimensional array into one element, using a given operand
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array__ops::*;
    /// 
    /// const A: [[(u8, u8); 3]; 2] = [
    ///     [(0, 0), (0, 1), (0, 2)],
    ///     [(1, 0), (1, 1), (1, 2)]
    /// ];
    /// 
    /// let r: (u8, u8) = A.reduce_nd(|(a1, a2), (b1, b2)| (a1 + b1, a2 + b2)).unwrap();
    /// 
    /// assert_eq!(r, (3, 6));
    /// ```
    fn reduce_nd<R>(self, reduce: R) -> Option<T>
    where
        R: /*~const*/ FnMut(T, T) -> T + /*~const*/ Destruct,
        T: /*~const*/ Destruct;
        
    /// Retrieves the inner item using an array of indices, sorted from outermost to innermost, as a reference
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(const_closures)]
    /// #![feature(const_option)]
    /// #![feature(const_trait_impl)]
    /// #![feature(generic_const_exprs)]
    /// #![feature(const_mut_refs)]
    /// 
    /// use array__ops::*;
    /// 
    /// const A: [[u8; 2]; 3] = [
    ///     [1, 2],
    ///     [3, 4],
    ///     [5, 6]
    /// ];
    /// let b: [[u8; 2]; 3] = ArrayNdOps::fill_nd(|[i, j]| {
    ///     let item = *A.get_nd([i, j]).unwrap();
    ///     assert_eq!(item, A[i][j]);
    ///     item
    /// });
    /// 
    /// assert_eq!(A, b);
    /// ```
    fn get_nd(&self, i: [usize; D]) -> Option<&T>;
    
    /// Retrieves the inner item using an array of indices, sorted from outermost to innermost, as a mutable reference
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(const_closures)]
    /// #![feature(const_option)]
    /// #![feature(const_trait_impl)]
    /// #![feature(generic_const_exprs)]
    /// #![feature(const_mut_refs)]
    /// 
    /// use array__ops::*;
    /// 
    /// const A: [[u8; 2]; 3] = [
    ///     [1, 2],
    ///     [3, 4],
    ///     [5, 6]
    /// ];
    /// let mut b: [[u8; 2]; 3] = [[0; 2]; 3];
    /// 
    /// let mut i = 0;
    /// while i < 3
    /// {
    ///     let mut j = 0;
    ///     while j < 2
    ///     {
    ///         let item = *A.get_nd([i, j]).unwrap();
    ///         assert_eq!(item, A[i][j]);
    ///         *b.get_nd_mut([i, j]).unwrap() = item;
    ///         j += 1;
    ///     }
    ///     i += 1;
    /// }
    /// 
    /// assert_eq!(A, b);
    /// ```
    fn get_nd_mut(&mut self, i: [usize; D]) -> Option<&mut T>;
}


macro_rules! count {
    () => {0};
    ($a:ident) => {1};
    ($a:ident $($b:ident)+) => {1 $(+ count!($b))+};
}
macro_rules! flat_len {
    () => {0};
    ($a:ident $($b:ident)*) => {$a $(* $b)*}
}

macro_rules! nd {
    ($t:ty;) => {
        $t
    };
    ($t:ty; $a:ident) => {
        [$t; $a]
    };
    ($t:ty; $a:ident $($b:ident)+) => {
        [nd!{$t; $($b)+}; $a]
    };
}

macro_rules! fill_nd {
    (($fill:ident, $dims:ident, $i:ident, $array:ident); $($c:ident)*) => {
        core::mem::swap($array, &mut MaybeUninit::new($fill($i)));
    };
    (($fill:ident, $dims:ident, $i:ident, $array:ident) $a:ident $($b:ident)*; $($c:ident)*) => {
        const J: usize = count!($($c)*);
        $i[J] = 0;
        while $i[J] < $dims[J]
        {
            let array = &mut $array[$i[J]];
            fill_nd!(($fill, $dims, $i, array) $($b)*; $a $($c)*);
            $i[J] += 1;
        }
    };
}

macro_rules! index_nd {
    (($this:tt.$fn:ident($i:ident)) $a:ident; $($c:ident)*) => {
        $this.$fn($i[count!{$($c)*}])
    };
    (($this:tt.$fn:ident($i:ident)) $a:ident $($b:ident)+; $($c:ident)*) => {
        $this.$fn($i[count!{$($c)*}])
            .and_then(|item| index_nd!{(item.$fn($i)) $($b)+; $a $($c)*})
    };
}

macro_rules! impl_nd_array {
    ($a:ident $($($b:ident)+)?) => {
        impl<T, const $a: usize $($(, const $b: usize)+)?> /*const*/ ArrayNdOps<{count!{$a $($($b)+)?}}, T, {flat_len!{$a $($($b)+)?}}> for nd!{T; $a $($($b)+)?}
        {
            type Mapped<M> = nd!{M; $a $($($b)+)?};

            fn as_ptr_nd(&self) -> *const T
            {
                self.as_ptr().cast()
            }
            fn as_mut_ptr_nd(&mut self) -> *mut T
            {
                self.as_mut_ptr().cast()
            }

            fn fill_nd<F>(mut fill: F) -> Self
            where
                F: /*~const*/ FnMut([usize; count!{$a $($($b)+)?}]) -> T + /*~const*/ Destruct
            {
                let dims: [usize; {count!{$a $($($b)+)?}}] = Self::DIMENSIONS;
                let mut i = [0; {count!{$a $($($b)+)?}}];
                let mut array: nd!{MaybeUninit<T>; $a $($($b)+)?} =
                    unsafe {private::transmute_unchecked_size(MaybeUninit::<nd!{MaybeUninit<T>; $($($b)+)?}>::uninit_array::<$a>())};
                while i[0] < dims[0]
                {
                    let array = &mut array[i[0]];
                    fill_nd!((fill, dims, i, array) $($($b)+)?; $a);
                    i[0] += 1;
                }
                unsafe {private::transmute_unchecked_size(array)}
            }

            fn map_nd<M>(self, mut map: M) -> Self::Mapped<<M as FnOnce<(T,)>>::Output>
            where
                M: /*~const*/ FnMut<(T,)> + /*~const*/ Destruct
            {
                let mut iter = ManuallyDrop::new(self.flatten_nd_array().into_iter());
                ArrayNdOps::fill_nd(|_| map(iter.deref_mut().next().unwrap()))
            }
            
            fn enumerate_nd(self) -> Self::Mapped<([usize; {count!{$a $($($b)+)?}}], T)>
            {
                let mut iter = ManuallyDrop::new(self.flatten_nd_array().into_iter());
                ArrayNdOps::fill_nd(|i| (i, iter.deref_mut().next().unwrap()))
            }

            fn flatten_nd_array(self) -> [T; {flat_len!{$a $($($b)+)?}}]
            where
                [(); {flat_len!{$a $($($b)+)?}}]:
            {
                unsafe {private::transmute_unchecked_size(self)}
            }

            fn flatten_nd_array_ref(&self) -> &[T; {flat_len!{$a $($($b)+)?}}]
            where
                [(); {flat_len!{$a $($($b)+)?}}]:
            {
                unsafe {core::mem::transmute(self)}
            }

            fn flatten_nd_array_mut(&mut self) -> &mut [T; {flat_len!{$a $($($b)+)?}}]
            where
                [(); {flat_len!{$a $($($b)+)?}}]:
            {
                unsafe {core::mem::transmute(self)}
            }
            
            fn each_ref_nd(&self) -> Self::Mapped<&T>
            {
                let mut ptr = unsafe {core::mem::transmute::<_, *const T>(self)};
                ArrayNdOps::fill_nd(|_| {
                    let y = unsafe {core::mem::transmute::<_, &T>(ptr)};
                    ptr = unsafe {ptr.add(1)};
                    y
                })
            }
            fn each_mut_nd(&mut self) -> Self::Mapped<&mut T>
            {
                let mut ptr = unsafe {core::mem::transmute::<_, *mut T>(self)};
                ArrayNdOps::fill_nd(|_| {
                    let y = unsafe {core::mem::transmute::<_, &mut T>(ptr)};
                    ptr = unsafe {ptr.add(1)};
                    y
                })
            }
            
            fn reduce_nd<R>(self, mut reduce: R) -> Option<T>
            where
                R: /*~const*/ FnMut(T, T) -> T + /*~const*/ Destruct,
                T: /*~const*/ Destruct
            {
                let this = ManuallyDrop::new(self);
                if flat_len!{$a $($($b)+)?} == 0
                {
                    return None
                }
                let mut i = 1;
                unsafe {
                    let mut ptr: *const T = core::mem::transmute(this.deref());
                    let mut reduction = core::ptr::read(ptr);
                    while i < flat_len!{$a $($($b)+)?}
                    {
                        ptr = ptr.add(1);
                        reduction = reduce(reduction, core::ptr::read(ptr));
                        i += 1;
                    }
                    Some(reduction)
                }
            }

            fn get_nd(&self, i: [usize; count!{$a $($($b)+)?}]) -> Option<&T>
            {
                index_nd!{(self.get(i)) $a $($($b)+)?;}
            }

            fn get_nd_mut(&mut self, i: [usize; count!{$a $($($b)+)?}]) -> Option<&mut T>
            {
                index_nd!{(self.get_mut(i)) $a $($($b)+)?;}
            }
        }
        $(impl_nd_array!($($b)+);)?
    };
}

mod r#impl
{
    use super::*;

    impl_nd_array!(
        _0 _1 _2 _3 _4 _5 _6 _7 _8 _9 _10 _11 _12 _13 _14 _15 _16
    );
}