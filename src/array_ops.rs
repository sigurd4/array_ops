use core::{borrow::{Borrow, BorrowMut}, alloc::Allocator, marker::Destruct, mem::{ManuallyDrop, MaybeUninit}, ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign}, simd::{LaneCount, Simd, SimdElement, SupportedLaneCount}};

use array_trait::Array;
use slice_ops::Padded;

use super::*;

#[const_trait]
pub trait ArrayOps<T, const N: usize>: Array + IntoIterator<Item = T>
    + Borrow<[T; N]>
    + BorrowMut<[T; N]>
{
    fn split_len(n: usize) -> (usize, usize);
    fn rsplit_len(n: usize) -> (usize, usize);
        
    fn split_ptr(&self, n: usize) -> (*const T, *const T);
    fn split_mut_ptr(&mut self, n: usize) -> (*mut T, *mut T);

    fn rsplit_ptr(&self, n: usize) -> (*const T, *const T);
    fn rsplit_mut_ptr(&mut self, n: usize) -> (*mut T, *mut T);

    fn fill<F>(fill: F) -> Self
    where
        F: FnMut(usize) -> T + ~const Destruct;
    fn rfill<F>(fill: F) -> Self
    where
        F: FnMut(usize) -> T + ~const Destruct;
        
    #[cfg(feature = "std")]
    fn fill_boxed<F>(fill: F) -> Box<Self>
    where
        F: FnMut(usize) -> T + ~const Destruct;
    #[cfg(feature = "std")]
    fn rfill_boxed<F>(fill: F) -> Box<Self>
    where
        F: FnMut(usize) -> T + ~const Destruct;
        
    #[cfg(feature = "std")]
    fn fill_boxed_in<F, A>(fill: F, alloc: A) -> Box<Self, A>
    where
        F: FnMut(usize) -> T + ~const Destruct,
        A: Allocator;
    #[cfg(feature = "std")]
    fn rfill_boxed_in<F, A>(fill: F, alloc: A) -> Box<Self, A>
    where
        F: FnMut(usize) -> T + ~const Destruct,
        A: Allocator;

    fn truncate<const M: usize>(self) -> [T; M]
    where
        T: ~const Destruct,
        [(); N - M]:;
    fn rtruncate<const M: usize>(self) -> [T; M]
    where
        T: ~const Destruct,
        [(); N - M]:;
        
    fn truncate_ref<const M: usize>(&self) -> &[T; M]
    where
        [(); N - M]:;
    fn rtruncate_ref<const M: usize>(&self) -> &[T; M]
    where
        [(); N - M]:;
        
    fn truncate_mut<const M: usize>(&mut self) -> &mut [T; M]
    where
        [(); N - M]:;
    fn rtruncate_mut<const M: usize>(&mut self) -> &mut [T; M]
    where
        [(); N - M]:;

    fn resize<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: FnMut(usize) -> T + ~const Destruct,
        T: ~const Destruct;
    fn rresize<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: FnMut(usize) -> T + ~const Destruct,
        T: ~const Destruct;

    fn extend<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: FnMut(usize) -> T + ~const Destruct,
        [(); M - N]:;
    fn rextend<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: FnMut(usize) -> T + ~const Destruct,
        [(); M - N]:;

    fn reformulate_length<const M: usize>(self) -> [T; M]
    where
        [(); M - N]:,
        [(); N - M]:;
    
    fn reformulate_length_ref<const M: usize>(&self) -> &[T; M]
    where
        [(); M - N]:,
        [(); N - M]:;
        
    fn reformulate_length_mut<const M: usize>(&mut self) -> &mut [T; M]
    where
        [(); M - N]:,
        [(); N - M]:;
        
    fn try_reformulate_length<const M: usize>(self) -> Result<[T; M], Self>;
    
    fn try_reformulate_length_ref<const M: usize>(&self) -> Option<&[T; M]>;
        
    fn try_reformulate_length_mut<const M: usize>(&mut self) -> Option<&mut [T; M]>;

    /// Maps all values of an array with a given function.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array__ops::*;
    /// 
    /// const A: [u8; 4] = [1, 2, 3, 4];
    /// let b = A.map2(|b| -(b as i8));
    /// 
    /// assert_eq!(b, [-1, -2, -3, -4]);
    /// ```
    fn map2<Map>(self, map: Map) -> [Map::Output; N]
    where
        Map: FnMut<(T,)> + ~const Destruct;
    fn map_outer<Map>(&self, map: Map) -> [[Map::Output; N]; N]
    where
        Map: FnMut<(T, T)> + ~const Destruct,
        T: Copy;
    fn comap<Map, Rhs>(self, rhs: [Rhs; N], map: Map) -> [Map::Output; N]
    where
        Map: FnMut<(T, Rhs)> + ~const Destruct;
    fn comap_outer<Map, Rhs, const M: usize>(&self, rhs: &[Rhs; M], map: Map) -> [[Map::Output; M]; N]
    where
        Map: FnMut<(T, Rhs)> + ~const Destruct,
        T: Copy,
        Rhs: Copy;
    fn flat_map<Map, O, const M: usize>(self, map: Map) -> [O; N*M]
    where
        Map: FnMut<(T,), Output = [O; M]> + ~const Destruct;

    /// Combines two arrays with possibly different items into parallel, where each element lines up in the same position.
    /// 
    /// This method can be executed at compile-time, as opposed to the standard-library method.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array__ops::*;
    /// 
    /// const A: [u8; 4] = [4, 3, 2, 1];
    /// const B: [&str; 4] = ["four", "three", "two", "one"];
    /// let c = A.zip(B);
    /// 
    /// assert_eq!(c, [(4, "four"), (3, "three"), (2, "two"), (1, "one")]);
    /// ```
    fn zip<Z>(self, other: [Z; N]) -> [(T, Z); N];
    fn zip_outer<Z, const M: usize>(&self, other: &[Z; M]) -> [[(T, Z); M]; N]
    where
        T: Copy,
        Z: Copy;

    fn enumerate(self) -> [(usize, T); N];

    fn diagonal<const H: usize, const W: usize>(self) -> [[T; W]; H]
    where
        T: Default + Copy,
        [(); H - N]:,
        [(); W - N]:;
    
    /// Differentiates array (discrete calculus)
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array__ops::*;
    /// 
    /// let mut a = [1, 2, 3];
    /// 
    /// a.differentiate();
    /// 
    /// assert_eq!(a, [1, 2 - 1, 3 - 2]);
    /// ```
    fn differentiate(&mut self)
    where
        T: SubAssign<T> + Copy;
    
    /// Integrates array (discrete calculus)
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array__ops::*;
    /// 
    /// let mut a = [1, 2, 3];
    /// 
    /// a.integrate();
    /// 
    /// assert_eq!(a, [1, 1 + 2, 1 + 2 + 3])
    /// ```
    fn integrate(&mut self)
    where
        T: AddAssign<T> + Copy;

    /// Reduces elements in array into one element, using a given operand
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array__ops::ArrayOps;
    /// 
    /// const A: [u8; 3] = [1, 2, 3];
    /// 
    /// let r: u8 = A.reduce(|a, b| a + b).unwrap();
    /// 
    /// assert_eq!(r, 6);
    /// ```
    fn reduce<R>(self, reduce: R) -> Option<T>
    where
        R: FnMut(T, T) -> T + ~const Destruct;

    fn try_sum(self) -> Option<T>
    where
        T: AddAssign;
        
    fn sum_from<S>(self, from: S) -> S
    where
        S: AddAssign<T>;
        
    fn try_product(self) -> Option<T>
    where
        T: MulAssign;
        
    fn product_from<P>(self, from: P) -> P
    where
        P: MulAssign<T>;

    fn max(self) -> Option<T>
    where
        T: Ord;
        
    fn min(self) -> Option<T>
    where
        T: Ord;
        
    fn first_max(self) -> Option<T>
    where
        T: PartialOrd<T>;
        
    fn first_min(self) -> Option<T>
    where
        T: PartialOrd<T>;
        
    fn argmax(&self) -> Option<usize>
    where
        T: PartialOrd<T>;
        
    fn argmin(&self) -> Option<usize>
    where
        T: PartialOrd<T>;

    fn add_all<Rhs>(self, rhs: Rhs) -> [<T as Add<Rhs>>::Output; N]
    where
        T: Add<Rhs>,
        Rhs: Copy;
    fn sub_all<Rhs>(self, rhs: Rhs) -> [<T as Sub<Rhs>>::Output; N]
    where
        T: Sub<Rhs>,
        Rhs: Copy;
    fn mul_all<Rhs>(self, rhs: Rhs) -> [<T as Mul<Rhs>>::Output; N]
    where
        T: Mul<Rhs>,
        Rhs: Copy;
    fn div_all<Rhs>(self, rhs: Rhs) -> [<T as Div<Rhs>>::Output; N]
    where
        T: Div<Rhs>,
        Rhs: Copy;
    fn rem_all<Rhs>(self, rhs: Rhs) -> [<T as Rem<Rhs>>::Output; N]
    where
        T: Rem<Rhs>,
        Rhs: Copy;
    fn shl_all<Rhs>(self, rhs: Rhs) -> [<T as Shl<Rhs>>::Output; N]
    where
        T: Shl<Rhs>,
        Rhs: Copy;
    fn shr_all<Rhs>(self, rhs: Rhs) -> [<T as Shr<Rhs>>::Output; N]
    where
        T: Shr<Rhs>,
        Rhs: Copy;
    fn bitor_all<Rhs>(self, rhs: Rhs) -> [<T as BitOr<Rhs>>::Output; N]
    where
        T: BitOr<Rhs>,
        Rhs: Copy;
    fn bitand_all<Rhs>(self, rhs: Rhs) -> [<T as BitAnd<Rhs>>::Output; N]
    where
        T: BitAnd<Rhs>,
        Rhs: Copy;
    fn bitxor_all<Rhs>(self, rhs: Rhs) -> [<T as BitXor<Rhs>>::Output; N]
    where
        T: BitXor<Rhs>,
        Rhs: Copy;

    fn add_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: AddAssign<Rhs>,
        Rhs: Copy;
    fn sub_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: SubAssign<Rhs>,
        Rhs: Copy;
    fn mul_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: MulAssign<Rhs>,
        Rhs: Copy;
    fn div_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: DivAssign<Rhs>,
        Rhs: Copy;
    fn rem_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: RemAssign<Rhs>,
        Rhs: Copy;
    fn shl_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: ShlAssign<Rhs>,
        Rhs: Copy;
    fn shr_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: ShrAssign<Rhs>,
        Rhs: Copy;
    fn bitor_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: BitOrAssign<Rhs>,
        Rhs: Copy;
    fn bitand_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: BitAndAssign<Rhs>,
        Rhs: Copy;
    fn bitxor_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: BitXorAssign<Rhs>,
        Rhs: Copy;
        
    fn add_all_neg<Rhs>(self, rhs: Rhs) -> [<Rhs as Sub<T>>::Output; N]
    where
        Rhs: Copy + Sub<T>;
    fn mul_all_inv<Rhs>(self, rhs: Rhs) -> [<Rhs as Div<T>>::Output; N]
    where
        Rhs: Copy + Div<T>;
    
    fn neg_all(self) -> [<T as Neg>::Output; N]
    where
        T: Neg;
    fn neg_assign_all(&mut self)
    where
        T: Neg<Output = T>;
    
    fn add_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Add<Rhs>>::Output; N]
    where
        T: Add<Rhs>;
    fn sub_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Sub<Rhs>>::Output; N]
    where
        T: Sub<Rhs>;
    fn mul_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Mul<Rhs>>::Output; N]
    where
        T: Mul<Rhs>;
    fn div_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Div<Rhs>>::Output; N]
    where
        T: Div<Rhs>;
    fn rem_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Rem<Rhs>>::Output; N]
    where
        T: Rem<Rhs>;
    fn shl_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Shl<Rhs>>::Output; N]
    where
        T: Shl<Rhs>;
    fn shr_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Shr<Rhs>>::Output; N]
    where
        T: Shr<Rhs>;
    fn bitor_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as BitOr<Rhs>>::Output; N]
    where
        T: BitOr<Rhs>;
    fn bitand_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as BitAnd<Rhs>>::Output; N]
    where
        T: BitAnd<Rhs>;
    fn bitxor_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as BitXor<Rhs>>::Output; N]
    where
        T: BitXor<Rhs>;

    fn add_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: AddAssign<Rhs>;
    fn sub_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: SubAssign<Rhs>;
    fn mul_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: MulAssign<Rhs>;
    fn div_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: DivAssign<Rhs>;
    fn rem_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: RemAssign<Rhs>;
    fn shl_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: ShlAssign<Rhs>;
    fn shr_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: ShrAssign<Rhs>;
    fn bitor_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: BitOrAssign<Rhs>;
    fn bitand_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: BitAndAssign<Rhs>;
    fn bitxor_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: BitXorAssign<Rhs>;

    fn try_mul_dot<Rhs>(self, rhs: [Rhs; N]) -> Option<<T as Mul<Rhs>>::Output>
    where
        T: Mul<Rhs, Output: AddAssign>;

    fn mul_dot_bias<Rhs>(self, rhs: [Rhs; N], bias: <T as Mul<Rhs>>::Output) -> <T as Mul<Rhs>>::Output
    where
        T: Mul<Rhs, Output: AddAssign>;

    fn mul_outer<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [[<T as Mul<Rhs>>::Output; M]; N]
    where
        T: Mul<Rhs> + Copy,
        Rhs: Copy;
        
    /// Computes the general cross-product of the two arrays (as if vectors, in the mathematical sense).
    /// 
    /// # Example
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array__ops::*;
    /// 
    /// const U: [f64; 3] = [1.0, 0.0, 0.0];
    /// const V: [f64; 3] = [0.0, 1.0, 0.0];
    /// 
    /// let w = U.mul_cross([&V]);
    /// 
    /// assert_eq!(w, [0.0, 0.0, 1.0]);
    /// ```
    fn mul_cross<Rhs>(&self, rhs: [&[Rhs; N]; N - 2]) -> [<T as Sub>::Output; N]
    where
        T: MulAssign<Rhs> + Sub + Copy,
        Rhs: Copy;

    fn try_magnitude_squared(self) -> Option<<T as Mul<T>>::Output>
    where
        T: Mul<T, Output: AddAssign> + Copy;

    /// Chains two arrays with the same item together.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array__ops::*;
    /// 
    /// let a = ["one", "two"];
    /// let b = ["three"];
    /// 
    /// assert_eq!(a.chain(b), ["one", "two", "three"]);
    /// ```
    fn chain<const M: usize>(self, rhs: [T; M]) -> [T; N + M];

    /// Chains two arrays with the same item together in reverse.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array__ops::*;
    /// 
    /// let a = ["two", "three"];
    /// let b = ["one"];
    /// 
    /// assert_eq!(a.rchain(b), ["one", "two", "three"]);
    /// ```
    fn rchain<const M: usize>(self, rhs: [T; M]) -> [T; N + M];
    
    fn into_rotate_left(self, n: usize) -> Self;

    fn into_rotate_right(self, n: usize) -> Self;

    fn into_shift_many_left<const M: usize>(self, items: [T; M]) -> ([T; M], Self);
        
    fn into_shift_many_right<const M: usize>(self, items: [T; M]) -> (Self, [T; M]);

    fn into_shift_left(self, item: T) -> (T, Self);
        
    fn into_shift_right(self, item: T) -> (Self, T);

    fn rotate_left2(&mut self, n: usize);

    fn rotate_right2(&mut self, n: usize);

    fn shift_many_left<const M: usize>(&mut self, items: [T; M]) -> [T; M];
    
    fn shift_many_right<const M: usize>(&mut self, items: [T; M]) -> [T; M];
    
    fn shift_left(&mut self, item: T) -> T;

    fn shift_right(&mut self, item: T) -> T;

    /// Distributes items of an array equally across a given width, then provides the rest as a separate array.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// 
    /// use array__ops::*;
    /// 
    /// let array = ["ping 1", "pong 1", "ping 2", "pong 2", "ping 3", "pong 3", "uhh..."];
    /// 
    /// let ([ping, pong], rest) = array.spread::<2>();
    /// 
    /// assert_eq!(ping, ["ping 1", "ping 2", "ping 3"]);
    /// assert_eq!(pong, ["pong 1", "pong 2", "pong 3"]);
    /// assert_eq!(rest, ["uhh..."]);
    /// ```
    fn spread<const M: usize>(self) -> ([[T; N / M]; M], [T; N % M])
    where
        [(); M - 1]:,
        [(); N / M]:,
        [(); N % M]:;

    /// Distributes items of an array-slice equally across a given width, then provides the rest as a separate array-slice.
    /// 
    /// The spread-out slices are given in padded arrays. Each padded item can be borrowed into a reference to the array's item.
    fn spread_ref<const M: usize>(&self) -> ([&[Padded<T, M>; N / M]; M], &[T; N % M])
    where
        [(); M - 1]:,
        [(); N % M]:;
    
    /// Distributes items of a mutable array-slice equally across a given width, then provides the rest as a separate mutable array-slice.
    /// 
    /// The spread-out slices are given in padded arrays. Each padded item can be borrowed into a reference to the array's item.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// 
    /// use array__ops::*;
    /// 
    /// let mut array = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"];
    /// 
    /// let (threes, _) = array.spread_mut::<3>();
    /// 
    /// for fizz in threes.into_iter().last().unwrap()
    /// {
    ///     **fizz = "fizz";
    /// }
    /// 
    /// let (fives, _) = array.spread_mut::<5>();
    /// 
    /// for buzz in fives.into_iter().last().unwrap()
    /// {
    ///     **buzz = "buzz";
    /// }
    /// 
    /// let (fifteens, _) = array.spread_mut::<15>();
    /// 
    /// for fizzbuzz in fifteens.into_iter().last().unwrap()
    /// {
    ///     **fizzbuzz = "fizzbuzz";
    /// }
    /// 
    /// assert_eq!(array, ["1", "2", "fizz", "4", "buzz", "fizz", "7", "8", "fizz", "buzz", "11", "fizz", "13", "14", "fizzbuzz", "16", "17", "fizz", "19", "buzz"]);
    /// 
    /// ```
    fn spread_mut<const M: usize>(&mut self) -> ([&mut [Padded<T, M>; N / M]; M], &mut [T; N % M])
    where
        [(); M - 1]:,
        [(); N % M]:;
    
    /// Distributes items of an array equally across a given width, then provides the leftmost rest as a separate array.
    fn rspread<const M: usize>(self) -> ([T; N % M], [[T; N / M]; M])
    where
        [(); M - 1]:,
        [(); N / M]:,
        [(); N % M]:,
        T: Copy;

    /// Distributes items of an array-slice equally across a given width, then provides the leftmost rest as a separate array-slice.
    /// 
    /// The spread-out slices are given in padded arrays. Each padded item can be borrowed into a reference to the array's item.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// #![feature(array_methods)]
    /// 
    /// use array__ops::*;
    /// 
    /// let array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    /// 
    /// let (zero, [odd, even]) = array.rspread_ref::<2>();
    /// 
    /// assert_eq!(*zero, [0]);
    /// assert_eq!(odd.each_ref().map(|padding| **padding), [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    /// assert_eq!(even.each_ref().map(|padding| **padding), [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    /// ```
    fn rspread_ref<const M: usize>(&self) -> (&[T; N % M], [&[Padded<T, M>; N / M]; M])
    where
        [(); M - 1]:,
        [(); N % M]:;
    /// Distributes items of a mutable array-slice equally across a given width, then provides the leftmost rest as a separate mutable array-slice.
    /// 
    /// The spread-out slices are given in padded arrays. Each padded item can be borrowed into a reference to the array's item.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// #![feature(array_methods)]
    /// 
    /// use array__ops::*;
    /// 
    /// let mut array = ["the", "beat", "goes", "1", "2", "3", "4", "5", "6", "7", "8"];
    /// 
    /// let (start, [boots, n, cats, and]) = array.rspread_mut::<4>();
    /// 
    /// for boots in boots
    /// {
    ///     **boots = "boots";
    /// }
    /// for n in n
    /// {
    ///     **n = "n";
    /// }
    /// for cats in cats
    /// {
    ///     **cats = "cats";
    /// }
    /// for and in and
    /// {
    ///     **and = "and";
    /// }
    /// 
    /// assert_eq!(array, ["the", "beat", "goes", "boots", "n", "cats", "and", "boots", "n", "cats", "and"]);
    /// ```
    fn rspread_mut<const M: usize>(&mut self) -> (&mut [T; N % M], [&mut [Padded<T, M>; N / M]; M])
    where
        [(); M - 1]:,
        [(); N % M]:;
    
    /// Distributes items of an array equally across a given width, with no rest.
    /// 
    /// The width must be a factor of the array length, otherwise it will not compile.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// 
    /// use array__ops::*;
    /// 
    /// let array = *b"aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ";
    /// 
    /// let [lower_case, upper_case] = array.spread_exact::<2>();
    /// 
    /// assert_eq!(lower_case, *b"abcdefghijklmnopqrstuvwxyz");
    /// assert_eq!(upper_case, *b"ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    /// ```
    fn spread_exact<const M: usize>(self) -> [[T; N / M]; M]
    where
        [(); M - 1]:,
        [(); 0 - N % M]:,
        [(); N / M]:;
    
    /// Distributes items of an array-slice equally across a given width, with no rest.
    /// 
    /// The width must be a factor of the array length, otherwise it will not compile.
    /// 
    /// The spread-out slices are given in padded arrays. Each padded item can be borrowed into a reference to the array's item.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// #![feature(array_methods)]
    /// 
    /// use array__ops::*;
    /// 
    /// let statement = ["s", "he", "be", "lie", "ve", "d"];
    /// 
    /// let [interpretation2, interpretation1] = statement.spread_exact_ref::<2>();
    /// 
    /// assert_eq!(interpretation1, &["he", "lie", "d"]);
    /// assert_eq!(interpretation2, &["s", "be", "ve"]);
    /// ```
    fn spread_exact_ref<const M: usize>(&self) -> [&[Padded<T, M>; N / M]; M]
    where
        [(); M - 1]:,
        [(); 0 - N % M]:;

    /// Distributes items of a mutable array-slice equally across a given width, with no rest.
    /// 
    /// The width must be a factor of the array length, otherwise it will not compile.
    /// 
    /// The spread-out slices are given in padded arrays. Each padded item can be borrowed into a reference to the array's item.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// #![feature(array_methods)]
    /// 
    /// use array__ops::*;
    /// 
    /// let mut array = *b"aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ";
    /// 
    /// let [lower_case, upper_case] = array.spread_exact_mut::<2>();
    /// 
    /// assert_eq!(lower_case.each_ref().map(|padding| padding.borrow()), b"abcdefghijklmnopqrstuvwxyz".each_ref());
    /// assert_eq!(upper_case.each_ref().map(|padding| padding.borrow()), b"ABCDEFGHIJKLMNOPQRSTUVWXYZ".each_ref());
    /// 
    /// for c in upper_case
    /// {
    ///     **c = b'_';
    /// }
    /// 
    /// assert_eq!(array, *b"a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u_v_w_x_y_z_")
    /// ```
    fn spread_exact_mut<const M: usize>(&mut self) -> [&mut [Padded<T, M>; N / M]; M]
    where
        [(); M - 1]:,
        [(); 0 - N % M]:;

    /// Divides an array into chunks, then yielding the rest in a separate array.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// 
    /// use array__ops::*;
    /// 
    /// let array = ["carrot", "potato", "beet", "tomato", "kiwi", "banana", "cherry", "peach", "strawberry", "nine volt batteries"];
    /// 
    /// let ([root_vegetables, technically_berries, stone_fruits], not_for_human_consumption) = array.chunks::<3>();
    /// 
    /// assert_eq!(root_vegetables, ["carrot", "potato", "beet"]);
    /// assert_eq!(technically_berries, ["tomato", "kiwi", "banana"]);
    /// assert_eq!(stone_fruits, ["cherry", "peach", "strawberry"]);
    /// assert_eq!(not_for_human_consumption, ["nine volt batteries"]);
    /// ```
    fn chunks<const M: usize>(self) -> ([[T; M]; N / M], [T; N % M])
    where
        [(); N % M]:,
        [(); N / M]:;
    /// Divides an array-slice into chunks, then yielding the rest in a separate array-slice.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// 
    /// use array__ops::*;
    /// 
    /// let transistors = ["2N3904", "2N2222A", "BC107", "AC127", "OC7", "NKT275", "2SK30A", "2N5458", "J108", "2N7000", "BS170"];
    /// 
    /// let ([silicon_bjts, germanium_bjts, jfets], mosfets) = transistors.chunks_ref::<3>();
    /// 
    /// assert_eq!(silicon_bjts, &["2N3904", "2N2222A", "BC107"]);
    /// assert_eq!(germanium_bjts, &["AC127", "OC7", "NKT275"]);
    /// assert_eq!(jfets, &["2SK30A", "2N5458", "J108"]);
    /// assert_eq!(mosfets, &["2N7000", "BS170"]);
    /// ```
    fn chunks_ref<const M: usize>(&self) -> (&[[T; M]; N / M], &[T; N % M])
    where
        [(); N % M]:,
        [(); N / M]:;
    /// Divides a mutable array-slice into chunks, then yielding the rest in a separate mutable array-slice.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// 
    /// use array__ops::*;
    /// 
    /// let mut array = [0, 1, 0, 1, 0, 1, 6];
    /// 
    /// let (pairs, last) = array.chunks_mut::<2>();
    /// 
    /// for (i, pair) in pairs.into_iter().enumerate()
    /// {
    ///     for number in pair
    ///     {
    ///         *number += i*2;
    ///     }
    /// }
    /// 
    /// assert_eq!(array, [0, 1, 2, 3, 4, 5, 6]);
    /// ```
    fn chunks_mut<const M: usize>(&mut self) -> (&mut [[T; M]; N / M], &mut [T; N % M])
    where
        [(); N % M]:,
        [(); N / M]:;
    
    /// Divides a mutable array-slice into chunks, then yielding the leftmost rest in a separate mutable array-slice.
    fn array_rchunks<const M: usize>(self) -> ([T; N % M], [[T; M]; N / M])
    where
        [(); N % M]:,
        [(); N / M]:;
    /// Divides an array-slice into chunks, then yielding the leftmost rest in a separate array-slice.
    fn array_rchunks_ref<const M: usize>(&self) -> (&[T; N % M], &[[T; M]; N / M])
    where
        [(); N % M]:,
        [(); N / M]:;
    /// Divides a mutable array-slice into chunks, then yielding the leftmost rest in a separate array-slice.
    fn array_rchunks_mut<const M: usize>(&mut self) -> (&mut [T; N % M], &mut [[T; M]; N / M])
    where
        [(); N % M]:,
        [(); N / M]:;
    
    /// Divides an array into chunks, with no rest.
    /// 
    /// The chunk length must be a factor of the array length, otherwise it will not compile.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// 
    /// use array__ops::*;
    /// 
    /// let array = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    /// 
    /// let [lower_half, upper_half] = array.chunks_exact::<5>();
    /// 
    /// assert_eq!(lower_half, [0.0, 0.1, 0.2, 0.3, 0.4]);
    /// assert_eq!(upper_half, [0.5, 0.6, 0.7, 0.8, 0.9]);
    /// ```
    fn chunks_exact<const M: usize>(self) -> [[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:;
    /// Divides an array-slice into chunks, with no rest.
    /// 
    /// The chunk length must be a factor of the array length, otherwise it will not compile.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(generic_arg_infer)]
    /// 
    /// use array__ops::*;
    /// 
    /// let array = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    /// 
    /// let [lower_half, upper_half] = array.chunks_exact_ref::<5>();
    /// 
    /// assert_eq!(lower_half, &[0.0, 0.1, 0.2, 0.3, 0.4]);
    /// assert_eq!(upper_half, &[0.5, 0.6, 0.7, 0.8, 0.9]);
    /// ```
    fn chunks_exact_ref<const M: usize>(&self) -> &[[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:;
    /// Divides a mutable array-slice into chunks, with no rest.
    /// 
    /// The chunk length must be a factor of the array length, otherwise it will not compile.
    fn chunks_exact_mut<const M: usize>(&mut self) -> &mut [[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:;

    fn array_simd<const M: usize>(self) -> ([Simd<T, M>; N / M], [T; N % M])
    where
        T: SimdElement,
        LaneCount<M>: SupportedLaneCount,
        [(); N % M]:,
        [(); N / M]:;
    
    fn array_rsimd<const M: usize>(self) -> ([T; N % M], [Simd<T, M>; N / M])
    where
        T: SimdElement,
        LaneCount<M>: SupportedLaneCount,
        [(); N % M]:,
        [(); N / M]:;
    
    fn array_simd_exact<const M: usize>(self) -> [Simd<T, M>; N / M]
    where
        T: SimdElement,
        LaneCount<M>: SupportedLaneCount,
        [(); 0 - N % M]:,
        [(); N / M]:;

    /// Splits an array at a chosen index.
    fn split_array<const M: usize>(self) -> ([T; M], [T; N - M])
    where
        [(); N - M]:;
    /// Splits an array at a chosen index as array-slices.
    fn split_array_ref2<const M: usize>(&self) -> (&[T; M], &[T; N - M])
    where
        [(); N - M]:;
    /// Splits an array at a chosen index as mutable array-slices.
    fn split_array_mut2<const M: usize>(&mut self) -> (&mut [T; M], &mut [T; N - M])
    where
        [(); N - M]:;
    
    /// Splits an array at a chosen index, where the index goes from right to left.
    fn rsplit_array<const M: usize>(self) -> ([T; N - M], [T; M])
    where
        [(); N - M]:;
    /// Splits an array at a chosen index as array-slices, where the index goes from right to left.
    fn rsplit_array_ref2<const M: usize>(&self) -> (&[T; N - M], &[T; M])
    where
        [(); N - M]:;
    /// Splits an array at a chosen index as mutable array-slices, where the index goes from right to left.
    fn rsplit_array_mut2<const M: usize>(&mut self) -> (&mut [T; N - M], &mut [T; M])
    where
        [(); N - M]:;

    fn each_ref2(&self) -> [&T; N];
    fn each_mut2(&mut self) -> [&mut T; N];
    
    /// Performs the bit-reverse permutation. Length must be a power of 2.
    /// 
    /// # Example
    /// ```rust
    /// use array__ops::*;
    /// 
    /// let mut arr = [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111];
    /// 
    /// arr.bit_reverse_permutation();
    /// 
    /// assert_eq!(arr, [0b000, 0b100, 0b010, 0b110, 0b001, 0b101, 0b011, 0b111])
    /// ```
    fn bit_reverse_permutation(&mut self)
    where
        [(); N.is_power_of_two() as usize - 1]:;
}

pub const fn split_ptr<T, const N: usize>(array: &[T; N], mid: usize) -> (*const T, *const T)
{
    let ptr = array.as_ptr();
    (ptr, unsafe {ptr.add(slice_ops::split_len(N, mid).0)})
}

pub const fn split_mut_ptr<T, const N: usize>(array: &mut [T; N], mid: usize) -> (*mut T, *mut T)
{
    let ptr = array.as_mut_ptr();
    (ptr, unsafe {ptr.add(slice_ops::split_len(N, mid).0)})
}

pub const fn rsplit_ptr<T, const N: usize>(array: &[T; N], mid: usize) -> (*const T, *const T)
{
    let ptr = array.as_ptr();
    (ptr, unsafe {ptr.add(slice_ops::rsplit_len(N, mid).0)})
}

pub const fn rsplit_mut_ptr<T, const N: usize>(array: &mut [T; N], mid: usize) -> (*mut T, *mut T)
{
    let ptr = array.as_mut_ptr();
    (ptr, unsafe {ptr.add(slice_ops::rsplit_len(N, mid).0)})
}

pub const fn truncate_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> &[T; M]
where
    [(); N - M]:
{
    crate::split_array_ref(array).0
}
pub const fn rtruncate_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> &[T; M]
where
    [(); N - M]:
{
    crate::rsplit_array_ref(array).1
}

pub const fn truncate_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> &mut [T; M]
where
    [(); N - M]:
{
    crate::split_array_mut(array).0
}
pub const fn rtruncate_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> &mut [T; M]
where
    [(); N - M]:
{
    crate::rsplit_array_mut(array).1
}

pub const fn into_rotate_left<T, const N: usize>(array: [T; N], n: usize) -> [T; N]
{
    let n = n % N;
    let mut rotated = MaybeUninit::<[T; N]>::uninit();

    let (left, right) = slice_ops::split_len(N, n);
    let (src_left, src_right) = crate::split_ptr(&array, n);

    unsafe {
        let (dst_left, dst_right) = crate::rsplit_mut_ptr(rotated.assume_init_mut(), n);

        core::ptr::copy_nonoverlapping(src_right, dst_left, right);
        core::ptr::copy_nonoverlapping(src_left, dst_right, left);
    }

    core::mem::forget(array);

    unsafe {
        MaybeUninit::assume_init(rotated)
    }
}

pub const fn into_rotate_right<T, const N: usize>(array: [T; N], n: usize) -> [T; N]
{
    let n = n % N;
    let mut rotated = MaybeUninit::<[T; N]>::uninit();

    let (left, right) = slice_ops::rsplit_len(N, n);
    let (src_left, src_right) = crate::rsplit_ptr(&array, n);

    unsafe {
        let (dst_left, dst_right) = crate::split_mut_ptr(rotated.assume_init_mut(), n);

        core::ptr::copy_nonoverlapping(src_right, dst_left, right);
        core::ptr::copy_nonoverlapping(src_left, dst_right, left);
    }

    core::mem::forget(array);

    unsafe {
        MaybeUninit::assume_init(rotated)
    }
}

pub const fn into_shift_many_left<T, const N: usize, const M: usize>(array: [T; N], items: [T; M]) -> ([T; M], [T; N])
{
    unsafe {
        private::overlap_swap_transmute(array, items)
    }
}

pub const fn into_shift_many_right<T, const N: usize, const M: usize>(array: [T; N], items: [T; M]) -> ([T; N], [T; M])
{
    unsafe {
        private::overlap_swap_transmute(items, array)
    }
}

pub const fn into_shift_left<T, const N: usize>(array: [T; N], item: T) -> (T, [T; N])
{
    unsafe {
        private::overlap_swap_transmute(array, item)
    }
}
pub const fn into_shift_right<T, const N: usize>(array: [T; N], item: T) -> ([T; N], T)
{
    unsafe {
        private::overlap_swap_transmute(item, array)
    }
}

pub const fn rotate_left<T, const N: usize>(array: &mut [T; N], n: usize)
{
    let n = n % N;
    unsafe {
        let mut buffer: [T; N] = private::uninit();

        let (left, right) = slice_ops::split_len(N, n);
        let (src_left, src_right) = crate::split_mut_ptr(&mut buffer, n);
        let (dst_left, dst_right) = crate::rsplit_mut_ptr(array, n);

        core::ptr::copy_nonoverlapping(
            dst_left,
            src_left,
            N
        );
        core::ptr::copy_nonoverlapping(
            src_right,
            dst_left,
            right
        );
        core::ptr::copy_nonoverlapping(
            src_left,
            dst_right,
            left
        );
        core::mem::forget(buffer);
    }
}

pub const fn rotate_right<T, const N: usize>(array: &mut [T; N], n: usize)
{
    let n = n % N;
    unsafe {
        let mut buffer: [T; N] = private::uninit();

        let (left, right) = slice_ops::rsplit_len(N, n);
        let (src_left, src_right) = crate::rsplit_mut_ptr(&mut buffer, n);
        let (dst_left, dst_right) = crate::split_mut_ptr(array, n);

        core::ptr::copy_nonoverlapping(
            dst_left,
            src_left,
            N
        );
        core::ptr::copy_nonoverlapping(
            src_right,
            dst_left,
            right
        );
        core::ptr::copy_nonoverlapping(
            src_left,
            dst_right,
            left
        );
        core::mem::forget(buffer);
    }
}

pub const fn shift_many_left<T, const N: usize, const M: usize>(array: &mut [T; N], items: [T; M]) -> [T; M]
{
    unsafe {
        let mut buffer: private::Pair<[T; M], [T; N]> = private::Pair::new(items, private::uninit());
        let buf_left = buffer.left.as_mut_ptr();
        let buf_right = buf_left.add(N);

        core::ptr::copy_nonoverlapping(buffer.left.as_ptr(), buf_right, M);
        core::ptr::copy_nonoverlapping(array.as_ptr(), buf_left, N);

        let (overflow, shifted) = buffer.unpack_mandrop();

        core::ptr::copy_nonoverlapping((&shifted as *const ManuallyDrop<[T; N]>).cast::<T>(), array.as_mut_ptr(), N);
        core::mem::forget(shifted);

        ManuallyDrop::into_inner(overflow)
    }
}

pub const fn shift_many_right<T, const N: usize, const M: usize>(array: &mut [T; N], items: [T; M]) -> [T; M]
{
    unsafe {
        let mut buffer: private::Pair<[T; N], [T; M]> = private::Pair::new(private::uninit(), items);
        let buf_left = buffer.left.as_mut_ptr();
        let buf_right = buf_left.add(M);

        core::ptr::copy_nonoverlapping(buffer.right.as_ptr(), buf_left, M);
        core::ptr::copy_nonoverlapping(array.as_ptr(), buf_right, N);

        let (shifted, overflow) = buffer.unpack_mandrop();

        core::ptr::copy_nonoverlapping((&shifted as *const ManuallyDrop<[T; N]>).cast::<T>(), array.as_mut_ptr(), N);
        core::mem::forget(shifted);

        ManuallyDrop::into_inner(overflow)
    }
}

pub const fn shift_left<T, const N: usize>(array: &mut [T; N], item: T) -> T
{
    unsafe {
        let mut buffer: private::Pair<T, [T; N]> = private::Pair::new(item, private::uninit());
        let buf_left = &mut buffer.left as *mut T;
        let buf_right = buf_left.add(N);

        core::ptr::copy_nonoverlapping(&buffer.left as *const T, buf_right, 1);
        core::ptr::copy_nonoverlapping(array.as_ptr(), buf_left, N);

        let (overflow, shifted) = buffer.unpack_mandrop();

        core::ptr::copy_nonoverlapping((&shifted as *const ManuallyDrop<[T; N]>).cast::<T>(), array.as_mut_ptr(), N);
        core::mem::forget(shifted);

        ManuallyDrop::into_inner(overflow)
    }
}
pub const fn shift_right<T, const N: usize>(array: &mut [T; N], item: T) -> T
{
    unsafe {
        let mut buffer: private::Pair<[T; N], T> = private::Pair::new(private::uninit(), item);
        let buf_left = buffer.left.as_mut_ptr();
        let buf_right = buf_left.add(1);

        core::ptr::copy_nonoverlapping(&buffer.right as *const T, buf_left, 1);
        core::ptr::copy_nonoverlapping(array.as_ptr(), buf_right, N);

        let (shifted, overflow) = buffer.unpack_mandrop();

        core::ptr::copy_nonoverlapping((&shifted as *const ManuallyDrop<[T; N]>).cast::<T>(), array.as_mut_ptr(), N);
        core::mem::forget(shifted);

        ManuallyDrop::into_inner(overflow)
    }
}

pub const fn from_item<T>(item: T) -> [T; 1]
{
    [item]
}

pub const fn reformulate_length<T, const N: usize, const M: usize>(array: [T; N]) -> [T; M]
where
    [(); M - N]:,
    [(); N - M]:
{
    unsafe {private::transmute_unchecked_size(array)}
}
pub const fn reformulate_length_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> &[T; M]
where
    [(); M - N]:,
    [(); N - M]:
{
    unsafe {&*array.as_ptr().cast()}
}
pub const fn reformulate_length_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> &mut [T; M]
where
    [(); M - N]:,
    [(); N - M]:
{
    unsafe {&mut *array.as_mut_ptr().cast()}
}
pub const fn try_reformulate_length<T, const N: usize, const M: usize>(array: [T; N]) -> Result<[T; M], [T; N]>
{
    if N == M
    {
        Ok(unsafe {private::transmute_unchecked_size(array)})
    }
    else
    {
        Err(array)
    }
}
pub const fn try_reformulate_length_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> Option<&[T; M]>
{
    if N == M
    {
        Some(unsafe {&*array.as_ptr().cast()})
    }
    else
    {
        None
    }
}
pub const fn try_reformulate_length_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> Option<&mut [T; M]>
{
    if N == M
    {
        Some(unsafe {&mut *array.as_mut_ptr().cast()})
    }
    else
    {
        None
    }
}

pub const fn chain<T, const N: usize, const M: usize>(array: [T; N], rhs: [T; M]) -> [T; N + M]
{
    unsafe {private::merge_transmute(array, rhs)}
}
pub const fn rchain<T, const N: usize, const M: usize>(array: [T; N], rhs: [T; M]) -> [T; N + M]
{
    unsafe {private::merge_transmute(rhs, array)}
}

pub const fn spread<T, const N: usize, const M: usize>(array: [T; N]) -> ([[T; N / M]; M], [T; N % M])
where
    [(); M - 1]:,
    [(); N % M]:,
    [(); N / M]:
{
    let split = crate::chunks(array);

    let spread_t = unsafe {core::ptr::read(&split.0 as *const [[T; _]; _])};
    let rest = unsafe {core::ptr::read(&split.1 as *const [T; _])};
    core::mem::forget(split);

    (crate::transpose(spread_t), rest)
}

pub const fn rspread<T, const N: usize, const M: usize>(array: [T; N]) -> ([T; N % M], [[T; N / M]; M])
where
    [(); M - 1]:,
    [(); N % M]:,
    [(); N / M]:
{
    let split = crate::array_rchunks(array);
    
    let start = unsafe {core::ptr::read(&split.0 as *const [T; _])};
    let spread_t = unsafe {core::ptr::read(&split.1 as *const [[T; _]; _])};
    core::mem::forget(split);

    (start, crate::transpose(spread_t))
}

pub const fn spread_exact<T, const N: usize, const M: usize>(array: [T; N]) -> [[T; N / M]; M]
where
    [(); M - 1]:,
    [(); 0 - N % M]:,
    [(); N / M]:
{
    let spread_t: [[T; M]; N / M] = unsafe {
        private::transmute_unchecked_size(array)
    };
    crate::transpose(spread_t)
}

pub const fn chunks<T, const N: usize, const M: usize>(array: [T; N]) -> ([[T; M]; N / M], [T; N % M])
{
    unsafe {private::split_transmute(array)}
}
pub const fn chunks_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> (&[[T; M]; N / M], &[T; N % M])
{
    let (ptr_left, ptr_right) = crate::rsplit_ptr(array, N % M);
    unsafe {(&*ptr_left.cast(), &*ptr_right.cast())}
}
pub const fn chunks_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> (&mut [[T; M]; N / M], &mut [T; N % M])
{
    let (ptr_left, ptr_right) = crate::rsplit_mut_ptr(array, N % M);
    unsafe {(&mut *ptr_left.cast(), &mut *ptr_right.cast())}
}

pub const fn array_rchunks<T, const N: usize, const M: usize>(array: [T; N]) -> ([T; N % M], [[T; M]; N / M])
{
    unsafe {private::split_transmute(array)}
}
pub const fn array_rchunks_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> (&[T; N % M], &[[T; M]; N / M])
{
    let (ptr_left, ptr_right) = crate::split_ptr(array, N % M);
    unsafe {(&*ptr_left.cast(), &*ptr_right.cast())}
}
pub const fn array_rchunks_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> (&mut [T; N % M], &mut [[T; M]; N / M])
{
    let (ptr_left, ptr_right) = crate::split_mut_ptr(array, N % M);
    unsafe {(&mut *ptr_left.cast(), &mut *ptr_right.cast())}
}

pub const fn chunks_exact<T, const N: usize, const M: usize>(array: [T; N]) -> [[T; M]; N / M]
where
    [(); 0 - N % M]:,
    [(); N / M]:
{
    unsafe {private::transmute_unchecked_size(array)}
}
pub const fn chunks_exact_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> &[[T; M]; N / M]
where
    [(); 0 - N % M]:,
    [(); N / M]:
{
    unsafe {&*array.as_ptr().cast()}
}
pub const fn chunks_exact_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> &mut [[T; M]; N / M]
where
    [(); 0 - N % M]:,
    [(); N / M]:
{
    unsafe {&mut *array.as_mut_ptr().cast()}
}

pub const fn array_simd<T, const N: usize, const M: usize>(array: [T; N]) -> ([Simd<T, M>; N / M], [T; N % M])
where
    T: SimdElement,
    LaneCount<M>: SupportedLaneCount
{
    unsafe {private::split_transmute(array)}
}

pub const fn array_rsimd<T, const N: usize, const M: usize>(array: [T; N]) -> ([T; N % M], [Simd<T, M>; N / M])
where
    T: SimdElement,
    LaneCount<M>: SupportedLaneCount
{
    unsafe {private::split_transmute(array)}
}

pub const fn array_simd_exact<T, const N: usize, const M: usize>(array: [T; N]) -> [Simd<T, M>; N / M]
where
    T: SimdElement,
    LaneCount<M>: SupportedLaneCount,
    [(); 0 - N % M]:,
    [(); N / M]:
{
    unsafe {private::transmute_unchecked_size(array)}
}

pub const fn split_array<T, const N: usize, const M: usize>(array: [T; N]) -> ([T; M], [T; N - M])
where
    [(); N - M]:
{
    unsafe {private::split_transmute(array)}
}
pub const fn split_array_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> (&[T; M], &[T; N - M])
where
    [(); N - M]:
{
    let (ptr_left, ptr_right) = crate::split_ptr(array, M);
    unsafe {(&*ptr_left.cast(), &*ptr_right.cast())}
}
pub const fn split_array_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> (&mut [T; M], &mut [T; N - M])
where
    [(); N - M]:
{
    let (ptr_left, ptr_right) = crate::split_mut_ptr(array, M);
    unsafe {(&mut *ptr_left.cast(), &mut *ptr_right.cast())}
}

pub const fn rsplit_array<T, const N: usize, const M: usize>(array: [T; N]) -> ([T; N - M], [T; M])
where
    [(); N - M]:
{
    unsafe {private::split_transmute(array)}
}
pub const fn rsplit_array_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> (&[T; N - M], &[T; M])
where
    [(); N - M]:
{
    let (ptr_left, ptr_right) = crate::rsplit_ptr(array, M);
    unsafe {(&*ptr_left.cast(), &*ptr_right.cast())}
}
pub const fn rsplit_array_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> (&mut [T; N - M], &mut [T; M])
where
    [(); N - M]:
{
    let (ptr_left, ptr_right) = crate::rsplit_mut_ptr(array, M);
    unsafe {(&mut *ptr_left.cast(), &mut *ptr_right.cast())}
}

#[test]
fn bench()
{
    use std::time::SystemTime;

    const N: usize = 1 << 10;
    let mut a: [usize; N] = ArrayOps::fill(|i| i);
    let t0 = SystemTime::now();
    for _ in 0..1000
    {
        a.bit_reverse_permutation();
    }
    let dt = SystemTime::now().duration_since(t0);

    // 8.8810513s
    println!("{:?}", dt);
}

pub const fn bit_reverse_permutation<T, const N: usize>(array: &mut [T; N])
where
    [(); N.is_power_of_two() as usize - 1]:
{
    let mut i = 0;
    while i < N/2
    {
        let j = i.reverse_bits() >> (N.leading_zeros() + 1);
        if i != j
        {
            unsafe {
                core::ptr::swap_nonoverlapping(array.as_mut_ptr().add(i), array.as_mut_ptr().add(j), 1);
            }
        }
        i += 1;
    }
}

impl<T, const N: usize> ArrayOps<T, N> for [T; N]
{
    //type Array<I, const M: usize> = [I; M];
    
    fn split_len(mid: usize) -> (usize, usize)
    {
        slice_ops::split_len(N, mid)
    }
    fn rsplit_len(mid: usize) -> (usize, usize)
    {
        slice_ops::rsplit_len(N, mid)
    }
    
    fn split_ptr(&self, mid: usize) -> (*const T, *const T)
    {
        crate::split_ptr(self, mid)
    }
    fn split_mut_ptr(&mut self, mid: usize) -> (*mut T, *mut T)
    {
        crate::split_mut_ptr(self, mid)
    }

    fn rsplit_ptr(&self, mid: usize) -> (*const T, *const T)
    {
        crate::rsplit_ptr(self, mid)
    }
    fn rsplit_mut_ptr(&mut self, mid: usize) -> (*mut T, *mut T)
    {
        crate::rsplit_mut_ptr(self, mid)
    }

    fn fill<F>(mut fill: F) -> Self
    where
        F: FnMut(usize) -> T + Destruct
    {
        let mut array = MaybeUninit::uninit_array();
        let mut i = 0;
        while i != N
        {
            array[i] = MaybeUninit::new(fill(i));
            i += 1;
        }
        unsafe {MaybeUninit::array_assume_init(array)}
    }
    fn rfill<F>(mut fill: F) -> Self
    where
        F: FnMut(usize) -> T + Destruct
    {
        let mut array = MaybeUninit::uninit_array();
        if N != 0
        {
            let mut i = N - 1;
            loop
            {
                array[i] = MaybeUninit::new(fill(i));
                if i == 0
                {
                    break
                }
                i -= 1;
            }
        }
        unsafe {MaybeUninit::array_assume_init(array)}
    }
    
    #[cfg(feature = "std")]
    fn fill_boxed<F>(mut fill: F) -> Box<Self>
    where
        F: FnMut(usize) -> T + Destruct
    {
        let array = Box::new_uninit();
        let mut array: Box<[<F as FnOnce<(usize,)>>::Output; N]> = unsafe {
            array.assume_init()
        };
        let mut i = 0;
        while i < N
        {
            unsafe {
                array.as_mut_ptr().add(i).write(fill(i));
            }
            i += 1;
        }
        array
    }
    #[cfg(feature = "std")]
    fn rfill_boxed<F>(mut fill: F) -> Box<Self>
    where
        F: FnMut(usize) -> T + Destruct
    {
        let array = Box::new_uninit();
        let mut array: Box<[<F as FnOnce<(usize,)>>::Output; N]> = unsafe {
            array.assume_init()
        };
        if N != 0
        {
            let mut i = N - 1;
            loop
            {
                unsafe {
                    array.as_mut_ptr().add(i).write(fill(i));
                }
                if i == 0
                {
                    break
                }
                i -= 1;
            }
        }
        array
    }
    
    #[cfg(feature = "std")]
    fn fill_boxed_in<F, A>(mut fill: F, alloc: A) -> Box<Self, A>
    where
        F: FnMut(usize) -> T + Destruct,
        A: Allocator
    {
        let array = Box::new_uninit_in(alloc);
        let mut array: Box<[T; N], A> = unsafe {
            array.assume_init()
        };
        let mut i = 0;
        while i < N
        {
            unsafe {
                array.as_mut_ptr().add(i).write(fill(i));
            }
            i += 1;
        }
        array
    }
    #[cfg(feature = "std")]
    fn rfill_boxed_in<F, A>(mut fill: F, alloc: A) -> Box<Self, A>
    where
        F: FnMut(usize) -> T + Destruct,
        A: Allocator
    {
        let array = Box::new_uninit_in(alloc);
        let mut array: Box<[T; N], A> = unsafe {
            array.assume_init()
        };
        if N != 0
        {
            let mut i = N - 1;
            loop
            {
                unsafe {
                    array.as_mut_ptr().add(i).write(fill(i));
                }
                if i == 0
                {
                    break
                }
                i -= 1;
            }
        }
        array
    }
    
    fn truncate<const M: usize>(self) -> [T; M]
    where
        T: Destruct,
        [(); N - M]:
    {
        crate::split_array(self).0
    }
    fn rtruncate<const M: usize>(self) -> [T; M]
    where
        T: Destruct,
        [(); N - M]:
    {
        crate::rsplit_array(self).1
    }
    
    fn truncate_ref<const M: usize>(&self) -> &[T; M]
    where
        [(); N - M]:
    {
        crate::truncate_ref(self)
    }
    fn rtruncate_ref<const M: usize>(&self) -> &[T; M]
    where
        [(); N - M]:
    {
        crate::rtruncate_ref(self)
    }
        
    fn truncate_mut<const M: usize>(&mut self) -> &mut [T; M]
    where
        [(); N - M]:
    {
        crate::truncate_mut(self)
    }
    fn rtruncate_mut<const M: usize>(&mut self) -> &mut [T; M]
    where
        [(); N - M]:
    {
        crate::rtruncate_mut(self)
    }

    fn resize<const M: usize, F>(self, mut fill: F) -> [T; M]
    where
        F: FnMut(usize) -> T + Destruct,
        T: Destruct
    {
        let mut i = N.min(M);
        while i < N
        {
            let _ = unsafe {(&self[i] as *const T).read()};
            i += 1;
        }
    
        let mut dst = unsafe {private::uninit()};
        let mut ptr = &mut dst as *mut T;
    
        unsafe {core::ptr::copy_nonoverlapping(core::mem::transmute(&self), ptr, N.min(M))};
        core::mem::forget(self);
    
        let mut i = N;
        ptr = unsafe {ptr.add(N)};
        while i < M
        {
            unsafe {core::ptr::write(ptr, fill(i))};
            i += 1;
            ptr = unsafe {ptr.add(1)};
        }
        dst
    }
    fn rresize<const M: usize, F>(self, mut fill: F) -> [T; M]
    where
        F: FnMut(usize) -> T + Destruct,
        T: Destruct
    {
        let mut i = 0;
        while i < N.saturating_sub(M)
        {
            let _ = unsafe {(&self[i] as *const T).read()};
            i += 1;
        }
        
        let mut dst = unsafe {private::uninit()};
        let mut ptr = unsafe {(&mut dst as *mut T).add(M.saturating_sub(N))};
        
        unsafe {core::ptr::copy_nonoverlapping((&self as *const T).add(N.saturating_sub(M)), ptr, N.min(M))};
        core::mem::forget(self);
    
        let mut i = M.saturating_sub(N);
        while i > 0
        {
            i -= 1;
            ptr = unsafe {ptr.sub(1)};
            unsafe {core::ptr::write(ptr, fill(i))};
        }
    
        dst
    }

    fn into_rotate_left(self, n: usize) -> Self
    {
        crate::into_rotate_left(self, n)
    }
    
    fn into_rotate_right(self, n: usize) -> Self
    {
        crate::into_rotate_right(self, n)
    }

    fn into_shift_many_left<const M: usize>(self, items: [T; M]) -> ([T; M], Self)
    {
        crate::into_shift_many_left(self, items)
    }

    fn into_shift_many_right<const M: usize>(self, items: [T; M]) -> (Self, [T; M])
    {
        crate::into_shift_many_right(self, items)
    }

    fn into_shift_left(self, item: T) -> (T, Self)
    {
        crate::into_shift_left(self, item)
    }

    fn into_shift_right(self, item: T) -> (Self, T)
    {
        crate::into_shift_right(self, item)
    }

    fn rotate_left2(&mut self, n: usize)
    {
        crate::rotate_left(self, n)
    }

    fn rotate_right2(&mut self, n: usize)
    {
        crate::rotate_right(self, n)
    }

    fn shift_many_left<const M: usize>(&mut self, items: [T; M]) -> [T; M]
    {
        crate::shift_many_left(self, items)
    }

    fn shift_many_right<const M: usize>(&mut self, items: [T; M]) -> [T; M]
    {
        crate::shift_many_right(self, items)
    }
    
    fn shift_left(&mut self, item: T) -> T
    {
        crate::shift_left(self, item)
    }

    fn shift_right(&mut self, item: T) -> T
    {
        crate::shift_right(self, item)
    }
    
    fn extend<const M: usize, F>(self, mut fill: F) -> [T; M]
    where
        F: FnMut(usize) -> T + Destruct,
        [(); M - N]:
    {
        let filled: [T; M - N] = ArrayOps::fill(|i| fill(i + N));
        unsafe {private::merge_transmute(self, filled)}
    }
    fn rextend<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: FnMut(usize) -> T + Destruct,
        [(); M - N]:
    {
        let filled: [T; M - N] = ArrayOps::rfill(fill);
        unsafe {private::merge_transmute(filled, self)}
    }
    
    fn reformulate_length<const M: usize>(self) -> [T; M]
    where
        [(); M - N]:,
        [(); N - M]:
    {
        crate::reformulate_length(self)
    }
    
    fn reformulate_length_ref<const M: usize>(&self) -> &[T; M]
    where
        [(); M - N]:,
        [(); N - M]:
    {
        crate::reformulate_length_ref(self)
    }
        
    fn reformulate_length_mut<const M: usize>(&mut self) -> &mut [T; M]
    where
        [(); M - N]:,
        [(); N - M]:
    {
        crate::reformulate_length_mut(self)
    }
    
    fn try_reformulate_length<const M: usize>(self) -> Result<[T; M], Self>
    {
        crate::try_reformulate_length(self)
    }
    
    fn try_reformulate_length_ref<const M: usize>(&self) -> Option<&[T; M]>
    {
        crate::try_reformulate_length_ref(self)
    }
        
    fn try_reformulate_length_mut<const M: usize>(&mut self) -> Option<&mut [T; M]>
    {
        crate::try_reformulate_length_mut(self)
    }

    /*fn into_const_iter(self) -> IntoConstIter<T, N, true>
    {
        IntoConstIter::from(self)
    }
    fn into_const_iter_reverse(self) -> IntoConstIter<T, N, false>
    {
        IntoConstIter::from(self)
    }
    
    fn const_iter(&self) -> ConstIter<'_, T, N>
    {
        ConstIter::from(self)
    }
    fn const_iter_mut(&mut self) -> ConstIterMut<'_, T, N>
    {
        ConstIterMut::from(self)
    }*/
    
    fn map2<Map>(self, mut map: Map) -> [Map::Output; N]
    where
        Map: FnMut<(T,)> + Destruct
    {
        let ptr = &self as *const T;
    
        let dst = ArrayOps::fill(|i| unsafe {
            map(ptr.add(i).read())
        });
    
        core::mem::forget(self);
    
        dst
    }
    fn map_outer<Map>(&self, map: Map) -> [[Map::Output; N]; N]
    where
        Map: FnMut<(T, T)> + Destruct,
        T: Copy
    {
        ArrayOps::comap_outer(self, self, map)
    }

    fn comap<Map, Rhs>(self, rhs: [Rhs; N], mut map: Map) -> [Map::Output; N]
    where
        Map: FnMut<(T, Rhs)> + Destruct
    {
        let ptr0 = &self as *const T;
        let ptr1 = &rhs as *const Rhs;
    
        let dst = ArrayOps::fill(|i| unsafe {
            map(
                ptr0.add(i).read(),
                ptr1.add(i).read()
            )
        });
    
        core::mem::forget(self);
        core::mem::forget(rhs);
    
        dst
    }
    fn comap_outer<Map, Rhs, const M: usize>(&self, rhs: &[Rhs; M], mut map: Map) -> [[Map::Output; M]; N]
    where
        Map: FnMut<(T, Rhs)> + Destruct,
        T: Copy,
        Rhs: Copy
    {
        self.map2(|x| rhs.map2(|y| map(x, y)))
    }
    fn flat_map<Map, O, const M: usize>(self, map: Map) -> [O; N*M]
    where
        Map: FnMut<(T,), Output = [O; M]> + Destruct
    {
        let mapped = self.map2(map);
        unsafe {
            private::transmute_unchecked_size(mapped)
        }
    }
    
    fn zip<Z>(self, other: [Z; N]) -> [(T, Z); N]
    {
        self.comap(other, const |x, y| (x, y))
    }
    
    fn zip_outer<Z, const M: usize>(&self, other: &[Z; M]) -> [[(T, Z); M]; N]
    where
        T: Copy,
        Z: Copy
    {
        self.comap_outer(other, const |x, y| (x, y))
    }
    
    fn enumerate(self) -> [(usize, T); N]
    {
        let ptr = &self as *const T;
    
        let dst = ArrayOps::fill(|i| unsafe {
            (i, ptr.add(i).read())
        });
    
        core::mem::forget(self);
    
        dst
    }
    
    fn diagonal<const H: usize, const W: usize>(self) -> [[T; W]; H]
    where
        T: Default,
        [(); H - N]:,
        [(); W - N]:
    {
        let ptr = self.as_ptr();
        
        let dst = ArrayOps::fill(|i| ArrayOps::fill(|j| if i == j && i < N
            {
                unsafe {
                    ptr.add(i).read()
                }
            }
            else
            {
                T::default()
            }
        ));
    
        core::mem::forget(self);
    
        dst
    }

    fn differentiate(&mut self)
    where
        T: SubAssign<T> + Copy + Destruct
    {
        if N > 0
        {
            let mut i = N - 1;
            while i > 0
            {
                self[i] -= self[i - 1];
                i -= 1;
            }
        }
    }

    fn integrate(&mut self)
    where
        T: AddAssign<T> + Copy + Destruct
    {
        let mut i = 1;
        while i < N
        {
            self[i] += self[i - 1];
            i += 1;
        }
    }

    fn reduce<R>(self, mut reduce: R) -> Option<T>
    where
        R: FnMut(T, T) -> T + Destruct
    {
        let this = ManuallyDrop::new(self);
        if N == 0
        {
            return None
        }
        let ptr = this.deref() as *const T;
        let mut i = 1;
        unsafe {
            let mut reduction = core::ptr::read(ptr);
            while i < N
            {
                reduction = reduce(reduction, core::ptr::read(ptr.add(i)));
                i += 1;
            }
            Some(reduction)
        }
    }
    
    fn try_sum(self) -> Option<T>
    where
        T: AddAssign
    {
        let this = ManuallyDrop::new(self);
        if N == 0
        {
            return None
        }
        let ptr = this.deref() as *const T;
        let mut i = 1;
        unsafe {
            let mut reduction = core::ptr::read(ptr);
            while i < N
            {
                reduction += core::ptr::read(ptr.add(i));
                i += 1;
            }
            Some(reduction)
        }
    }

    fn sum_from<S>(self, mut from: S) -> S
    where
        S: AddAssign<T>
    {
        let this = ManuallyDrop::new(self);
        let ptr = this.deref() as *const T;
        let mut i = 0;
        unsafe {
            while i < N
            {
                from += core::ptr::read(ptr.add(i));
                i += 1;
            }
            from
        }
    }
        
    fn try_product(self) -> Option<T>
    where
        T: MulAssign
    {
        let this = ManuallyDrop::new(self);
        if N == 0
        {
            return None
        }
        let ptr = this.deref() as *const T;
        let mut i = 1;
        unsafe {
            let mut reduction = core::ptr::read(ptr);
            while i < N
            {
                reduction *= core::ptr::read(ptr.add(i));
                i += 1;
            }
            Some(reduction)
        }
    }

    fn product_from<P>(self, mut from: P) -> P
    where
        P: MulAssign<T>
    {
        let this = ManuallyDrop::new(self);
        let ptr = this.deref() as *const T;
        let mut i = 0;
        unsafe {
            while i < N
            {
                from *= core::ptr::read(ptr.add(i));
                i += 1;
            }
            from
        }
    }
    
    fn max(self) -> Option<T>
    where
        T: Ord
    {
        self.reduce(T::max)
    }
        
    fn min(self) -> Option<T>
    where
        T: Ord
    {
        self.reduce(T::min)
    }
    
    fn first_max(self) -> Option<T>
    where
        T: PartialOrd<T>
    {
        self.reduce(|a, b| if a >= b {a} else {b})
    }
        
    fn first_min(self) -> Option<T>
    where
        T: PartialOrd<T>
    {
        self.reduce(|a, b| if a <= b {a} else {b})
    }
    
    fn argmax(&self) -> Option<usize>
    where
        T: PartialOrd<T>
    {
        match self.each_ref2().enumerate().reduce(|a, b| if a.1 >= b.1 {a} else {b})
        {
            Some((i, _)) => Some(i),
            None => None
        }
    }
        
    fn argmin(&self) -> Option<usize>
    where
        T: PartialOrd<T>
    {
        match self.each_ref2().enumerate().reduce(|a, b| if a.1 <= b.1 {a} else {b})
        {
            Some((i, _)) => Some(i),
            None => None
        }
    }
    
    fn add_all<Rhs>(self, rhs: Rhs) -> [<T as Add<Rhs>>::Output; N]
    where
        T: Add<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x + rhs)
    }
    fn sub_all<Rhs>(self, rhs: Rhs) -> [<T as Sub<Rhs>>::Output; N]
    where
        T: Sub<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x - rhs)
    }
    fn mul_all<Rhs>(self, rhs: Rhs) ->  [<T as Mul<Rhs>>::Output; N]
    where
        T: Mul<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x * rhs)
    }
    fn div_all<Rhs>(self, rhs: Rhs) -> [<T as Div<Rhs>>::Output; N]
    where
        T: Div<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x / rhs)
    }
    fn rem_all<Rhs>(self, rhs: Rhs) -> [<T as Rem<Rhs>>::Output; N]
    where
        T: Rem<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x % rhs)
    }
    fn shl_all<Rhs>(self, rhs: Rhs) -> [<T as Shl<Rhs>>::Output; N]
    where
        T: Shl<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x << rhs)
    }
    fn shr_all<Rhs>(self, rhs: Rhs) -> [<T as Shr<Rhs>>::Output; N]
    where
        T: Shr<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x >> rhs)
    }
    fn bitor_all<Rhs>(self, rhs: Rhs) -> [<T as BitOr<Rhs>>::Output; N]
    where
        T: BitOr<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x | rhs)
    }
    fn bitand_all<Rhs>(self, rhs: Rhs) -> [<T as BitAnd<Rhs>>::Output; N]
    where
        T: BitAnd<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x & rhs)
    }
    fn bitxor_all<Rhs>(self, rhs: Rhs) -> [<T as BitXor<Rhs>>::Output; N]
    where
        T: BitXor<Rhs>,
        Rhs: Copy
    {
        self.map(|x| x ^ rhs)
    }

    fn add_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: AddAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] += rhs;
            i += 1;
        }
    }
    fn sub_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: SubAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] -= rhs;
            i += 1;
        }
    }
    fn mul_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: MulAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] *= rhs;
            i += 1;
        }
    }
    fn div_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: DivAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] /= rhs;
            i += 1;
        }
    }
    fn rem_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: RemAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] %= rhs;
            i += 1;
        }
    }
    fn shl_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: ShlAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] <<= rhs;
            i += 1;
        }
    }
    fn shr_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: ShrAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] >>= rhs;
            i += 1;
        }
    }
    fn bitor_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: BitOrAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] |= rhs;
            i += 1;
        }
    }
    fn bitand_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: BitAndAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] &= rhs;
            i += 1;
        }
    }
    fn bitxor_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: BitXorAssign<Rhs>,
        Rhs: Copy
    {
        let mut i = 0;
        while i < N
        {
            self[i] ^= rhs;
            i += 1;
        }
    }
    
    fn add_all_neg<Rhs>(self, rhs: Rhs) -> [<Rhs as Sub<T>>::Output; N]
    where
        Rhs: Copy + Sub<T>
    {
        self.map(|x| rhs - x)
    }
    fn mul_all_inv<Rhs>(self, rhs: Rhs) -> [<Rhs as Div<T>>::Output; N]
    where
        Rhs: Copy + Div<T>
    {
        self.map(|x| rhs / x)
    }
    
    fn neg_all(self) -> [<T as Neg>::Output; N]
    where
        T: Neg
    {
        self.map(|x| -x)
    }
    fn neg_assign_all(&mut self)
    where
        T: Neg<Output = T>
    {
        let mut i = 0;
        while i < N
        {
            unsafe {
                let ptr = self.as_mut_ptr().add(i);
                let x = ptr.read();
                ptr.write(-x)
            }
            i += 1;
        }
    }
    
    fn add_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Add<Rhs>>::Output; N]
    where
        T: Add<Rhs>
    {
        self.comap(rhs, Add::add)
    }
    fn sub_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Sub<Rhs>>::Output; N]
    where
        T: Sub<Rhs>
    {
        self.comap(rhs, Sub::sub)
    }
    fn mul_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Mul<Rhs>>::Output; N]
    where
        T: Mul<Rhs>
    {
        self.comap(rhs, Mul::mul)
    }
    fn div_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Div<Rhs>>::Output; N]
    where
        T: Div<Rhs>
    {
        self.comap(rhs, Div::div)
    }
    fn rem_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Rem<Rhs>>::Output; N]
    where
        T: Rem<Rhs>
    {
        self.comap(rhs, Rem::rem)
    }
    fn shl_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Shl<Rhs>>::Output; N]
    where
        T: Shl<Rhs>
    {
        self.comap(rhs, Shl::shl)
    }
    fn shr_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Shr<Rhs>>::Output; N]
    where
        T: Shr<Rhs>
    {
        self.comap(rhs, Shr::shr)
    }
    fn bitor_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as BitOr<Rhs>>::Output; N]
    where
        T: BitOr<Rhs>
    {
        self.comap(rhs, BitOr::bitor)
    }
    fn bitand_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as BitAnd<Rhs>>::Output; N]
    where
        T: BitAnd<Rhs>
    {
        self.comap(rhs, BitAnd::bitand)
    }
    fn bitxor_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as BitXor<Rhs>>::Output; N]
    where
        T: BitXor<Rhs>
    {
        self.comap(rhs, BitXor::bitxor)
    }

    fn add_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: AddAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] += unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }
    fn sub_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: SubAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] -= unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }
    fn mul_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: MulAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] *= unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }
    fn div_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: DivAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] /= unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }
    fn rem_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: RemAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] %= unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }
    fn shl_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: ShlAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] <<= unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }
    fn shr_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: ShrAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] >>= unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }
    fn bitor_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: BitOrAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] |= unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }
    fn bitand_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: BitAndAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] &= unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }
    fn bitxor_assign_each<Rhs>(&mut self, rhs: [Rhs; N])
    where
        T: BitXorAssign<Rhs>
    {
        let mut i = 0;
        while i < N
        {
            self[i] ^= unsafe {rhs.as_ptr().add(i).read()};
            i += 1;
        }
        core::mem::forget(rhs);
    }

    fn try_mul_dot<Rhs>(self, rhs: [Rhs; N]) -> Option<<T as Mul<Rhs>>::Output>
    where
        T: Mul<Rhs, Output: AddAssign>
    {
        if N == 0
        {
            return None
        }
        
        let ptr1 = self.as_ptr();
        let ptr2 = rhs.as_ptr();
    
        unsafe {
            let mut sum = ptr1.read()*ptr2.read();
            let mut i = 1;
            while i < N
            {
                sum += ptr1.add(i).read()*ptr2.add(i).read();
                i += 1;
            }
            core::mem::forget(self);
            core::mem::forget(rhs);
            Some(sum)
        }
    }
    
    fn mul_dot_bias<Rhs>(self, rhs: [Rhs; N], bias: <T as Mul<Rhs>>::Output) -> <T as Mul<Rhs>>::Output
    where
        T: Mul<Rhs, Output: AddAssign>
    {
        let ptr1 = self.as_ptr();
        let ptr2 = rhs.as_ptr();
    
        let mut sum = bias;
        let mut i = 0;
        while i < N
        {
            sum += unsafe {
                ptr1.add(i).read()*ptr2.add(i).read()
            };
            i += 1;
        }
        core::mem::forget(self);
        core::mem::forget(rhs);
        sum
    }

    fn mul_outer<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [[<T as Mul<Rhs>>::Output; M]; N]
    where
        T: Mul<Rhs> + Copy,
        Rhs: Copy
    {
        self.comap_outer(rhs, Mul::mul)
    }
    
    fn mul_cross<Rhs>(&self, rhs: [&[Rhs; N]; N - 2]) -> [<T as Sub>::Output; N]
    where
        T: MulAssign<Rhs> + Sub + Copy,
        Rhs: Copy
    {
        ArrayOps::fill(|i| {
            let mut m_p = self[(i + 1) % N];
            let mut m_m = self[(i + (N - 1)) % N];
    
            let mut n = 2;
            while n < N
            {
                m_p *= rhs[n - 2][(i + n) % N];
                m_m *= rhs[n - 2][(i + (N - n)) % N];
                
                n += 1;
            }
    
            m_p - m_m
        })
    }
    
    fn try_magnitude_squared(self) -> Option<<T as Mul<T>>::Output>
    where
        T: Mul<T, Output: AddAssign> + Copy
    {
        self.try_mul_dot(self)
    }
    
    fn chain<const M: usize>(self, rhs: [T; M]) -> [T; N + M]
    {
        crate::chain(self, rhs)
    }
    
    fn rchain<const M: usize>(self, rhs: [T; M]) -> [T; N + M]
    {
        crate::rchain(self, rhs)
    }
    
    fn spread<const M: usize>(self) -> ([[T; N / M]; M], [T; N % M])
    where
        [(); M - 1]:,
        [(); N % M]:,
        [(); N / M]:
    {
        crate::spread(self)
    }
    fn spread_ref<const M: usize>(&self) -> ([&[Padded<T, M>; N / M]; M], &[T; N % M])
    where
        [(); M - 1]:,
        [(); N % M]:
    {
        let (left, right) = crate::rsplit_ptr(self, N % M);
    
        unsafe {(
            ArrayOps::fill(|i| &*left.add(i).cast()),
            &*right.cast()
        )}
    }
    fn spread_mut<const M: usize>(&mut self) -> ([&mut [Padded<T, M>; N / M]; M], &mut [T; N % M])
    where
        [(); M - 1]:,
        [(); N % M]:
    {
        let (left, right) = crate::rsplit_mut_ptr(self, N % M);
    
        unsafe {(
            ArrayOps::fill(|i| &mut *left.add(i).cast()),
            &mut *right.cast()
        )}
    }
    
    fn rspread<const M: usize>(self) -> ([T; N % M], [[T; N / M]; M])
    where
        [(); M - 1]:,
        [(); N % M]:,
        [(); N / M]:
    {
        crate::rspread(self)
    }
    fn rspread_ref<const M: usize>(&self) -> (&[T; N % M], [&[Padded<T, M>; N / M]; M])
    where
        [(); M - 1]:,
        [(); N % M]:
    {
        let (left, right) = crate::split_ptr(self, N % M);
    
        unsafe {(
            &*left.cast(),
            ArrayOps::fill(|i| &*right.add(i).cast())
        )}
    }
    fn rspread_mut<const M: usize>(&mut self) -> (&mut [T; N % M], [&mut [Padded<T, M>; N / M]; M])
    where
        [(); M - 1]:,
        [(); N % M]:
    {
        let (left, right) = crate::split_mut_ptr(self, N % M);
    
        unsafe {(
            &mut *left.cast(),
            ArrayOps::fill(|i| &mut *right.add(i).cast())
        )}
    }
    fn spread_exact<const M: usize>(self) -> [[T; N / M]; M]
    where
        [(); M - 1]:,
        [(); 0 - N % M]:,
        [(); N / M]:
    {
        crate::spread_exact(self)
    }
    fn spread_exact_ref<const M: usize>(&self) -> [&[Padded<T, M>; N / M]; M]
    where
        [(); M - 1]:,
        [(); 0 - N % M]:
    {
        let ptr = self as *const T;
        
        ArrayOps::fill(|i| unsafe {&*ptr.add(i).cast()})
    }
    fn spread_exact_mut<const M: usize>(&mut self) -> [&mut [Padded<T, M>; N / M]; M]
    where
        [(); M - 1]:,
        [(); 0 - N % M]:
    {
        let ptr = self as *mut T;
        
        ArrayOps::fill(|i| unsafe {&mut *ptr.add(i).cast()})
    }
    
    fn chunks<const M: usize>(self) -> ([[T; M]; N / M], [T; N % M])
    {
        crate::chunks(self)
    }
    fn chunks_ref<const M: usize>(&self) -> (&[[T; M]; N / M], &[T; N % M])
    {
        crate::chunks_ref(self)
    }
    fn chunks_mut<const M: usize>(&mut self) -> (&mut [[T; M]; N / M], &mut [T; N % M])
    {
        crate::chunks_mut(self)
    }

    fn array_rchunks<const M: usize>(self) -> ([T; N % M], [[T; M]; N / M])
    {
        crate::array_rchunks(self)
    }
    fn array_rchunks_ref<const M: usize>(&self) -> (&[T; N % M], &[[T; M]; N / M])
    {
        crate::array_rchunks_ref(self)
    }
    fn array_rchunks_mut<const M: usize>(&mut self) -> (&mut [T; N % M], &mut [[T; M]; N / M])
    {
        crate::array_rchunks_mut(self)
    }
    
    fn chunks_exact<const M: usize>(self) -> [[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:
    {
        crate::chunks_exact(self)
    }
    fn chunks_exact_ref<const M: usize>(&self) -> &[[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:
    {
        crate::chunks_exact_ref(self)
    }
    fn chunks_exact_mut<const M: usize>(&mut self) -> &mut [[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:
    {
        crate::chunks_exact_mut(self)
    }
    
    fn array_simd<const M: usize>(self) -> ([Simd<T, M>; N / M], [T; N % M])
    where
        T: SimdElement,
        LaneCount<M>: SupportedLaneCount,
        [(); N % M]:,
        [(); N / M]:
    {
        crate::array_simd(self)
    }
    
    fn array_rsimd<const M: usize>(self) -> ([T; N % M], [Simd<T, M>; N / M])
    where
        T: SimdElement,
        LaneCount<M>: SupportedLaneCount,
        [(); N % M]:,
        [(); N / M]:
    {
        crate::array_rsimd(self)
    }
    
    fn array_simd_exact<const M: usize>(self) -> [Simd<T, M>; N / M]
    where
        T: SimdElement,
        LaneCount<M>: SupportedLaneCount,
        [(); 0 - N % M]:,
        [(); N / M]:
    {
        crate::array_simd_exact(self)
    }
    
    fn split_array<const M: usize>(self) -> ([T; M], [T; N - M])
    where
        [(); N - M]:
    {
        crate::split_array(self)
    }
    fn split_array_ref2<const M: usize>(&self) -> (&[T; M], &[T; N - M])
    where
        [(); N - M]:
    {
        crate::split_array_ref(self)
    }
    fn split_array_mut2<const M: usize>(&mut self) -> (&mut [T; M], &mut [T; N - M])
    where
        [(); N - M]:
    {
        crate::split_array_mut(self)
    }
    
    fn rsplit_array<const M: usize>(self) -> ([T; N - M], [T; M])
    where
        [(); N - M]:
    {
        crate::rsplit_array(self)
    }
    fn rsplit_array_mut2<const M: usize>(&mut self) -> (&mut [T; N - M], &mut [T; M])
    where
        [(); N - M]:
    {
        crate::rsplit_array_mut(self)
    }
    fn rsplit_array_ref2<const M: usize>(&self) -> (&[T; N - M], &[T; M])
    where
        [(); N - M]:
    {
        crate::rsplit_array_ref(self)
    }

    fn each_ref2(&self) -> [&T; N]
    {
        let ptr = self as *const T;
        ArrayOps::fill(|i| {
            let y = unsafe {&*ptr.add(i)};
            y
        })
    }
    fn each_mut2(&mut self) -> [&mut T; N]
    {
        let ptr = self as *mut T;
        ArrayOps::fill(|i| {
            let y = unsafe {&mut *ptr.add(i)};
            y
        })
    }
    
    fn bit_reverse_permutation(&mut self)
    where
        [(); N.is_power_of_two() as usize - 1]:
    {
        crate::bit_reverse_permutation(self)
    }
}