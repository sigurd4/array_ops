use core::{ops::{Sub, AddAssign, Deref, DerefMut, Mul, Div, Add, Neg, MulAssign}, mem::{ManuallyDrop, MaybeUninit}, borrow::{Borrow, BorrowMut}, marker::Destruct};

use array_trait::Array;

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
        F: /*~const*/ FnMut(usize) -> T + ~const Destruct;
    fn rfill<F>(fill: F) -> Self
    where
        F: /*~const*/ FnMut(usize) -> T + ~const Destruct;

    /*fn for_each<F>(self, action: F) -> ()
    where
        F: /*~const*/ FnMut(T) -> () + ~const Destruct;
    fn for_each_ref<F>(&self, action: F) -> ()
    where
        F: /*~const*/ FnMut(&T) -> () + ~const Destruct;
    fn for_each_mut<F>(&mut self, action: F) -> ()
    where
        F: /*~const*/ FnMut(&mut T) -> () + ~const Destruct;*/
    
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
        F: /*~const*/ FnMut(usize) -> T + ~const Destruct,
        T: ~const Destruct;
    fn rresize<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: /*~const*/ FnMut(usize) -> T + ~const Destruct,
        T: ~const Destruct;

    fn extend<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: /*~const*/ FnMut(usize) -> T + ~const Destruct,
        [(); M - N]:;
    fn rextend<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: /*~const*/ FnMut(usize) -> T + ~const Destruct,
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

    /*
    /// Converts an array into a const interator.
    /// 
    /// The const iterator does not implement [std::iter::Iterator](Iterator), and as such is more limited in its usage.
    /// However it can be used at compile-time.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(inline_const)]
    /// #![feature(const_trait_impl)]
    /// #![feature(const_mut_refs)]
    /// #![feature(const_deref)]
    /// 
    /// use core::{mem::ManuallyDrop, ops::DerefMut};
    /// use array_trait::*;
    /// 
    /// const A: [u8; 3] = [1, 2, 3];
    /// 
    /// const A_SUM: u8 = const {
    ///     let mut iter = ManuallyDrop::new(A.into_const_iter());
    ///     let mut sum = 0;
    /// 
    ///     while let Some(b) = iter.deref_mut().next()
    ///     {
    ///         sum += b;
    ///     }
    /// 
    ///     sum
    /// };
    /// 
    /// assert_eq!(A_SUM, 1 + 2 + 3);
    /// ```
    fn into_const_iter(self) -> IntoConstIter<T, N, true>;
    fn into_const_iter_reverse(self) -> IntoConstIter<T, N, false>;

    /// Makes a const iterator over the array-slice.
    /// 
    /// The const iterator does not implement [std::iter::Iterator](Iterator), and as such is more limited in its usage.
    /// However it can be used at compile-time.
    fn const_iter(&self) -> ConstIter<'_, T, N>;
    /// Makes a mutable const iterator over the mutable array-slice.
    /// 
    /// The const iterator does not implement [std::iter::Iterator](Iterator), and as such is more limited in its usage.
    /// However it can be used at compile-time.
    fn const_iter_mut(&mut self) -> ConstIterMut<'_, T, N>;*/

    /// Maps all values of an array with a given function.
    /// 
    /// This method can be executed at compile-time, as opposed to the standard-library method.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(const_closures)]
    /// #![feature(const_mut_refs)]
    /// #![feature(const_trait_impl)]
    /// 
    /// use array_trait::*;
    /// 
    /// const A: [u8; 4] = [1, 2, 3, 4];
    /// const B: [i8; 4] = A.map2(const |b| -(b as i8));
    /// 
    /// assert_eq!(B, [-1, -2, -3, -4]);
    /// ```
    fn map2<Map>(self, map: Map) -> [Map::Output; N]
    where
        Map: /*~const*/ FnMut<(T,)> + ~const Destruct;
    fn map_outer<Map>(&self, map: Map) -> [[Map::Output; N]; N]
    where
        Map: /*~const*/ FnMut<(T, T)> + ~const Destruct,
        T: Copy;
    fn comap<Map, Rhs>(self, rhs: [Rhs; N], map: Map) -> [Map::Output; N]
    where
        Map: /*~const*/ FnMut<(T, Rhs)> + ~const Destruct;
    fn comap_outer<Map, Rhs, const M: usize>(&self, rhs: &[Rhs; M], map: Map) -> [[Map::Output; M]; N]
    where
        Map: /*~const*/ FnMut<(T, Rhs)> + ~const Destruct,
        T: Copy,
        Rhs: Copy;

    /// Combines two arrays with possibly different items into parallel, where each element lines up in the same position.
    /// 
    /// This method can be executed at compile-time, as opposed to the standard-library method.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(const_trait_impl)]
    /// 
    /// use array_trait::*;
    /// 
    /// const A: [u8; 4] = [4, 3, 2, 1];
    /// const B: [&str; 4] = ["four", "three", "two", "one"];
    /// const C: [(u8, &str); 4] = A.zip2(B);
    /// 
    /// assert_eq!(C, [(4, "four"), (3, "three"), (2, "two"), (1, "one")]);
    /// ```
    fn zip<Z>(self, other: [Z; N]) -> [(T, Z); N];
    fn zip_outer<Z, const M: usize>(&self, other: &[Z; M]) -> [[(T, Z); M]; N]
    where
        T: Copy,
        Z: Copy;

    fn enumerate(self) -> [(usize, T); N];

    fn diagonal<const H: usize, const W: usize>(self) -> [[T; W]; H]
    where
        T: /*~const*/ Default + Copy,
        [(); H - N]:,
        [(); W - N]:;
    
    /// Differentiates array (discrete calculus)
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array_trait::*;
    /// 
    /// let a = [1, 2, 3];
    /// 
    /// assert_eq!(a.differentiate(), [2 - 1, 3 - 2]);
    /// ```
    fn differentiate(self) -> [<T as Sub<T>>::Output; N.saturating_sub(1)]
    where
        [(); N.saturating_sub(1)]:,
        T: Sub<T> + Copy;
    
    /// Integrates array (discrete calculus)
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array_trait::*;
    /// 
    /// let a = [1, 2, 3];
    /// 
    /// assert_eq!(a.integrate(), [1, 1 + 2, 1 + 2 + 3])
    /// ```
    fn integrate(self) -> Self
    where
        T: /*~const*/ AddAssign<T> + Copy;

    fn integrate_from<const M: usize>(self, x0: T) -> [T; M]
    where
        T: /*~const*/ AddAssign<T> + Copy;

    /// Reduces elements in array into one element, using a given operand
    /// 
    /// # Example
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array_trait::ArrayOps;
    /// 
    /// const A: [u8; 3] = [1, 2, 3];
    /// 
    /// let r: u8 = A.reduce(|a, b| a + b).unwrap();
    /// 
    /// assert_eq!(r, 6);
    /// ```
    fn reduce<R>(self, reduce: R) -> Option<T>
    where
        R: /*~const*/ FnMut(T, T) -> T + ~const Destruct;

    fn try_sum(self) -> Option<T>
    where
        T: /*~const*/ AddAssign;
        
    fn sum_from<S>(self, from: S) -> S
    where
        S: /*~const*/ AddAssign<T>;
        
    fn try_product(self) -> Option<T>
    where
        T: /*~const*/ MulAssign;
        
    fn product_from<P>(self, from: P) -> P
    where
        P: /*~const*/ MulAssign<T>;

    fn max(self) -> Option<T>
    where
        T: /*~const*/ Ord;
        
    fn min(self) -> Option<T>
    where
        T: /*~const*/ Ord;
        
    fn first_max(self) -> Option<T>
    where
        T: /*~const*/ PartialOrd<T>;
        
    fn first_min(self) -> Option<T>
    where
        T: /*~const*/ PartialOrd<T>;
        
    fn argmax(&self) -> Option<usize>
    where
        T: /*~const*/ PartialOrd<T>;
        
    fn argmin(&self) -> Option<usize>
    where
        T: /*~const*/ PartialOrd<T>;

    fn add_all<Rhs>(self, rhs: Rhs) -> [<T as Add<Rhs>>::Output; N]
    where
        T: /*~const*/ Add<Rhs>,
        Rhs: Copy;
    fn sub_all<Rhs>(self, rhs: Rhs) -> [<T as Sub<Rhs>>::Output; N]
    where
        T: /*~const*/ Sub<Rhs>,
        Rhs: Copy;
    fn mul_all<Rhs>(self, rhs: Rhs) -> [<T as Mul<Rhs>>::Output; N]
    where
        T: /*~const*/ Mul<Rhs>,
        Rhs: Copy;
    fn div_all<Rhs>(self, rhs: Rhs) -> [<T as Div<Rhs>>::Output; N]
    where
        T: /*~const*/ Div<Rhs>,
        Rhs: Copy;
        
    fn add_all_neg<Rhs>(self, rhs: Rhs) -> [<Rhs as Sub<T>>::Output; N]
    where
        Rhs: Copy + /*~const*/ Sub<T>;
    fn mul_all_inv<Rhs>(self, rhs: Rhs) -> [<Rhs as Div<T>>::Output; N]
    where
        Rhs: Copy + /*~const*/ Div<T>;
    
    fn neg_all(self) -> [<T as Neg>::Output; N]
    where
        T: /*~const*/ Neg;
    
    fn add_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Add<Rhs>>::Output; N]
    where
        T: /*~const*/ Add<Rhs>;
    fn sub_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Sub<Rhs>>::Output; N]
    where
        T: /*~const*/ Sub<Rhs>;
    fn mul_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Mul<Rhs>>::Output; N]
    where
        T: /*~const*/ Mul<Rhs>;
    fn div_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Div<Rhs>>::Output; N]
    where
        T: /*~const*/ Div<Rhs>;

    fn try_mul_dot<Rhs>(self, rhs: [Rhs; N]) -> Option<<T as Mul<Rhs>>::Output>
    where
        T: /*~const*/ Mul<Rhs, Output: /*~const*/ AddAssign>;

    fn mul_dot_bias<Rhs>(self, rhs: [Rhs; N], bias: <T as Mul<Rhs>>::Output) -> <T as Mul<Rhs>>::Output
    where
        T: /*~const*/ Mul<Rhs, Output: /*~const*/ AddAssign>;

    fn mul_outer<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [[<T as Mul<Rhs>>::Output; M]; N]
    where
        T: /*~const*/ Mul<Rhs> + Copy,
        Rhs: Copy;
        
    /// Computes the general cross-product of the two arrays (as if vectors, in the mathematical sense).
    /// 
    /// # Example
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// #![feature(const_trait_impl)]
    /// 
    /// use array_trait::ArrayOps;
    /// 
    /// const U: [f64; 3] = [1.0, 0.0, 0.0];
    /// const V: [f64; 3] = [0.0, 1.0, 0.0];
    /// 
    /// const W: [f64; 3] = U.mul_cross([&V]);
    /// 
    /// assert_eq!(W, [0.0, 0.0, 1.0]);
    /// ```
    fn mul_cross<Rhs>(&self, rhs: [&[Rhs; N]; N - 2]) -> [<T as Sub>::Output; N]
    where
        T: /*~const*/ MulAssign<Rhs> + /*~const*/ Sub + Copy,
        Rhs: Copy;

    fn try_magnitude_squared(self) -> Option<<T as Mul<T>>::Output>
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign> + Copy;

    /// Chains two arrays with the same item together.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use array_trait::*;
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
    /// use array_trait::*;
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
    /// use array_trait::*;
    /// 
    /// let array = ["ping 1", "pong 1", "ping 2", "pong 2", "ping 3", "pong 3", "uhh..."];
    /// 
    /// let ([ping, pong], rest) = array.array_spread::<2>();
    /// 
    /// assert_eq!(ping, ["ping 1", "ping 2", "ping 3"]);
    /// assert_eq!(pong, ["pong 1", "pong 2", "pong 3"]);
    /// assert_eq!(rest, ["uhh..."]);
    /// ```
    fn array_spread<const M: usize>(self) -> ([[T; N / M]; M], [T; N % M])
    where
        [(); M - 1]:,
        [(); N / M]:,
        [(); N % M]:;

    /// Distributes items of an array-slice equally across a given width, then provides the rest as a separate array-slice.
    /// 
    /// The spread-out slices are given in padded arrays. Each padded item can be borrowed into a reference to the array's item.
    fn array_spread_ref<const M: usize>(&self) -> ([&[Padded<T, M>; N / M]; M], &[T; N % M])
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
    /// use array_trait::*;
    /// 
    /// let mut array = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"];
    /// 
    /// let (threes, _) = array.array_spread_mut::<3>();
    /// 
    /// for fizz in threes.into_iter().last().unwrap()
    /// {
    ///     **fizz = "fizz";
    /// }
    /// 
    /// let (fives, _) = array.array_spread_mut::<5>();
    /// 
    /// for buzz in fives.into_iter().last().unwrap()
    /// {
    ///     **buzz = "buzz";
    /// }
    /// 
    /// let (fifteens, _) = array.array_spread_mut::<15>();
    /// 
    /// for fizzbuzz in fifteens.into_iter().last().unwrap()
    /// {
    ///     **fizzbuzz = "fizzbuzz";
    /// }
    /// 
    /// assert_eq!(array, ["1", "2", "fizz", "4", "buzz", "fizz", "7", "8", "fizz", "buzz", "11", "fizz", "13", "14", "fizzbuzz", "16", "17", "fizz", "19", "buzz"]);
    /// 
    /// ```
    fn array_spread_mut<const M: usize>(&mut self) -> ([&mut [Padded<T, M>; N / M]; M], &mut [T; N % M])
    where
        [(); M - 1]:,
        [(); N % M]:;
    
    /// Distributes items of an array equally across a given width, then provides the leftmost rest as a separate array.
    fn array_rspread<const M: usize>(self) -> ([T; N % M], [[T; N / M]; M])
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
    /// use array_trait::*;
    /// 
    /// let array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    /// 
    /// let (zero, [odd, even]) = array.array_rspread_ref::<2>();
    /// 
    /// assert_eq!(*zero, [0]);
    /// assert_eq!(odd.each_ref().map(|padding| **padding), [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    /// assert_eq!(even.each_ref().map(|padding| **padding), [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    /// ```
    fn array_rspread_ref<const M: usize>(&self) -> (&[T; N % M], [&[Padded<T, M>; N / M]; M])
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
    /// use array_trait::*;
    /// 
    /// let mut array = ["the", "beat", "goes", "1", "2", "3", "4", "5", "6", "7", "8"];
    /// 
    /// let (start, [boots, n, cats, and]) = array.array_rspread_mut::<4>();
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
    fn array_rspread_mut<const M: usize>(&mut self) -> (&mut [T; N % M], [&mut [Padded<T, M>; N / M]; M])
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
    /// use array_trait::*;
    /// 
    /// let array = *b"aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ";
    /// 
    /// let [lower_case, upper_case] = array.array_spread_exact::<2>();
    /// 
    /// assert_eq!(lower_case, *b"abcdefghijklmnopqrstuvwxyz");
    /// assert_eq!(upper_case, *b"ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    /// ```
    fn array_spread_exact<const M: usize>(self) -> [[T; N / M]; M]
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
    /// use array_trait::*;
    /// 
    /// let statement = ["s", "he", "be", "lie", "ve", "d"];
    /// 
    /// let [interpretation2, interpretation1] = statement.array_spread_exact_ref::<2>();
    /// 
    /// assert_eq!(interpretation1.each_ref().map(|padding| &**padding), ["he", "lie", "d"].each_ref());
    /// assert_eq!(interpretation2.each_ref().map(|padding| &**padding), ["s", "be", "ve"].each_ref());
    /// ```
    fn array_spread_exact_ref<const M: usize>(&self) -> [&[Padded<T, M>; N / M]; M]
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
    /// use array_trait::*;
    /// 
    /// let mut array = *b"aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ";
    /// 
    /// let [lower_case, upper_case] = array.array_spread_exact_mut::<2>();
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
    fn array_spread_exact_mut<const M: usize>(&mut self) -> [&mut [Padded<T, M>; N / M]; M]
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
    /// use array_trait::*;
    /// 
    /// let array = ["carrot", "potato", "beet", "tomato", "kiwi", "banana", "cherry", "peach", "strawberry", "nine volt batteries"];
    /// 
    /// let ([root_vegetables, technically_berries, stone_fruits], not_for_human_consumption) = array.array_chunks::<3>();
    /// 
    /// assert_eq!(root_vegetables, ["carrot", "potato", "beet"]);
    /// assert_eq!(technically_berries, ["tomato", "kiwi", "banana"]);
    /// assert_eq!(stone_fruits, ["cherry", "peach", "strawberry"]);
    /// assert_eq!(not_for_human_consumption, ["nine volt batteries"]);
    /// ```
    fn array_chunks<const M: usize>(self) -> ([[T; M]; N / M], [T; N % M])
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
    /// use array_trait::*;
    /// 
    /// let transistors = ["2N3904", "2N2222A", "BC107", "AC127", "OC7", "NKT275", "2SK30A", "2N5458", "J108", "2N7000", "BS170"];
    /// 
    /// let ([silicon_bjts, germanium_bjts, jfets], mosfets) = transistors.array_chunks_ref::<3>();
    /// 
    /// assert_eq!(silicon_bjts, &["2N3904", "2N2222A", "BC107"]);
    /// assert_eq!(germanium_bjts, &["AC127", "OC7", "NKT275"]);
    /// assert_eq!(jfets, &["2SK30A", "2N5458", "J108"]);
    /// assert_eq!(mosfets, &["2N7000", "BS170"]);
    /// ```
    fn array_chunks_ref<const M: usize>(&self) -> (&[[T; M]; N / M], &[T; N % M])
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
    /// use array_trait::*;
    /// 
    /// let mut array = [0, 1, 0, 1, 0, 1, 6];
    /// 
    /// let (pairs, last) = array.array_chunks_mut::<2>();
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
    fn array_chunks_mut<const M: usize>(&mut self) -> (&mut [[T; M]; N / M], &mut [T; N % M])
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
    /// use array_trait::*;
    /// 
    /// let array = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    /// 
    /// let [lower_half, upper_half] = array.array_chunks_exact::<5>();
    /// 
    /// assert_eq!(lower_half, [0.0, 0.1, 0.2, 0.3, 0.4]);
    /// assert_eq!(upper_half, [0.5, 0.6, 0.7, 0.8, 0.9]);
    /// ```
    fn array_chunks_exact<const M: usize>(self) -> [[T; M]; N / M]
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
    /// use array_trait::*;
    /// 
    /// let array = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    /// 
    /// let [lower_half, upper_half] = array.array_chunks_exact_ref::<5>();
    /// 
    /// assert_eq!(lower_half, &[0.0, 0.1, 0.2, 0.3, 0.4]);
    /// assert_eq!(upper_half, &[0.5, 0.6, 0.7, 0.8, 0.9]);
    /// ```
    fn array_chunks_exact_ref<const M: usize>(&self) -> &[[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:;
    /// Divides a mutable array-slice into chunks, with no rest.
    /// 
    /// The chunk length must be a factor of the array length, otherwise it will not compile.
    fn array_chunks_exact_mut<const M: usize>(&mut self) -> &mut [[T; M]; N / M]
    where
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

pub /*const*/ fn fill<const N: usize, F>(mut fill: F) -> [<F as FnOnce<(usize,)>>::Output; N]
where
    F: FnMut<(usize,)> + /*~const*/ Destruct
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

pub /*const*/ fn rfill<const N: usize, F>(mut fill: F) -> [<F as FnOnce<(usize,)>>::Output; N]
where
    F: FnMut<(usize,)> + /*~const*/ Destruct
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

pub /*const*/ fn truncate<T, const N: usize, const M: usize>(array: [T; N]) -> [T; M]
where
    T: /*~const*/ Destruct,
    [(); N - M]:
{
    crate::split_array(array).0
}
pub /*const*/ fn rtruncate<T, const N: usize, const M: usize>(array: [T; N]) -> [T; M]
where
    T: /*~const*/ Destruct,
    [(); N - M]:
{
    crate::rsplit_array(array).1
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

pub /*const*/ fn resize<T, const N: usize, const M: usize, F>(array: [T; N], mut fill: F) -> [T; M]
where
    F: FnMut<(usize,), Output = T> + /*~const*/ Destruct,
    T: /*~const*/ Destruct
{
    let mut i = N.min(M);
    while i < N
    {
        let _ = unsafe {(&array[i] as *const T).read()};
        i += 1;
    }

    let mut dst = unsafe {private::uninit()};
    let mut ptr = &mut dst as *mut T;

    unsafe {core::ptr::copy_nonoverlapping(core::mem::transmute(&array), ptr, N.min(M))};
    core::mem::forget(array);

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

// NEEDS TEST
pub /*const*/ fn rresize<T, const N: usize, const M: usize, F>(array: [T; N], mut fill: F) -> [T; M]
where
    F: FnMut<(usize,), Output = T> + /*~const*/ Destruct,
    T: /*~const*/ Destruct
{
    let mut i = 0;
    while i < N.saturating_sub(M)
    {
        let _ = unsafe {(&array[i] as *const T).read()};
        i += 1;
    }
    
    let mut dst = unsafe {private::uninit()};
    let mut ptr = unsafe {(&mut dst as *mut T).add(M.saturating_sub(N))};
    
    unsafe {core::ptr::copy_nonoverlapping((&array as *const T).add(N.saturating_sub(M)), ptr, N.min(M))};
    core::mem::forget(array);

    let mut i = M.saturating_sub(N);
    while i > 0
    {
        i -= 1;
        ptr = unsafe {ptr.sub(1)};
        unsafe {core::ptr::write(ptr, fill(i))};
    }

    dst
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

pub /*const*/ fn extend<T, const N: usize, const M: usize, F>(array: [T; N], mut fill: F) -> [T; M]
where
    F: FnMut(usize) -> T + /*~const*/ Destruct,
    [(); M - N]:
{
    let filled: [T; M - N] = crate::fill(/*const*/ |i| fill(i + N));
    unsafe {private::merge_transmute(array, filled)}
}
pub /*const*/ fn rextend<T, const N: usize, const M: usize, F>(array: [T; N], mut fill: F) -> [T; M]
where
    F: FnMut(usize) -> T + /*~const*/ Destruct,
    [(); M - N]:
{
    let filled: [T; M - N] = crate::rfill(&mut fill);
    unsafe {private::merge_transmute(filled, array)}
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

pub /*const*/ fn map<T, const N: usize, Map>(array: [T; N], mut map: Map) -> [<Map as FnOnce<(T,)>>::Output; N]
where
    Map: FnMut<(T,)> + /*~const*/ Destruct
{
    let ptr = &array as *const T;

    let dst = crate::fill(/*const*/ |i| unsafe {
        map(ptr.add(i).read())
    });

    core::mem::forget(array);

    dst
}

pub /*const*/ fn map_outer<T, const N: usize, Map>(array: &[T; N], map: Map) -> [[Map::Output; N]; N]
where
    Map: FnMut<(T, T)> + /*~const*/ Destruct,
    T: Copy
{
    crate::comap_outer(array, array, map)
}

pub /*const*/ fn comap<T, const N: usize, Map, Rhs>(array: [T; N], rhs: [Rhs; N], mut map: Map) -> [Map::Output; N]
where
    Map: FnMut<(T, Rhs)> + /*~const*/ Destruct
{
    let ptr0 = &array as *const T;
    let ptr1 = &rhs as *const Rhs;

    let dst = crate::fill(/*const*/ |i| unsafe {
        map(
            ptr0.add(i).read(),
            ptr1.add(i).read()
        )
    });

    core::mem::forget(array);
    core::mem::forget(rhs);

    dst
}
pub /*const*/ fn comap_outer<T, const N: usize, Map, Rhs, const M: usize>(array: &[T; N], rhs: &[Rhs; M], mut map: Map) -> [[Map::Output; M]; N]
where
    Map: FnMut<(T, Rhs)> + /*~const*/ Destruct,
    T: Copy,
    Rhs: Copy
{
    crate::map(*array, /*const*/ |x| crate::map(*rhs, /*const*/ |y| map(x, y)))
}

pub /*const*/ fn zip<T, const N: usize, Z>(array: [T; N], other: [Z; N]) -> [(T, Z); N]
{
    crate::comap(array, other, const |x, y| (x, y))
}
pub /*const*/ fn zip_outer<T, const N: usize, Z, const M: usize>(array: &[T; N], other: &[Z; M]) -> [[(T, Z); M]; N]
where
    T: Copy,
    Z: Copy
{
    crate::comap_outer(array, other, const |x, y| (x, y))
}

pub /*const*/ fn enumerate<T, const N: usize>(array: [T; N]) -> [(usize, T); N]
{
    let ptr = &array as *const T;

    let dst = crate::fill(/*const*/ |i| unsafe {
        (i, ptr.add(i).read())
    });

    core::mem::forget(array);

    dst
}

pub /*const*/ fn diagonal<T, const N: usize, const H: usize, const W: usize>(array: [T; N]) -> [[T; W]; H]
where
    T: Default,
    [(); H - N]:,
    [(); W - N]:
{
    let ptr = array.as_ptr();
    
    let dst = crate::fill(/*const*/ |i| crate::fill(/*const*/ |j| if i == j && i < N
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

    core::mem::forget(array);

    dst
}

pub /*const*/ fn differentiate<T, const N: usize>(array: [T; N]) -> [<T as Sub<T>>::Output; N.saturating_sub(1)]
where
    [(); N.saturating_sub(1)]:,
    T: Sub<T> + Copy + /*~const*/ Destruct
{
    if let Some(&(mut x_prev)) = array.first()
    {
        crate::fill(/*const*/ |i| {
            let x = array[i + 1];
            let dx = x - x_prev;
            x_prev = x;
            dx
        })
    }
    else
    {
        // Return empty array
        unsafe {MaybeUninit::assume_init(MaybeUninit::uninit())}
    }
}

pub /*const*/ fn integrate<T, const N: usize>(array: [T; N]) -> [T; N]
where
    T: AddAssign<T> + Copy
{
    if let Some(&(mut x_accum)) = array.first()
    {
        crate::fill(/*const*/ |i| {
            let xi = x_accum;
            if i + 1 < N
            {
                x_accum += array[i + 1];
            }
            xi
        })
    }
    else
    {
        // Return empty array
        unsafe {MaybeUninit::assume_init(MaybeUninit::uninit())}
    }
}

pub /*const*/ fn integrate_from<T, const N: usize, const M: usize>(array: [T; N], x0: T) -> [T; M]
where
    T: AddAssign<T> + Copy
{
    let mut x_accum = x0;

    crate::fill(/*const*/ |i| {
        let xi = x_accum;
        if i < N
        {
            x_accum += array[i];
        }
        xi
    })
}

pub /*const*/ fn reduce<T, const N: usize, R>(array: [T; N], mut reduce: R) -> Option<T>
where
    R: FnMut(T, T) -> T + /*~const*/ Destruct
{
    let this = ManuallyDrop::new(array);
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

pub /*const*/ fn try_sum<T, const N: usize>(array: [T; N]) -> Option<T>
where
    T: AddAssign
{
    let this = ManuallyDrop::new(array);
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

pub /*const*/ fn sum_from<T, const N: usize, S>(array: [T; N], mut from: S) -> S
where
    S: AddAssign<T>
{
    let this = ManuallyDrop::new(array);
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

pub /*const*/ fn try_product<T, const N: usize>(array: [T; N]) -> Option<T>
where
    T: MulAssign
{
    let this = ManuallyDrop::new(array);
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

pub /*const*/ fn product_from<T, const N: usize, S>(array: [T; N], mut from: S) -> S
where
    S: MulAssign<T>
{
    let this = ManuallyDrop::new(array);
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

pub /*const*/ fn max<T, const N: usize>(array: [T; N]) -> Option<T>
where
    T: Ord
{
    crate::reduce(array, T::max)
}
pub /*const*/ fn min<T, const N: usize>(array: [T; N]) -> Option<T>
where
    T: Ord
{
    crate::reduce(array, T::min)
}
pub /*const*/ fn first_max<T, const N: usize>(array: [T; N]) -> Option<T>
where
    T: PartialOrd<T> + /*~const*/ Destruct
{
    crate::reduce(array, /*const*/ |a, b| if a >= b {a} else {b})
}
pub /*const*/ fn first_min<T, const N: usize>(array: [T; N]) -> Option<T>
where
    T: PartialOrd<T> + /*~const*/ Destruct
{
    crate::reduce(array, /*const*/ |a, b| if a <= b {a} else {b})
}

pub /*const*/ fn argmax<T, const N: usize>(array: &[T; N]) -> Option<usize>
where
    T: PartialOrd<T>
{
    match crate::reduce(crate::enumerate(crate::each_ref(array)), |a, b| if a.1 >= b.1 {a} else {b})
    {
        Some((i, _)) => Some(i),
        None => None
    }
}
pub /*const*/ fn argmin<T, const N: usize>(array: &[T; N]) -> Option<usize>
where
    T: PartialOrd<T>
{
    match crate::reduce(crate::enumerate(crate::each_ref(array)), |a, b| if a.1 <= b.1 {a} else {b})
    {
        Some((i, _)) => Some(i),
        None => None
    }
}

pub /*const*/ fn add_all<T, const N: usize, Rhs>(array: [T; N], rhs: Rhs) -> [<T as Add<Rhs>>::Output; N]
where
    T: Add<Rhs>,
    Rhs: Copy
{
    crate::map(array, /*const*/ |x| x + rhs)
}
pub /*const*/ fn sub_all<T, const N: usize, Rhs>(array: [T; N], rhs: Rhs) -> [<T as Sub<Rhs>>::Output; N]
where
    T: Sub<Rhs>,
    Rhs: Copy
{
    crate::map(array, /*const*/ |x| x - rhs)
}
pub /*const*/ fn mul_all<T, const N: usize, Rhs>(array: [T; N], rhs: Rhs) -> [<T as Mul<Rhs>>::Output; N]
where
    T: Mul<Rhs>,
    Rhs: Copy
{
    crate::map(array, /*const*/ |x| x * rhs)
}
pub /*const*/ fn div_all<T, const N: usize, Rhs>(array: [T; N], rhs: Rhs) -> [<T as Div<Rhs>>::Output; N]
where
    T: Div<Rhs>,
    Rhs: Copy
{
    crate::map(array, /*const*/ |x| x / rhs)
}

pub /*const*/ fn add_all_neg<T, const N: usize, Rhs>(array: [T; N], rhs: Rhs) -> [<Rhs as Sub<T>>::Output; N]
where
    Rhs: Copy + Sub<T>
{
    crate::map(array, /*const*/ |x| rhs - x)
}
pub /*const*/ fn mul_all_inv<T, const N: usize, Rhs>(array: [T; N], rhs: Rhs) -> [<Rhs as Div<T>>::Output; N]
where
    Rhs: Copy + Div<T>
{
    crate::map(array, /*const*/ |x| rhs / x)
}

pub /*const*/ fn neg_all<T, const N: usize>(array: [T; N]) -> [<T as Neg>::Output; N]
where
    T: Neg
{
    crate::map(array, Neg::neg)
}

pub /*const*/ fn add_each<T, const N: usize, Rhs>(array: [T; N], rhs: [Rhs; N]) -> [<T as Add<Rhs>>::Output; N]
where
    T: Add<Rhs>
{
    crate::comap(array, rhs, Add::add)
}
pub /*const*/ fn sub_each<T, const N: usize, Rhs>(array: [T; N], rhs: [Rhs; N]) -> [<T as Sub<Rhs>>::Output; N]
where
    T: Sub<Rhs>
{
    crate::comap(array, rhs, Sub::sub)
}
pub /*const*/ fn mul_each<T, const N: usize, Rhs>(array: [T; N], rhs: [Rhs; N]) -> [<T as Mul<Rhs>>::Output; N]
where
    T: Mul<Rhs>
{
    crate::comap(array, rhs, Mul::mul)
}
pub /*const*/ fn div_each<T, const N: usize, Rhs>(array: [T; N], rhs: [Rhs; N]) -> [<T as Div<Rhs>>::Output; N]
where
    T: Div<Rhs>
{
    crate::comap(array, rhs, Div::div)
}

pub /*const*/ fn try_mul_dot<T, const N: usize, Rhs>(array: [T; N], rhs: [Rhs; N]) -> Option<<T as Mul<Rhs>>::Output>
where
    T: Mul<Rhs, Output: AddAssign>
{
    crate::try_sum(crate::mul_each(array, rhs))
}
pub /*const*/ fn mul_dot_bias<T, const N: usize, Rhs>(array: [T; N], rhs: [Rhs; N], bias: <T as Mul<Rhs>>::Output) -> <T as Mul<Rhs>>::Output
where
    T: Mul<Rhs, Output: AddAssign>
{
    crate::sum_from(crate::mul_each(array, rhs), bias)
}

pub /*const*/ fn mul_outer<T, const N: usize, Rhs, const M: usize>(array: &[T; N], rhs: &[Rhs; M]) -> [[<T as Mul<Rhs>>::Output; M]; N]
where
    T: Mul<Rhs> + Copy,
    Rhs: Copy
{
    crate::comap_outer(array, rhs, Mul::mul)
}
pub /*const*/ fn mul_cross<T, const N: usize, Rhs>(array: &[T; N], rhs: [&[Rhs; N]; N - 2]) -> [<T as Sub>::Output; N]
where
    T: MulAssign<Rhs> + Sub + Copy,
    Rhs: Copy
{
    crate::fill(/*const*/ |i| {
        let mut m_p = array[(i + 1) % N];
        let mut m_m = array[(i + (N - 1)) % N];

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

pub /*const*/ fn try_magnitude_squared<T, const N: usize>(array: [T; N]) -> Option<<T as Mul<T>>::Output>
where
    T: Mul<T, Output: AddAssign> + Copy
{
    crate::try_mul_dot(array, array)
}

pub const fn chain<T, const N: usize, const M: usize>(array: [T; N], rhs: [T; M]) -> [T; N + M]
{
    unsafe {private::merge_transmute(array, rhs)}
}
pub const fn rchain<T, const N: usize, const M: usize>(array: [T; N], rhs: [T; M]) -> [T; N + M]
{
    unsafe {private::merge_transmute(rhs, array)}
}

pub const fn array_spread<T, const N: usize, const M: usize>(array: [T; N]) -> ([[T; N / M]; M], [T; N % M])
where
    [(); M - 1]:,
    [(); N % M]:,
    [(); N / M]:
{
    let split = crate::array_chunks(array);

    let spread_t = unsafe {core::ptr::read(&split.0 as *const [[T; _]; _])};
    let rest = unsafe {core::ptr::read(&split.1 as *const [T; _])};
    core::mem::forget(split);

    (crate::transpose(spread_t), rest)
}
pub /*const*/ fn array_spread_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> ([&[Padded<T, M>; N / M]; M], &[T; N % M])
where
    [(); M - 1]:,
    [(); N % M]:
{
    let (left, right) = crate::rsplit_ptr(array, N % M);

    unsafe {(
        crate::fill(/*const*/ |i| &*left.add(i).cast()),
        &*right.cast()
    )}
}
pub /*const*/ fn array_spread_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> ([&mut [Padded<T, M>; N / M]; M], &mut [T; N % M])
where
    [(); M - 1]:,
    [(); N % M]:
{
    let (left, right) = crate::rsplit_mut_ptr(array, N % M);

    unsafe {(
        crate::fill(/*const*/ |i| &mut *left.add(i).cast()),
        &mut *right.cast()
    )}
}

pub const fn array_rspread<T, const N: usize, const M: usize>(array: [T; N]) -> ([T; N % M], [[T; N / M]; M])
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
pub /*const*/ fn array_rspread_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> (&[T; N % M], [&[Padded<T, M>; N / M]; M])
where
    [(); M - 1]:,
    [(); N % M]:
{
    let (left, right) = crate::split_ptr(array, N % M);

    unsafe {(
        &*left.cast(),
        crate::fill(/*const*/ |i| &*right.add(i).cast())
    )}
}
pub /*const*/ fn array_rspread_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> (&mut [T; N % M], [&mut [Padded<T, M>; N / M]; M])
where
    [(); M - 1]:,
    [(); N % M]:
{
    let (left, right) = crate::split_mut_ptr(array, N % M);

    unsafe {(
        &mut *left.cast(),
        crate::fill(/*const*/ |i| &mut *right.add(i).cast())
    )}
}

pub const fn array_spread_exact<T, const N: usize, const M: usize>(array: [T; N]) -> [[T; N / M]; M]
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
pub /*const*/ fn array_spread_exact_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> [&[Padded<T, M>; N / M]; M]
where
    [(); M - 1]:,
    [(); 0 - N % M]:
{
    let ptr = array as *const T;
    
    crate::fill(/*const*/ |i| unsafe {&*ptr.add(i).cast()})
}
pub /*const*/ fn array_spread_exact_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> [&mut [Padded<T, M>; N / M]; M]
where
    [(); M - 1]:,
    [(); 0 - N % M]:
{
    let ptr = array as *mut T;
    
    crate::fill(/*const*/ |i| unsafe {&mut *ptr.add(i).cast()})
}

pub const fn array_chunks<T, const N: usize, const M: usize>(array: [T; N]) -> ([[T; M]; N / M], [T; N % M])
{
    unsafe {private::split_transmute(array)}
}
pub const fn array_chunks_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> (&[[T; M]; N / M], &[T; N % M])
{
    let (ptr_left, ptr_right) = crate::rsplit_ptr(array, N % M);
    unsafe {(&*ptr_left.cast(), &*ptr_right.cast())}
}
pub const fn array_chunks_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> (&mut [[T; M]; N / M], &mut [T; N % M])
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

pub const fn array_chunks_exact<T, const N: usize, const M: usize>(array: [T; N]) -> [[T; M]; N / M]
where
    [(); 0 - N % M]:,
    [(); N / M]:
{
    unsafe {private::transmute_unchecked_size(array)}
}
pub const fn array_chunks_exact_ref<T, const N: usize, const M: usize>(array: &[T; N]) -> &[[T; M]; N / M]
where
    [(); 0 - N % M]:,
    [(); N / M]:
{
    unsafe {&*array.as_ptr().cast()}
}
pub const fn array_chunks_exact_mut<T, const N: usize, const M: usize>(array: &mut [T; N]) -> &mut [[T; M]; N / M]
where
    [(); 0 - N % M]:,
    [(); N / M]:
{
    unsafe {&mut *array.as_mut_ptr().cast()}
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

pub /*const*/ fn each_ref<T, const N: usize>(array: &[T; N]) -> [&T; N]
{
    let ptr = array as *const T;
    crate::fill(/*const*/ |i| {
        let y = unsafe {&*ptr.add(i)};
        y
    })
}
pub /*const*/ fn each_mut<T, const N: usize>(array: &mut [T; N]) -> [&mut T; N]
{
    let ptr = array as *mut T;
    crate::fill(/*const*/ |i| {
        let y = unsafe {&mut *ptr.add(i)};
        y
    })
}

impl<T, const N: usize> /*const*/ ArrayOps<T, N> for [T; N]
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
        F: /*~const*/ FnMut(usize) -> T + /*~const*/ Destruct
    {
        crate::fill(&mut fill)
    }
    fn rfill<F>(mut fill: F) -> Self
    where
        F: /*~const*/ FnMut(usize) -> T + /*~const*/ Destruct
    {
        crate::rfill(&mut fill)
    }
    
    /*fn for_each<F>(self, mut action: F) -> ()
    where
        F: /*~const*/ FnMut(T) -> () + /*~const*/ Destruct
    {
        self.for_each_ref(/*const*/ |x| action(unsafe {(x as *const T).read()}));
        core::mem::forget(self)
    }
    fn for_each_ref<F>(&self, mut action: F) -> ()
    where
        F: /*~const*/ FnMut(&T) -> () + /*~const*/ Destruct
    {
        let mut iter = self.const_iter();
        while let Some(next) = iter.next()
        {
            action(next);
        }
    }
    fn for_each_mut<F>(&mut self, mut action: F) -> ()
    where
        F: /*~const*/ FnMut(&mut T) -> () + /*~const*/ Destruct
    {
        let mut iter = self.const_iter_mut();
        while let Some(next) = iter.next()
        {
            action(next);
        }
    }*/
    
    fn truncate<const M: usize>(self) -> [T; M]
    where
        T: /*~const*/ Destruct,
        [(); N - M]:
    {
        crate::truncate(self)
    }
    fn rtruncate<const M: usize>(self) -> [T; M]
    where
        T: /*~const*/ Destruct,
        [(); N - M]:
    {
        crate::rtruncate(self)
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

    fn resize<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: /*~const*/ FnMut(usize) -> T + /*~const*/ Destruct,
        T: /*~const*/ Destruct
    {
        crate::resize(self, fill)
    }
    fn rresize<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: /*~const*/ FnMut(usize) -> T + /*~const*/ Destruct,
        T: /*~const*/ Destruct
    {
        crate::rresize(self, fill)
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
    
    fn extend<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: /*~const*/ FnMut(usize) -> T + /*~const*/ Destruct,
        [(); M - N]:
    {
        crate::extend(self, fill)
    }
    fn rextend<const M: usize, F>(self, fill: F) -> [T; M]
    where
        F: FnMut(usize) -> T + /*~const*/ Destruct,
        [(); M - N]:
    {
        crate::rextend(self, fill)
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
    
    fn map2<Map>(self, map: Map) -> [Map::Output; N]
    where
        Map: /*~const*/ FnMut<(T,)> + /*~const*/ Destruct
    {
        crate::map(self, map)
    }
    fn map_outer<Map>(&self, map: Map) -> [[Map::Output; N]; N]
    where
        Map: /*~const*/ FnMut<(T, T)> + /*~const*/ Destruct,
        T: Copy
    {
        crate::map_outer(self, map)
    }

    fn comap<Map, Rhs>(self, rhs: [Rhs; N], map: Map) -> [Map::Output; N]
    where
        Map: /*~const*/ FnMut<(T, Rhs)> + /*~const*/ Destruct
    {
        crate::comap(self, rhs, map)
    }
    fn comap_outer<Map, Rhs, const M: usize>(&self, rhs: &[Rhs; M], map: Map) -> [[Map::Output; M]; N]
    where
        Map: /*~const*/ FnMut<(T, Rhs)> + /*~const*/ Destruct,
        T: Copy,
        Rhs: Copy
    {
        crate::comap_outer(self, rhs, map)
    }
    
    fn zip<Z>(self, other: [Z; N]) -> [(T, Z); N]
    {
        crate::zip(self, other)
    }
    
    fn zip_outer<Z, const M: usize>(&self, other: &[Z; M]) -> [[(T, Z); M]; N]
    where
        T: Copy,
        Z: Copy
    {
        crate::zip_outer(self, other)
    }
    
    fn enumerate(self) -> [(usize, T); N]
    {
        crate::enumerate(self)
    }
    
    fn diagonal<const H: usize, const W: usize>(self) -> [[T; W]; H]
    where
        T: /*~const*/ Default,
        [(); H - N]:,
        [(); W - N]:
    {
        crate::diagonal(self)
    }

    fn differentiate(self) -> [<T as Sub<T>>::Output; N.saturating_sub(1)]
    where
        [(); N.saturating_sub(1)]:,
        T: /*~const*/ Sub<T> + Copy + /*~const*/ Destruct
    {
        crate::differentiate(self)
    }

    fn integrate(self) -> Self
        where
            T: /*~const*/ AddAssign<T> + Copy + /*~const*/ Destruct
    {
        crate::integrate(self)
    }
    
    fn integrate_from<const M: usize>(self, x0: T) -> [T; M]
    where
        T: /*~const*/ AddAssign<T> + Copy + /*~const*/ Destruct
    {
        crate::integrate_from(self, x0)
    }

    fn reduce<R>(self, reduce: R) -> Option<T>
    where
        R: /*~const*/ FnMut(T, T) -> T + /*~const*/ Destruct
    {
        crate::reduce(self, reduce)
    }
    
    fn try_sum(self) -> Option<T>
    where
        T: /*~const*/ AddAssign
    {
        crate::try_sum(self)
    }

    fn sum_from<S>(self, from: S) -> S
    where
        S: /*~const*/ AddAssign<T>
    {
        crate::sum_from(self, from)
    }
        
    fn try_product(self) -> Option<T>
    where
        T: /*~const*/ MulAssign
    {
        crate::try_product(self)
    }

    fn product_from<P>(self, from: P) -> P
    where
        P: /*~const*/ MulAssign<T>
    {
        crate::product_from(self, from)
    }
    
    fn max(self) -> Option<T>
    where
        T: /*~const*/ Ord
    {
        crate::max(self)
    }
        
    fn min(self) -> Option<T>
    where
        T: /*~const*/ Ord
    {
        crate::min(self)
    }
    
    fn first_max(self) -> Option<T>
    where
        T: /*~const*/ PartialOrd<T>
    {
        crate::first_max(self)
    }
        
    fn first_min(self) -> Option<T>
    where
        T: /*~const*/ PartialOrd<T>
    {
        crate::first_min(self)
    }
    
    fn argmax(&self) -> Option<usize>
    where
        T: /*~const*/ PartialOrd<T>
    {
        crate::argmax(self)
    }
        
    fn argmin(&self) -> Option<usize>
    where
        T: /*~const*/ PartialOrd<T>
    {
        crate::argmin(self)
    }
    
    fn add_all<Rhs>(self, rhs: Rhs) -> [<T as Add<Rhs>>::Output; N]
    where
        T: /*~const*/ Add<Rhs>,
        Rhs: Copy
    {
        crate::add_all(self, rhs)
    }
    fn sub_all<Rhs>(self, rhs: Rhs) -> [<T as Sub<Rhs>>::Output; N]
    where
        T: /*~const*/ Sub<Rhs>,
        Rhs: Copy
    {
        crate::sub_all(self, rhs)
    }
    fn mul_all<Rhs>(self, rhs: Rhs) ->  [<T as Mul<Rhs>>::Output; N]
    where
        T: /*~const*/ Mul<Rhs>,
        Rhs: Copy
    {
        crate::mul_all(self, rhs)
    }
    fn div_all<Rhs>(self, rhs: Rhs) -> [<T as Div<Rhs>>::Output; N]
    where
        T: /*~const*/ Div<Rhs>,
        Rhs: Copy
    {
        crate::div_all(self, rhs)
    }
    
    fn add_all_neg<Rhs>(self, rhs: Rhs) -> [<Rhs as Sub<T>>::Output; N]
    where
        Rhs: Copy + /*~const*/ Sub<T>
    {
        crate::add_all_neg(self, rhs)
    }
    fn mul_all_inv<Rhs>(self, rhs: Rhs) -> [<Rhs as Div<T>>::Output; N]
    where
        Rhs: Copy + /*~const*/ Div<T>
    {
        crate::mul_all_inv(self, rhs)
    }
    
    fn neg_all(self) -> [<T as Neg>::Output; N]
    where
        T: /*~const*/ Neg
    {
        crate::neg_all(self)
    }
    
    fn add_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Add<Rhs>>::Output; N]
    where
        T: /*~const*/ Add<Rhs>
    {
        crate::add_each(self, rhs)
    }
    fn sub_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Sub<Rhs>>::Output; N]
    where
        T: /*~const*/ Sub<Rhs>
    {
        crate::sub_each(self, rhs)
    }
    fn mul_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Mul<Rhs>>::Output; N]
    where
        T: /*~const*/ Mul<Rhs>
    {
        crate::mul_each(self, rhs)
    }
    fn div_each<Rhs>(self, rhs: [Rhs; N]) -> [<T as Div<Rhs>>::Output; N]
    where
        T: /*~const*/ Div<Rhs>
    {
        crate::div_each(self, rhs)
    }

    fn try_mul_dot<Rhs>(self, rhs: [Rhs; N]) -> Option<<T as Mul<Rhs>>::Output>
    where
        T: /*~const*/ Mul<Rhs, Output: /*~const*/ AddAssign>
    {
        crate::try_mul_dot(self, rhs)
    }
    
    fn mul_dot_bias<Rhs>(self, rhs: [Rhs; N], bias: <T as Mul<Rhs>>::Output) -> <T as Mul<Rhs>>::Output
    where
        T: /*~const*/ Mul<Rhs, Output: /*~const*/ AddAssign>
    {
        crate::mul_dot_bias(self, rhs, bias)
    }

    fn mul_outer<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [[<T as Mul<Rhs>>::Output; M]; N]
    where
        T: /*~const*/ Mul<Rhs> + Copy,
        Rhs: Copy
    {
        crate::mul_outer(self, rhs)
    }
    
    fn mul_cross<Rhs>(&self, rhs: [&[Rhs; N]; N - 2]) -> [<T as Sub>::Output; N]
    where
        T: /*~const*/ MulAssign<Rhs> + /*~const*/ Sub + Copy,
        Rhs: Copy
    {
        crate::mul_cross(self, rhs)
    }
    
    fn try_magnitude_squared(self) -> Option<<T as Mul<T>>::Output>
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign> + Copy
    {
        crate::try_magnitude_squared(self)
    }
    
    fn chain<const M: usize>(self, rhs: [T; M]) -> [T; N + M]
    {
        crate::chain(self, rhs)
    }
    
    fn rchain<const M: usize>(self, rhs: [T; M]) -> [T; N + M]
    {
        crate::rchain(self, rhs)
    }
    
    fn array_spread<const M: usize>(self) -> ([[T; N / M]; M], [T; N % M])
    where
        [(); M - 1]:,
        [(); N % M]:,
        [(); N / M]:
    {
        crate::array_spread(self)
    }
    fn array_spread_ref<const M: usize>(&self) -> ([&[Padded<T, M>; N / M]; M], &[T; N % M])
    where
        [(); M - 1]:,
        [(); N % M]:
    {
        crate::array_spread_ref(self)
    }
    fn array_spread_mut<const M: usize>(&mut self) -> ([&mut [Padded<T, M>; N / M]; M], &mut [T; N % M])
    where
        [(); M - 1]:,
        [(); N % M]:
    {
        crate::array_spread_mut(self)
    }
    
    fn array_rspread<const M: usize>(self) -> ([T; N % M], [[T; N / M]; M])
    where
        [(); M - 1]:,
        [(); N % M]:,
        [(); N / M]:
    {
        crate::array_rspread(self)
    }
    fn array_rspread_ref<const M: usize>(&self) -> (&[T; N % M], [&[Padded<T, M>; N / M]; M])
    where
        [(); M - 1]:,
        [(); N % M]:
    {
        crate::array_rspread_ref(self)
    }
    fn array_rspread_mut<const M: usize>(&mut self) -> (&mut [T; N % M], [&mut [Padded<T, M>; N / M]; M])
    where
        [(); M - 1]:,
        [(); N % M]:
    {
        crate::array_rspread_mut(self)
    }
    fn array_spread_exact<const M: usize>(self) -> [[T; N / M]; M]
    where
        [(); M - 1]:,
        [(); 0 - N % M]:,
        [(); N / M]:
    {
        crate::array_spread_exact(self)
    }
    fn array_spread_exact_ref<const M: usize>(&self) -> [&[Padded<T, M>; N / M]; M]
    where
        [(); M - 1]:,
        [(); 0 - N % M]:
    {
        crate::array_spread_exact_ref(self)
    }
    fn array_spread_exact_mut<const M: usize>(&mut self) -> [&mut [Padded<T, M>; N / M]; M]
    where
        [(); M - 1]:,
        [(); 0 - N % M]:
    {
        crate::array_spread_exact_mut(self)
    }
    
    fn array_chunks<const M: usize>(self) -> ([[T; M]; N / M], [T; N % M])
    {
        crate::array_chunks(self)
    }
    fn array_chunks_ref<const M: usize>(&self) -> (&[[T; M]; N / M], &[T; N % M])
    {
        crate::array_chunks_ref(self)
    }
    fn array_chunks_mut<const M: usize>(&mut self) -> (&mut [[T; M]; N / M], &mut [T; N % M])
    {
        crate::array_chunks_mut(self)
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
    
    fn array_chunks_exact<const M: usize>(self) -> [[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:
    {
        crate::array_chunks_exact(self)
    }
    fn array_chunks_exact_ref<const M: usize>(&self) -> &[[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:
    {
        crate::array_chunks_exact_ref(self)
    }
    fn array_chunks_exact_mut<const M: usize>(&mut self) -> &mut [[T; M]; N / M]
    where
        [(); 0 - N % M]:,
        [(); N / M]:
    {
        crate::array_chunks_exact_mut(self)
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
        crate::each_ref(self)
    }
    fn each_mut2(&mut self) -> [&mut T; N]
    {
        crate::each_mut(self)
    }
}