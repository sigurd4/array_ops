use super::*;

pub trait PartitionedArraySplit<const PARTS: usize, const M: usize>
where
    [(); PARTS - M]:
{
    const SPLIT_PART_LENGTHS_LEFT: [usize; M];
    const SPLIT_SERIALIZED_LENGTH_LEFT: usize = Self::SPLIT_PART_LENGTHS_LEFT.sum();
    const SPLIT_PART_LENGTHS_LEFT_SLICE: &'static [usize] = &Self::SPLIT_PART_LENGTHS_LEFT;

    const SPLIT_PART_LENGTHS_RIGHT: [usize; PARTS - M];
    const SPLIT_SERIALIZED_LENGTH_RIGHT: usize = Self::SPLIT_PART_LENGTHS_RIGHT.sum();
    const SPLIT_PART_LENGTHS_RIGHT_SLICE: &'static [usize] = &Self::SPLIT_PART_LENGTHS_RIGHT;
    
    const RSPLIT_PART_LENGTHS_LEFT: [usize; M];
    const RSPLIT_SERIALIZED_LENGTH_LEFT: usize = Self::RSPLIT_PART_LENGTHS_LEFT.sum();
    const RSPLIT_PART_LENGTHS_LEFT_SLICE: &'static [usize] = &Self::RSPLIT_PART_LENGTHS_LEFT;

    const RSPLIT_PART_LENGTHS_RIGHT: [usize; PARTS - M];
    const RSPLIT_SERIALIZED_LENGTH_RIGHT: usize = Self::RSPLIT_PART_LENGTHS_RIGHT.sum();
    const RSPLIT_PART_LENGTHS_RIGHT_SLICE: &'static [usize] = &Self::RSPLIT_PART_LENGTHS_RIGHT;
}

pub trait PartitionedArrayOps<T, const PARTS: usize>
{
    const PART_LENGTHS: [usize; PARTS];
    const PART_LENGTHS_SLICE: &'static [usize] = &Self::PART_LENGTHS;
    const SERIALIZED_LENGTH: usize = sum_len::<{Self::PART_LENGTHS}>();

    type SplitLeft<const M: usize, const S: usize>: PartitionedArrayOps<T, {M}> = PartitionedArray<T, {Self::SPLIT_PART_LENGTHS_LEFT_SLICE}>
    where
        [(); PARTS - M]:,
        [(); S]:,
        Self: PartitionedArraySplit<PARTS, M, SPLIT_SERIALIZED_LENGTH_LEFT = {S}>;
    type SplitRight<const M: usize, const S: usize>: PartitionedArrayOps<T, {PARTS - M}> = PartitionedArray<T, {Self::SPLIT_PART_LENGTHS_RIGHT_SLICE}>
    where
        [(); PARTS - M]:,
        [(); S]:,
        Self: PartitionedArraySplit<PARTS, M, SPLIT_SERIALIZED_LENGTH_RIGHT = {S}>;

    fn split_lengths_left(mid: usize) -> &'static [usize];
    fn split_lengths_right(mid: usize) -> &'static [usize];
    fn split_lengths(mid: usize) -> (&'static [usize], &'static [usize]);

    fn split_parts<const M: usize, const S: usize>(self)
        -> (Self::SplitLeft<M, S>, Self::SplitRight<M, S>)
    where
        [(); PARTS - M]:,
        [(); S]:,
        Self: PartitionedArraySplit<PARTS, M, SPLIT_SERIALIZED_LENGTH_LEFT = {S}, SPLIT_SERIALIZED_LENGTH_RIGHT = {S}>;
}