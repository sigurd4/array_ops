#![cfg_attr(not(any(feature = "std", test)), no_std)]

#![feature(associated_type_defaults)]
#![feature(const_trait_impl)]
#![feature(unboxed_closures)]
#![feature(const_for)]
#![feature(associated_type_bounds)]
#![feature(generic_arg_infer)]
#![feature(const_maybe_uninit_uninit_array)]
#![feature(const_maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
//#![feature(effects)]
#![feature(const_mut_refs)]
#![feature(const_refs_to_cell)]
#![feature(const_ptr_write)]
#![feature(const_maybe_uninit_assume_init)]
#![feature(associated_const_equality)]
#![feature(portable_simd)]
#![feature(const_swap)]
#![cfg_attr(feature = "std", feature(new_uninit))]

#![feature(const_closures)]
#![feature(generic_const_exprs)]

moddef::moddef!(
    flat(pub) mod {
        array_2d_ops,
        array_nd_ops,
        array_ops,
        array_simd_ops,
        //partitioned_array,
        //partitioned_array_ops
    },
    mod private
);

pub use array_trait::*;
pub use slice_ops;

pub const fn min_len(a: usize, b: usize) -> usize
{
    if a < b
    {
        a
    }
    else
    {
        b
    }
}
pub const fn max_len(a: usize, b: usize) -> usize
{
    if a > b
    {
        a
    }
    else
    {
        b
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use array_trait::ArrayNd;
    use slice_ops::Padded;

    use super::*;

    #[test]
    fn mod0()
    {
        let a = [1, 2, 3];
        let c = a.array_chunks_exact::<1>();
        println!("{:?}", c);
    }

    #[test]
    fn gpa()
    {
        #[repr(u8)]
        enum Grade
        {
            A = 5,
            B = 4,
            C = 3,
            D = 2,
            E = 1
        }

        const GRADES_UNI: [(u8, Grade); 21] = [
            (5, Grade::C), // Ingeniørrollen
            (5, Grade::A), // Programmering for beregning
            (5, Grade::B), // Elektrisitetslære
            (5, Grade::D), // Digitalteknikk
            (10, Grade::A), // Programmering og mikrokontrollere
            (10, Grade::A), // Matematikk 1
            (5, Grade::C), // Fysikk 1 - Mekanikk
            (5, Grade::A), // Elektrisitetslære 2
            (5, Grade::A), // Programmerbare logiske kretser
            (10, Grade::A), // Matematikk 2
            (5, Grade::C), // Kommunikasjon
            (10, Grade::B), // Analog elektronikk
            (10, Grade::B), // Systems design and engineering
            (5, Grade::C), // Statistikk
            (10, Grade::E), // Signalbehandling
            (10, Grade::C), // Reguleringsteknikk 1
            (5, Grade::B), // Fysikk 2 - Elektromagnetisme
            (10, Grade::C), // Reguleringsteknikk 2
            (10, Grade::C), // Matematikk 3
            (10, Grade::C), // Instrumentering og styring
            (20, Grade::B) // Bacheloroppgave - Automatisk gir-system for Lone Wolf ATV
        ];
        const GRADES_VGS: [u8; 23] = [
            5, // Engelsk
            2, // Spansk II
            4, // Geografi
            4, // Historie
            4, // Kroppsøving
            4, // Matematikk 1T
            5, // Naturfag
            4, // Norsk hovedmål
            4, // Norsk hovedmål, eksamen
            3, // Norsk sidemål
            2, // Norsk sidemål, eksamen
            3, // Norsk
            3, // Religion og etikk
            4, // Samfunnsfag
            4, // Fysikk 1
            4, // Fysikk 2
            5, // Fysikk 2, eksamen
            3, // Kjemi
            4, // Informasjonsteknologi 1
            5, // Informasjonsteknologi 2
            4, // Teknologi og forskningslære 1
            3, // Matematikk R1
            4, // Matematikk R2
        ];

        let gpa_uni: f32 = GRADES_UNI.map2(|(pts, grade)| (pts*grade as u8) as u16)
            .sum_from(0) as f32
            /GRADES_UNI.map2(const |(pts, _)| pts as u16)
            .sum_from(0) as f32;

        println!("{}", gpa_uni);

        let gpa_vgs: f32 = GRADES_VGS.map2(|grade| grade as u16)
            .sum_from(0) as f32
            /GRADES_VGS.len() as f32;
            
        println!("{}", gpa_vgs);
    }

    #[test]
    fn benchmark()
    {
        const N: usize = 64;
        const M: usize = 256;
        
        assert_eq!(<[[[u8; 2]; N]; M]>::DIMENSIONS, [M, N, 2]);

        let a: [[[u8; 2]; N]; M] = ArrayNdOps::fill_nd(|i| i.map2(|i| i as u8));

        let t0 = SystemTime::now();
        for m in 0..M
        {
            for n in 0..N
            {
                //<[u8; N]>::fill(|i| i as u8);
                //a[m].truncate::<{N/2}>();
                //a[m].resize::<{N/2}, _>(|i| [m as u8, i as u8]);
                //let (matrix, _) = a[m].array_spread::<3>();
                for k in 0..2
                {
                    let i = [m, n, k];
                    let _ = *a.get_nd(i).unwrap();
                }
            }
        }
        let t = t0.elapsed().unwrap();
        println!("t = {:?}", t); //10.5832ms
    }

    #[test]
    fn reduce()
    {
        const A: [[(u8, u8); 3]; 2] = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)]
        ];
        
        let r: (u8, u8) = A.reduce_nd(|(a1, a2), (b1, b2)| (a1 + b1, a2 + b2)).unwrap();
        
        assert_eq!(r, (3, 6));
    }

    #[test]
    fn rotate()
    {
        let mut a = [1, 2, 3, 4, 5];

        a.rotate_left2(2);
        println!("{:?}", a);
    }

    #[test]
    fn test_spread_align()
    {
        let str = b"abcdefghijklmnopqrstuvwxyz".map(|c| c as char);
        
        println!("Alignment char = {}", core::mem::align_of::<char>());
        println!("Alignment padded x3 char = {}", core::mem::align_of::<Padded<char, 3>>());
        
        println!("Alignment String = {}", core::mem::align_of::<String>());
        println!("Alignment padded x3 String = {}", core::mem::align_of::<Padded<String, 3>>());

        println!("str: {:?}", str);
        println!("spread: {:?}", str.array_spread_ref::<3>());
        println!("chunks: {:?}", str.array_chunks_ref::<3>());

        assert_eq!(
            str.array_spread::<3>(),
            (
                [
                    ['a', 'd', 'g', 'j', 'm', 'p', 's', 'v'],
                    ['b', 'e', 'h', 'k', 'n', 'q', 't', 'w'],
                    ['c', 'f', 'i', 'l', 'o', 'r', 'u', 'x']
                ],
                ['y', 'z']
            )
        );
        assert_eq!(
            str.array_chunks::<3>(),
            (
                [
                    ['a', 'b', 'c'],
                    ['d', 'e', 'f'],
                    ['g', 'h', 'i'],
                    ['j', 'k', 'l'],
                    ['m', 'n', 'o'],
                    ['p', 'q', 'r'],
                    ['s', 't', 'u'],
                    ['v', 'w', 'x']
                ],
                ['y', 'z']
            )
        );
    }

    #[test]
    fn nd_test()
    {
        type T = u8;

        const ND: [[T; 3]; 3] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ];
        let flat: [T; 9] = ND.flatten_nd_array();
        assert_eq!(flat, [1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let nd_t: [[T; 3]; 3] = ND.transpose();

        let flat_t: [T; 9] = nd_t.flatten_nd_array();
        assert_eq!(flat_t, [1, 4, 7, 2, 5, 8, 3, 6, 9]);
    }

    #[test]
    fn generate_impl_nd_array_macro_args()
    {
        const R: usize = 110;

        print!("impl_nd_array!(\n   ");
        let mut c = 0;
        for i in 0usize..256
        {
            c += (i.max(1)).ilog10() as usize + 3;
            if c > R
            {
                print!("\n   ");
                c = 0;
            }
            print!(" _{}", i);
        }
        println!("\n);")
    }
}