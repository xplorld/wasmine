// use nom::{map_opt, IResult};
// use nom::number::complete::le_u8;

// enum E {
//     A, B,
// }

// fn f(u: u8) -> Option<E> {
//     match u {
//         0x2 => Some(E::A),
//         0x4 => Some(E::B),
//         _ => None,
//     }
// }

// fn parse(input: &[u8]) -> IResult<&[u8], E> {
//     map_opt!(le_u8, f)
// }