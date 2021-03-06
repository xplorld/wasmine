use crate::instrs::Instr;
use crate::types::*;
use nom::bytes::complete::tag;
use nom::combinator::opt;
use nom::error::ErrorKind;
use nom::number::complete::{le_f32, le_f64, le_u32, le_u64, le_u8};
use nom::IResult;

use std::str::from_utf8;

fn flatten<T>(o: Option<Option<T>>) -> Option<T> {
    o.unwrap_or(None)
}

fn zero_or_one<T, F>(i: &[u8], f: F) -> IResult<&[u8], Option<T>> where
    F: Fn(&[u8]) -> IResult<&[u8], T> {
        let (i, count) = le_u32(i)?;
        match count {
            0 => Ok((i, None)),
            1 => {
                let (i, t) = f(i)?;
                Ok((i, Some(t)))
            }
            _ => Err(nom::Err::Error((i, ErrorKind::Verify)))
        }
}

/**
 * both is None -> Some(empty)
 * one is None -> None
 * both is Some, size mismatch -> None
 * both is Some, size matches -> Some(vec)
 */
fn zip_funcs(input: (Option<Vec<usize>>, Option<Vec<Code>>)) -> Option<Vec<Function>> {
    match input {
        (None, None) => Some(vec![]),
        (Some(types), Some(codes)) => {
            if types.len() != codes.len() {
                None
            } else {
                Some(
                    types
                        .into_iter()
                        .zip(codes.into_iter())
                        .map(|(type_, code)| Function { type_, code })
                        .collect(),
                )
            }
        }
        _ => None,
    }
}

named!(le_usize<usize>, map!(le_u32, |l| l as usize));

/**
 * various attenpts to implement tag_byte using tag!, switch!, verify! or alt!
 * has failed. Have to write one on my own.
 */
fn tag_byte(input: &[u8], b: u8) -> IResult<&[u8], u8> {
    let (i, o) = le_u8(input)?;
    if o == b {
        Ok((i, o))
    } else {
        Err(nom::Err::Error((i, ErrorKind::Tag)))
    }
}

named!(name<&str>, map_res!(length_data!(le_u32), from_utf8));


named!(
    valtype<ValType>,
    switch!(le_u8,
          0x7f => value!(ValType::I32)
        | 0x7e => value!(ValType::I64)
        | 0x7d => value!(ValType::F32)
        | 0x7c => value!(ValType::F64)
    )
);

#[rustfmt::skip]
named!(
    blocktype<Option<ValType>>,
    alt!(
        value!(None, call!(tag_byte, 0x40))
      | opt!(valtype))
);

#[rustfmt::skip]
named!(
    functype<Type>,
    do_parse!(
        call!(tag_byte, 0x60) >>
        args: length_count!(le_u32, valtype) >>
        ret: verify!(length_count!(le_u32, valtype), 
                    |v: &Vec<ValType>| v.len() <= 1) >>
        (Type {
            args,
            ret: ret.first().map(ValType::clone)
        })
    )
);

named!(
    limits<Limits>,
    switch!(le_u8,
        0x00 => map!(le_usize, |min| Limits {min, max : None})
        | 0x01 => do_parse!(
            min: le_usize >>
            max: le_usize >>
            (Limits {min, max: Some(max)})
        )
    )
);

named!(memtype<Mem>, map!(limits, |limits| Mem { limits }));

named!(
    tabletype<Table>,
    do_parse!(
        call!(tag_byte, 0x70) >> // funcref
        limits: limits >>
        (Table {
            limits,
        })
    )
);

named!(
    mut_<Mut>,
    switch!(le_u8,
          0x00 => value!(Mut::Const)
        | 0x01 => value!(Mut::Var)
    )
);

named!(
    globaltype<GlobalType>,
    do_parse!(vt: valtype >> m: mut_ >> (GlobalType { mut_: m, type_: vt }))
);

named!(
    memarg<Memarg>,
    do_parse!(align: le_u32 >> offset: le_u32 >> (Memarg { align, offset }))
);

named!(
    instr<Instr>,
    switch!(le_u8,
          0x00 => value!(Instr::Unreachable)
        | 0x01 => value!(Instr::Nop)
        | 0x02 => do_parse!(
            type_: blocktype >>
            instrs: call!(instrs_till, 0x0b) >>
            (Instr::Block (Block {type_, expr: Expr {instrs}})))
        | 0x03 => do_parse!(
            type_: blocktype >>
            instrs: call!(instrs_till, 0x0b) >>
            (Instr::Loop (Block {type_, expr: Expr{instrs}})))
        | 0x04 => alt!(
            do_parse!(
                type_: blocktype >>
                then: call!(instrs_till, 0x05) >>
                else_: call!(instrs_till, 0x0b) >>
                (Instr::IfElse{
                    then: Block{type_, expr: Expr{instrs: then}},
                    else_: Some(Block{type_, expr: Expr{instrs: else_}})})
            )
          | do_parse!(
                type_: blocktype >>
                instrs: call!(instrs_till, 0x0b) >>
                (Instr::IfElse{
                    then: Block {type_, expr:Expr{instrs}},
                    else_: None})
            )
        )
        | 0x0c => map!(le_usize, |idx| Instr::Br(idx))
        | 0x0d => map!(le_usize, |idx| Instr::BrIf(idx))
        | 0x0d => do_parse!(
            table: length_count!(le_u32, le_usize) >>
            label: le_usize >>
            (Instr::BrTable(BrTableArgs {table, label}))
        )
        | 0x0f => value!(Instr::Return)
        | 0x10 => map!(le_usize, |idx| Instr::Call(idx))
        | 0x11 => do_parse!(
            idx: le_usize >>
            call!(tag_byte, 0x00) >>
            (Instr::CallIndirect(idx))
        )

        | 0x1a => value!(Instr::Drop)
        | 0x1b => value!(Instr::Select)

        | 0x20 => map!(le_usize, |idx| Instr::LocalGet(idx))
        | 0x21 => map!(le_usize, |idx| Instr::LocalSet(idx))
        | 0x22 => map!(le_usize, |idx| Instr::LocalTee(idx))
        | 0x23 => map!(le_usize, |idx| Instr::GlobalGet(idx))
        | 0x24 => map!(le_usize, |idx| Instr::GlobalSet(idx))

        | 0x28 => map!(memarg, |m| Instr::I32Load(m))
        | 0x29 => map!(memarg, |m| Instr::I64Load(m))
        | 0x2A => map!(memarg, |m| Instr::F32Load(m))
        | 0x2B => map!(memarg, |m| Instr::F64Load(m))
        | 0x2C => map!(memarg, |m| Instr::I32Load8S(m))
        | 0x2D => map!(memarg, |m| Instr::I32Load8U(m))
        | 0x2E => map!(memarg, |m| Instr::I32Load16S(m))
        | 0x2F => map!(memarg, |m| Instr::I32Load16U(m))
        | 0x30 => map!(memarg, |m| Instr::I64Load8S(m))
        | 0x31 => map!(memarg, |m| Instr::I64Load8U(m))
        | 0x32 => map!(memarg, |m| Instr::I64Load16S(m))
        | 0x33 => map!(memarg, |m| Instr::I64Load16U(m))
        | 0x34 => map!(memarg, |m| Instr::I64Load32S(m))
        | 0x35 => map!(memarg, |m| Instr::I64Load32U(m))
        | 0x36 => map!(memarg, |m| Instr::I32Store(m))
        | 0x37 => map!(memarg, |m| Instr::I64Store(m))
        | 0x38 => map!(memarg, |m| Instr::F32Store(m))
        | 0x39 => map!(memarg, |m| Instr::F64Store(m))
        | 0x3A => map!(memarg, |m| Instr::I32Store8(m))
        | 0x3B => map!(memarg, |m| Instr::I32Store16(m))
        | 0x3C => map!(memarg, |m| Instr::I64Store8(m))
        | 0x3D => map!(memarg, |m| Instr::I64Store16(m))
        | 0x3E => map!(memarg, |m| Instr::I64Store32(m))
        | 0x3F => value!(Instr::MemorySize, call!(tag_byte, 0x00))
        | 0x40 => value!(Instr::MemoryGrow, call!(tag_byte, 0x00))

        | 0x41 => map!(le_u32, |i| Instr::I32Const(i))
        | 0x42 => map!(le_u64, |i| Instr::I64Const(i))
        | 0x43 => map!(le_f32, |i| Instr::F32Const(i))
        | 0x44 => map!(le_f64, |i| Instr::F64Const(i))

        | 0x45 => value!(Instr::I32Eqz)
        | 0x46 => value!(Instr::I32Eq)
        | 0x47 => value!(Instr::I32Ne)
        | 0x48 => value!(Instr::I32LtS)
        | 0x49 => value!(Instr::I32LtU)
        | 0x4A => value!(Instr::I32GtS)
        | 0x4B => value!(Instr::I32GtU)
        | 0x4C => value!(Instr::I32LeS)
        | 0x4D => value!(Instr::I32LeU)
        | 0x4E => value!(Instr::I32GeS)
        | 0x4F => value!(Instr::I32GeU)
        | 0x50 => value!(Instr::I64Eqz)
        | 0x51 => value!(Instr::I64Eq)
        | 0x52 => value!(Instr::I64Ne)
        | 0x53 => value!(Instr::I64LtS)
        | 0x54 => value!(Instr::I64LtU)
        | 0x55 => value!(Instr::I64GtS)
        | 0x56 => value!(Instr::I64GtU)
        | 0x57 => value!(Instr::I64LeS)
        | 0x58 => value!(Instr::I64LeU)
        | 0x59 => value!(Instr::I64GeS)
        | 0x5A => value!(Instr::I64GeU)
        | 0x5B => value!(Instr::F32Eq)
        | 0x5C => value!(Instr::F32Ne)
        | 0x5D => value!(Instr::F32Lt)
        | 0x5E => value!(Instr::F32Gt)
        | 0x5F => value!(Instr::F32Le)
        | 0x60 => value!(Instr::F32Ge)
        | 0x61 => value!(Instr::F64Eq)
        | 0x62 => value!(Instr::F64Ne)
        | 0x63 => value!(Instr::F64Lt)
        | 0x64 => value!(Instr::F64Gt)
        | 0x65 => value!(Instr::F64Le)
        | 0x66 => value!(Instr::F64Ge)
        | 0x67 => value!(Instr::I32Clz)
        | 0x68 => value!(Instr::I32Ctz)
        | 0x69 => value!(Instr::I32Popcnt)
        | 0x6A => value!(Instr::I32Add)
        | 0x6B => value!(Instr::I32Sub)
        | 0x6C => value!(Instr::I32Mul)
        | 0x6D => value!(Instr::I32DivS)
        | 0x6E => value!(Instr::I32DivU)
        | 0x6F => value!(Instr::I32RemS)
        | 0x70 => value!(Instr::I32RemU)
        | 0x71 => value!(Instr::I32And)
        | 0x72 => value!(Instr::I32Or)
        | 0x73 => value!(Instr::I32Xor)
        | 0x74 => value!(Instr::I32Shl)
        | 0x75 => value!(Instr::I32ShrS)
        | 0x76 => value!(Instr::I32ShrU)
        | 0x77 => value!(Instr::I32Rotl)
        | 0x78 => value!(Instr::I32Rotr)
        | 0x79 => value!(Instr::I64Clz)
        | 0x7A => value!(Instr::I64Ctz)
        | 0x7B => value!(Instr::I64Popcnt)
        | 0x7C => value!(Instr::I64Add)
        | 0x7D => value!(Instr::I64Sub)
        | 0x7E => value!(Instr::I64Mul)
        | 0x7F => value!(Instr::I64DivS)
        | 0x80 => value!(Instr::I64DivU)
        | 0x81 => value!(Instr::I64RemS)
        | 0x82 => value!(Instr::I64RemU)
        | 0x83 => value!(Instr::I64And)
        | 0x84 => value!(Instr::I64Or)
        | 0x85 => value!(Instr::I64Xor)
        | 0x86 => value!(Instr::I64Shl)
        | 0x87 => value!(Instr::I64ShrS)
        | 0x88 => value!(Instr::I64ShrU)
        | 0x89 => value!(Instr::I64Rotl)
        | 0x8A => value!(Instr::I64Rotr)
        | 0x8B => value!(Instr::F32Abs)
        | 0x8C => value!(Instr::F32Neg)
        | 0x8D => value!(Instr::F32Ceil)
        | 0x8E => value!(Instr::F32Floor)
        | 0x8F => value!(Instr::F32Trunc)
        | 0x90 => value!(Instr::F32Nearest)
        | 0x91 => value!(Instr::F32Sqrt)
        | 0x92 => value!(Instr::F32Add)
        | 0x93 => value!(Instr::F32Sub)
        | 0x94 => value!(Instr::F32Mul)
        | 0x95 => value!(Instr::F32Div)
        | 0x96 => value!(Instr::F32Min)
        | 0x97 => value!(Instr::F32Max)
        | 0x98 => value!(Instr::F32Copysign)
        | 0x99 => value!(Instr::F64Abs)
        | 0x9A => value!(Instr::F64Neg)
        | 0x9B => value!(Instr::F64Ceil)
        | 0x9C => value!(Instr::F64Floor)
        | 0x9D => value!(Instr::F64Trunc)
        | 0x9E => value!(Instr::F64Nearest)
        | 0x9F => value!(Instr::F64Sqrt)
        | 0xA0 => value!(Instr::F64Add)
        | 0xA1 => value!(Instr::F64Sub)
        | 0xA2 => value!(Instr::F64Mul)
        | 0xA3 => value!(Instr::F64Div)
        | 0xA4 => value!(Instr::F64Min)
        | 0xA5 => value!(Instr::F64Max)
        | 0xA6 => value!(Instr::F64Copysign)
        | 0xA7 => value!(Instr::I32WrapI64)
        | 0xA8 => value!(Instr::I32TruncF32S)
        | 0xA9 => value!(Instr::I32TruncF32U)
        | 0xAA => value!(Instr::I32TruncF64S)
        | 0xAB => value!(Instr::I32TruncF64U)
        | 0xAC => value!(Instr::I64ExtendI32S)
        | 0xAD => value!(Instr::I64ExtendI32U)
        | 0xAE => value!(Instr::I64TruncF32S)
        | 0xAF => value!(Instr::I64TruncF32U)
        | 0xB0 => value!(Instr::I64TruncF64S)
        | 0xB1 => value!(Instr::I64TruncF64U)
        | 0xB2 => value!(Instr::F32ConvertI32S)
        | 0xB3 => value!(Instr::F32ConvertI32U)
        | 0xB4 => value!(Instr::F32ConvertI64S)
        | 0xB5 => value!(Instr::F32ConvertI64U)
        | 0xB6 => value!(Instr::F32DemoteF64)
        | 0xB7 => value!(Instr::F64ConvertI32S)
        | 0xB8 => value!(Instr::F64ConvertI32U)
        | 0xB9 => value!(Instr::F64ConvertI64S)
        | 0xBA => value!(Instr::F64ConvertI64U)
        | 0xBB => value!(Instr::F64PromoteF32)
        | 0xBC => value!(Instr::I32ReinterpretF32)
        | 0xBD => value!(Instr::I64ReinterpretF64)
        | 0xBE => value!(Instr::F32ReinterpretI32)
        | 0xBF => value!(Instr::F64ReinterpretI64)
    )
);


/**
 * take next byte to see.
 * If is End (0x0B), return the vec including content so far.
 * 0x0B is eaten but not included in the vec.
 * Else, parse as an Instr.
 */
named_args!(
    instrs_till(end: u8)<Vec<Instr>>,
    map!(many_till!(instr, call!(tag_byte, end)), |t| t.0)
);

named!(expr<Vec<Instr>>, call!(instrs_till, 0x0b));

named!(
    global<Global>,
    do_parse!(
        type_: globaltype >> init: map!(expr, |instrs| Expr { instrs }) >> (Global { type_, init })
    )
);

named!(
    code<Code>,
    do_parse!(
        le_u32 >> // length of binary, ignored
        locals: length_count!(le_usize, valtype) >>
        instrs: expr >>
        (Code {locals, body: Expr {instrs}})
    )
);

named!(
    elem<Elem>,
    do_parse!(
        call!(tag_byte, 0x00) >> // table idx
        instrs: expr >>
        init: length_count!(le_usize, le_usize) >>
        (Elem {offset: Expr {instrs}, init})
    )
);

named!(
    section_custom<()>,
    do_parse!(call!(tag_byte, 0x00) >> size: le_usize >> take!(size) >> (()))
);

named!(
    section_type<Vec<Type>>,
    do_parse!(call!(tag_byte, 0x01) >> types: length_count!(le_usize, functype) >> (types))
);


//TODO
named!(
    section_import<()>,
    do_parse!(call!(tag_byte, 0x02) >> size: le_usize >> take!(size) >> (()))
);


//Function section contains only typeIdxes
named!(
    section_func<Vec<usize>>,
    do_parse!(call!(tag_byte, 0x03) >> funcs: length_count!(le_usize, le_usize) >> (funcs))
);

//TODO
named!(
    section_table<Option<Table>>,
    do_parse!(
        call!(tag_byte, 0x04) >>
        opt_table: call!(zero_or_one, tabletype) >>
        (opt_table))
);

// expects 0 or 1 Mem in a vector.
named!(
    section_mem<Option<Mem>>,
    do_parse!(
        call!(tag_byte, 0x05)
            >> mem: call!(zero_or_one, memtype) >>
            (mem)
    )
);

named!(
    section_global<Vec<Global>>,
    do_parse!(call!(tag_byte, 0x06) >> globals: length_count!(le_usize, global) >> (globals))
);


//TODO
named!(
    section_export<()>,
    do_parse!(call!(tag_byte, 0x07) >> size: le_usize >> take!(size) >> (()))
);

//TODO
named!(
    section_start<()>,
    do_parse!(call!(tag_byte, 0x08) >> size: le_usize >> take!(size) >> (()))
);

//TODO
named!(
    section_elem<Vec<Elem>>,
    do_parse!(
        call!(tag_byte, 0x09) >> 
        elems: length_count!(le_usize, elem) >>
        (elems)
    )
);

named!(
    section_code<Vec<Code>>,
    do_parse!(call!(tag_byte, 0x0a) >> code: length_count!(le_usize, code) >> (code))
);

//TODO
named!(
    section_data<()>,
    do_parse!(call!(tag_byte, 0x0b) >> size: le_usize >> take!(size) >> (()))
);

pub fn module(i: &[u8]) -> IResult<&[u8], Module> {
    let (i, _) = tag(&[0x00, 0x61, 0x73, 0x6D])(i)?; //magic
    let (i, _) = tag(&[0x01, 0x00, 0x00, 0x00])(i)?; //version
    let (i, _) = opt(section_custom)(i)?;
    let (i, types) = opt(section_type)(i)?;
    let (i, _) = opt(section_custom)(i)?;
    let (i, imports) = opt(section_import)(i)?;
    let (i, _) = opt(section_custom)(i)?;
    let (i, functypes) = opt(section_func)(i)?;
    let (i, _) = opt(section_custom)(i)?;
    let (i, table) = opt(section_table)(i)?;
    let (i, _) = opt(section_custom)(i)?;
    let (i, mem) = opt(section_mem)(i)?;
    let (i, _) = opt(section_custom)(i)?;
    let (i, globals) = opt(section_global)(i)?;
    let (i, _) = opt(section_custom)(i)?;
    //export
    //start
    let (i, elems) = opt(section_elem)(i)?;
    let (i, _) = opt(section_custom)(i)?;
    let (i, codes) = opt(section_code)(i)?;
    let (i, _) = opt(section_custom)(i)?;
    let (i, data) = opt(section_data)(i)?;
    let (i, _) = opt(section_custom)(i)?;
    let funcs =
        zip_funcs((functypes, codes)).ok_or_else(|| nom::Err::Error((i, ErrorKind::MapOpt)))?;

    Ok((
        i,
        Module {
            types: types.unwrap_or_else(|| vec![]),
            funcs,
            table: flatten(table),
            mem: flatten(mem),
            globals: globals.unwrap_or_else(|| vec![]),
            elems: elems.unwrap_or_else(|| vec![]),
        },
    ))
}
