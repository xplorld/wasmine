
use instrs::{Instr, ParseInstr};

use nom::bytes::complete::tag;
use nom::error::ErrorKind;
use nom::number::complete::{le_u32, le_u8};
use types::*;
use nom::{alt, count, map_opt, named, take, IResult};
use std::str::from_utf8;


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
        0x7f => value!(ValType::I32) |
        0x7e => value!(ValType::I64) |
        0x7d => value!(ValType::F32) |
        0x7c => value!(ValType::F64)
    )
);

#[rustfmt::skip]
named!(
    blocktype<Option<ValType>>,
    alt!(
        value!(None, call!(tag_byte, 0x40)) | 
        opt!(valtype))
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
        0x00 => map!(le_usize, |min| Limits {min, max : None}) |
        0x01 => do_parse!(
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
        call!(tag_byte, 0x70)
            >> limits: limits
            >> (Table {
                limits,
                elemtype: FuncRef {}
            })
    )
);

named!(
    mut_<Mut>,
    switch!(le_u8,
        0x00 => value!(Mut::Const) |
        0x01 => value!(Mut::Var)
    )
);

named!(
    globaltype<GlobalType>,
    do_parse!(vt: valtype >> m: mut_ >> (GlobalType { mut_: m, type_: vt }))
);


// during parsing, we could not find continuations for Labels. They are assigned
// usize::MAX (which is appearently illegal) and wait for an operation of
// validation-and-assign, namely Expr::try_from.
named!(
    instr<ParseInstr>,
    switch!(le_u8,
        0x00 => value!(ParseInstr::Instr(Instr::Unreachable)) |
        0x01 => value!(ParseInstr::Instr(Instr::Nop)) |
        0x02 => do_parse!(
            type_: blocktype >>
            instrs: call!(instrs_till, 0x0b) >>
            (ParseInstr::Block {type_, instrs})
        ) |
        0x03 => do_parse!(
            type_: blocktype >>
            instrs: call!(instrs_till, 0x0b) >>
            (ParseInstr::Loop {type_, instrs})
        ) |
        0x04 => alt!(
            do_parse!(
                type_: blocktype >>
                then: call!(instrs_till, 0x05) >>
                else_: call!(instrs_till, 0x0b) >>
                (ParseInstr::IfElse{type_, then, else_})
            ) |
            do_parse!(
                type_: blocktype >>
                then: call!(instrs_till, 0x0b) >>
                (ParseInstr::If {type_, then})
            )
        ) |
        0x0c => map!(le_usize, |idx| ParseInstr::Instr(Instr::Br(idx)))
    )
);


/**
 * take next byte to see.
 * If is End (0x0B), return the vec including content so far.
 * 0x0B is eaten but not included in the vec.
 * Else, parse as an ParseInstr.
 */
named_args!(
    instrs_till(end: u8)<Vec<ParseInstr>>,
    map!(many_till!(instr, call!(tag_byte, end)), |t| t.0)
);
