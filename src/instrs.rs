// Instructions, and their implementations.

use types::*;
use runtime::Trap;

pub fn i32unop<F>(val: &Val, op: F) -> Result<Val, Trap> 
    where F: Fn(u32) -> u32 {
    Ok(Val::I32(op(val.as_i32()?)))
}

pub fn i64binop<F>(val1: &Val, val2: &Val, op: F) -> Result<Val, Trap> 
    where F: Fn(u64, u64) -> u64 {
       Ok(Val::I64(op(val1.as_i64()?, val2.as_i64()?))) 
}


/**
 * Instructions.
 * 
 * One principle is to keep as much parsing time info as possible, so that
 * instead of a `Const(Val)` we have four `I32Const(u32)`s. In the case of
 * `Const` we did not make much difference, but in `i.unop` we save one tag
 * inspection.
 * 
 * This however introduces more instr cases and more boilercode. To cope with
 * that we use macros to save some effort.
 */
#[derive(Debug)]
pub enum Instr {
    /* numeric instrs */
    I32Const(u32),
    I64Const(u64),
    F32Const(f32),
    F64Const(f64),
    // unop
    I32Clz,
    // biop
        I64Sub,
    I64Mul,

    I64Eq,

    /* more numeric instrs */
    /* parametric instrs */
    Drop,
    Select,
    /* variable instructions */
    LocalGet(Idx),
    LocalSet(Idx),
    LocalTee(Idx),
    GlobalGet(Idx),
    GlobalSet(Idx),
    /* memory instructions */
    I32Load(Memarg),
    I32Store(Memarg),
    /* more mem instrs */
    /* ctrl instrs */
    Nop,
    Unreachable,
    // Block, Loop are reduced to Label
    Label(Label),
    If { not_taken: Idx, label: Label },
    Else,
    End,
    Br(Idx),
    BrIf(Idx),
    BrTable(BrTableArgs),
    Return,
    Call(Idx),
    CallIndirect(Idx),
}