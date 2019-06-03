// Instructions, and their implementations.


use runtime::Trap;
use types::*;


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