// Instructions, and their implementations.
use crate::types::*;


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

    //iunop
    I32Clz,
    I64Clz,
    I32Ctz,
    I64Ctz,
    I32Popcnt,
    I64Popcnt,
    //ibinop
    I32Add,
    I64Add,
    I32Sub,
    I64Sub,
    I32Mul,
    I64Mul,
    I32Divu,
    I64Divu,
    I32Divs,
    I64Divs,
    I32Remu,
    I64Remu,
    I32Rems,
    I64Rems,
    I32And,
    I64And,
    I32Or,
    I64Or,
    I32Xor,
    I64Xor,
    I32Shl,
    I64Shl,
    I32Shru,
    I64Shru,
    I32Shrs,
    I64Shrs,
    I32Rotl,
    I64Rotl,
    I32Rotr,
    I64Rotr,
    //funop
    F32Abs,
    F64Abs,
    F32Neg,
    F64Neg,
    F32Sqrt,
    F64Sqrt,
    F32Ceil,
    F64Ceil,
    F32Floor,
    F64Floor,
    F32Trunc,
    F64Trunc,
    F32Nearest,
    F64Nearest,
    //fbinop
    F32Add,
    F64Add,
    F32Sub,
    F64Sub,
    F32Mul,
    F64Mul,
    F32Div,
    F64Div,
    F32Min,
    F64Min,
    F32Max,
    F64Max,
    F32Copysign,
    F64Copysign,
    //itestop
    I32Eqz,
    I64Eqz,
    //irelop
    I32Eq,
    I64Eq,
    I32Ne,
    I64Ne,
    I32Ltu,
    I64Ltu,
    I32Lts,
    I64Lts,
    I32Gtu,
    I64Gtu,
    I32Gts,
    I64Gts,
    I32Leu,
    I64Leu,
    I32Les,
    I64Les,
    I32Geu,
    I64Geu,
    I32Ges,
    I64Ges,
    //frelop
    F32Eq,
    F64Eq,
    F32Ne,
    F64Ne,
    F32Lt,
    F64Lt,
    F32Gt,
    F64Gt,
    F32Le,
    F64Le,
    F32Ge,
    F64Ge,

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
    Block(Block),
    IfElse { then: Block, else_: Option<Block> },
    Br(Idx),
    BrIf(Idx),
    BrTable(BrTableArgs),
    Return,
    Call(Idx),
    CallIndirect(Idx),
}
