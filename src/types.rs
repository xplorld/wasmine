/**
Types defined in WASM.
https://webassembly.github.io/spec/core/syntax/values.html

*/

use crate::instrs::Instr;
pub use crate::val::*;


pub type Idx = usize;
pub type Byte = u8;

// 64 KB
pub const PAGE_SIZE: usize = 65535;

#[derive(Debug)]
pub struct Module {
    pub types: Vec<Type>,
    pub funcs: Vec<Function>,
    pub table: Option<Table>,
    pub mem: Option<Mem>,
    pub globals: Vec<Global>,
    pub elems: Vec<Elem>,
    // data: Vec<Data>,
    // start: Option<Start>,
    // imports: Vec<Import>,
    // exports: Vec<Export>,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type {
    // args could have 0 or more arguments
    pub args: Vec<ValType>,
    // In the current version of WebAssembly, at most one value is allowed as a result. However, this may be generalized to sequences of values in future versions.
    // https://webassembly.github.io/spec/core/syntax/types.html#syntax-functype
    pub ret: Option<ValType>,
}

impl Type {
    pub fn arity(&self) -> usize {
        match self.ret {
            Some(_) => 1,
            None => 0,
        }
    }
}

#[derive(Debug)]
pub struct Code {
    pub locals: Vec<ValType>,
    pub body: Expr,
}

#[derive(Debug)]
pub struct Function {
    pub type_: Idx,
    pub code: Code,
}


#[derive(Debug)]
pub struct Table {
    pub limits: Limits,
    // only elemtype available is FuncRef
    // pub elemtype: FuncRef,
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub min: usize,
    pub max: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct Mem {
    pub limits: Limits,
}

impl Mem {
    pub fn empty() -> Self {
        Mem {
            limits: Limits {
                min: 0,
                max: Some(0),
            },
        }
    }
}

#[derive(Debug)]
pub struct Global {
    pub type_: GlobalType,
    pub init: Expr,
}

#[derive(Debug, Copy, Clone)]
pub struct GlobalType {
    pub mut_: Mut,
    pub type_: ValType,
}

#[derive(Debug, Copy, Clone)]
pub enum Mut {
    Const,
    Var,
}

#[derive(Debug)]
pub struct Elem {
    // table is always 0
    // table: Idx,
    pub offset: Expr,
    pub init: Vec<Idx>, // index of function
}

#[derive(Debug)]
pub struct Data {
    data: Idx, // index of Mem
    offset: Expr,
    init: Vec<Byte>,
}

#[derive(Debug)]
pub struct Start {
    func: Idx, // index of function
}

#[derive(Debug)]
pub struct Export {
    name: String,
    desc: ExportDesc,
}

#[derive(Debug)]
pub enum ExportDesc {
    Func { i: Idx },
    Table { i: Idx },
    Mem { i: Idx },
    Global { i: Idx },
}

#[derive(Debug)]
pub struct Import {
    module: String,
    name: String,
    desc: ImportDesc,
}

#[derive(Debug)]
pub enum ImportDesc {
    Func { type_: Idx },
    Table { t: Table },
    Mem { m: Mem },
    Global { g: GlobalType },
}

#[derive(Debug)]
pub struct Expr {
    pub instrs: Vec<Instr>,
}

impl Expr {
    /**
     * If `self` is a constant, evaluate it and return the result.
     * Else, return `None`.
     * 
     * According to spec, a constant expression can either be a t.const c,
     * or a global.get whose index points to an imported global.
     * 
     * TODO: support 
     *
     */
    pub fn const_val(&self) -> Option<Val> {
        if self.instrs.len() == 1 {
            match self.instrs.first() {
                Some(Instr::I32Const(val)) => Some(Val::I32(*val)),
                Some(Instr::F32Const(val)) => Some(Val::F32(*val)),
                Some(Instr::I64Const(val)) => Some(Val::I64(*val)),
                Some(Instr::F64Const(val)) => Some(Val::F64(*val)),
                _ => None,
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct Memarg {
    pub offset: u32,
    pub align: u32,
}

#[derive(Debug)]
pub struct BrTableArgs {
    pub table: Vec<Idx>,
    pub label: Idx,
}


/**
 * A block of control flow.
 */
#[derive(Debug)]
pub struct Block {
    pub type_: Option<ValType>, // 0 or 1
    pub expr: Expr,
}

impl Block {
    pub fn arity(&self) -> usize {
        match self.type_ {
            Some(_) => 1,
            None => 0,
        }
    }
}

/**
 * Runtime Structures
 */

pub struct WasmineHostFunction {}
