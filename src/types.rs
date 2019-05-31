/**
Types defined in WASM.
https://webassembly.github.io/spec/core/syntax/values.html

*/

pub type Idx = usize;
pub type Byte = u8;

// 64 KB
pub const PAGE_SIZE: usize = 65535;

#[derive(Debug)]
pub struct Module {
    pub types: Vec<Type>,
    pub funcs: Vec<Function>,
    // tables: Vec<Table>, // only have one for now
    // have one and only one mem
    // if there are none, we instead have a Mem with zero min and max
    pub mems: Mem,
    pub globals: Vec<Global>,
    // elem: Vec<Elem>,
    // data: Vec<Data>,
    // start: Option<Start>,
    // imports: Vec<Import>,
    // exports: Vec<Export>,
}


#[derive(Debug, PartialEq, Copy, Clone)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

// TODO: Val should have a handwritten PartialEq
// to compare the content of the same variant
#[derive(Debug, Clone, Copy)]
pub enum Val {
    I32(u32),
    I64(u64),
    F32(f32),
    F64(f64),
}

impl Val {
    pub fn matches(&self, type_: &ValType) -> bool {
        match (self, type_) {
            (Val::I32(_), ValType::I32)
            | (Val::F32(_), ValType::F32)
            | (Val::I64(_), ValType::I64)
            | (Val::F64(_), ValType::F64) => true,
            _ => false,
        }
    }

}

impl From<&ValType> for Val {
    fn from(item: &ValType) -> Val {
        match item {
            ValType::I32 => Val::I32(0),
            ValType::I64 => Val::I64(0),
            ValType::F32 => Val::F32(0.0),
            ValType::F64 => Val::F64(0.0),
        }
    }
}

#[derive(Debug, Clone)]
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

//TODO: do we need a enum of Function and FuncRef?
#[derive(Debug)]
pub struct FuncRef {}

#[derive(Debug)]
pub struct Function {
    pub type_: Type,
    pub locals: Vec<ValType>,
    pub body: Expr,
}

impl Function {
    pub fn new_locals(&self) -> Vec<Val> {
        self.locals.iter().map(Val::from).collect()
    }
}

#[derive(Debug)]
pub struct Table {
    limits: Limits,
    elemtype: FuncRef,
}

#[derive(Debug)]
pub struct Limits {
    pub min: usize,
    pub max: Option<usize>,
}

#[derive(Debug)]
pub struct Mem {
    pub limits: Limits,
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
    table: Idx,
    offset: Expr,
    init: Vec<Idx>, // index of function
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
     * Currently only support returning one value.
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
    table: Vec<Idx>,
    label: Idx,
}

#[derive(Debug, Clone, Copy)]
pub struct Label {
    pub arity: usize, // 0 or 1
    pub continuation: Idx,
}


#[derive(Debug)]
pub enum Instr {
    /* numeric instrs */
    I32Const(u32),
    F32Const(f32),
    I64Const(u64),
    F64Const(f64),
    I64Eq,
    I64Sub,
    I64Mul,

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

/**
 * Runtime Structures
 */

pub struct WasmineHostFunction {}

pub enum FuncInst {
    Wasm {
        type_: Type,
        code: Function,
    },
    Host {
        type_: Type,
        hostcode: WasmineHostFunction,
    },
}


struct Store {
    funcs: FuncInst,
    // tables: TableInst,
    // mems: MemInst,
    // globals: GlobalInst,
}
