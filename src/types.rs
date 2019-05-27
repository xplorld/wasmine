/**
 Types defined in WASM.
 https://webassembly.github.io/spec/core/syntax/values.html

 */

pub type Idx = usize;
pub type Byte = u8;

pub struct Module {
    types: Vec<FunctionType>,
    funcs: Vec<Function>,
    tables: Vec<Table>, // only have one for now
    mems: Vec<Mem>,
    globals: Vec<Global>,
    elem: Vec<Elem>,
    data: Vec<Data>,
    start: Option<Start>,
    imports: Vec<Import>,
    exports: Vec<Export>,
}

#[derive(Debug)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

// TODO: Val should have a handwritten PartialEq
// to compare the content of the same variant
#[derive(Debug)]
#[derive(PartialEq)]
pub enum Val {
    I32 { i: u32 },
    F32 { f: f32 },
    I64 { i: u64 },
    F64 { f: f64 },
}

#[derive(Debug)]
pub struct FunctionType {
    // args could have 0 or more arguments
    args: Vec<ValType>,
    // In the current version of WebAssembly, at most one value is allowed as a result. However, this may be generalized to sequences of values in future versions.
    // https://webassembly.github.io/spec/core/syntax/types.html#syntax-functype
    ret: ValType,
}

//TODO: do we need a enum of Function and FuncRef?
#[derive(Debug)]
pub struct FuncRef {}

#[derive(Debug)]
pub struct Function {
    type_: Idx,
    locals: Vec<ValType>,
    body: Expr,
}

#[derive(Debug)]
pub struct Table {
    limits: Limits,
    elemtype: FuncRef,
}

#[derive(Debug)]
pub struct Limits {
    min: u32,
    max: Option<u32>,
}

#[derive(Debug)]
pub struct Mem {
    limits: Limits,
}

#[derive(Debug)]
pub struct Global {
    type_: GlobalType,
    init: Expr,
}

#[derive(Debug)]
pub struct GlobalType {
    mut_: Mut,
    type_: ValType,
}

#[derive(Debug)]
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

/**
 * In spec, expr ::= instr* end,
 * but end is a marker but not an instr, so we do not have
 * to represent it in struct Expr.
 */
#[derive(Debug)]
pub struct Expr {
    instrs: Vec<Instr>,
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

#[derive(Debug)]
#[derive(PartialEq)]
pub struct Label {
    pub arity: u32, // 0 or 1
    pub continuation: Idx,
}

#[derive(Debug)]
pub enum Instr {
    /* numeric instrs */
    I32Const(u32),
    F32Const(f32),
    I64Const(u64),
    F64Const(f64),
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
    IfElse {not_taken: Idx, label: Label},
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
        type_: FunctionType,
        /* In runtime, we always have reference to the module
        so we do not have to keep a reference here. */
        // module: ModuleInst,
        code: Function,
    },
    Host {
        type_: FunctionType,
        hostcode: WasmineHostFunction,
    },
}



struct Store {
    funcs: FuncInst,
    // tables: TableInst,
    // mems: MemInst,
    // globals: GlobalInst,
}
