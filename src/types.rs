/**
 Types defined in WASM.
 https://webassembly.github.io/spec/core/syntax/values.html

 */
#[derive(Debug)]
pub enum Val {
    I32 { i: i32 },
    I64 { i: i64 },
    F32 { f: f32 },
    F64 { f: f64 },
}

#[derive(Debug)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug)]
pub struct FunctionType {
    // args could have 0 or more arguments
    args: Vec<ValType>,
    // In the current version of WebAssembly, at most one value is allowed as a result. However, this may be generalized to sequences of values in future versions.
    // https://webassembly.github.io/spec/core/syntax/types.html#syntax-functype
    ret: ValType,
}

