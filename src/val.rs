
use crate::runtime::Trap;

use byteorder::{ByteOrder, LittleEndian};
use std::convert::TryFrom;
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
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

impl From<u32> for Val {
    fn from(i: u32) -> Val {
        Val::I32(i)
    }
}

impl TryFrom<Val> for u32 {
    type Error = Trap;
    fn try_from(value: Val) -> Result<u32, Trap> {
        match value {
            Val::I32(i) => Ok(i),
            _ => Err(Trap {}),
        }
    }
}

impl From<u64> for Val {
    fn from(i: u64) -> Val {
        Val::I64(i)
    }
}

impl TryFrom<Val> for u64 {
    type Error = Trap;
    fn try_from(value: Val) -> Result<u64, Trap> {
        match value {
            Val::I64(i) => Ok(i),
            _ => Err(Trap {}),
        }
    }
}

impl From<f32> for Val {
    fn from(i: f32) -> Val {
        Val::F32(i)
    }
}

impl TryFrom<Val> for f32 {
    type Error = Trap;
    fn try_from(value: Val) -> Result<f32, Trap> {
        match value {
            Val::F32(i) => Ok(i),
            _ => Err(Trap {}),
        }
    }
}

impl From<f64> for Val {
    fn from(i: f64) -> Val {
        Val::F64(i)
    }
}

impl TryFrom<Val> for f64 {
    type Error = Trap;
    fn try_from(value: Val) -> Result<f64, Trap> {
        match value {
            Val::F64(i) => Ok(i),
            _ => Err(Trap {}),
        }
    }
}

//TODO: when trait alias is stablized, use it
pub trait RawVal: Copy + Into<Val> + TryFrom<Val, Error = Trap> {
}

impl RawVal for u32 {}
impl RawVal for u64 {}
impl RawVal for f32 {}
impl RawVal for f64 {}

pub trait FromSlice {
   fn from_slice(s: &[u8]) -> Self; 
}

// cast rules: https://doc.rust-lang.org/nomicon/casts.html
impl FromSlice for u8 {     fn from_slice(s: &[u8]) -> Self {          s[0]     }      }
impl FromSlice for i8 {     fn from_slice(s: &[u8]) -> Self {          s[0] as i8     }      }
impl FromSlice for u16 {     fn from_slice(s: &[u8]) -> Self {          LittleEndian::read_u16(s)     }      }
impl FromSlice for i16 {     fn from_slice(s: &[u8]) -> Self {          LittleEndian::read_i16(s)     }      }
impl FromSlice for u32 {     fn from_slice(s: &[u8]) -> Self {          LittleEndian::read_u32(s)     }      }
impl FromSlice for i32 {     fn from_slice(s: &[u8]) -> Self {          LittleEndian::read_i32(s)     }      }
impl FromSlice for u64 {     fn from_slice(s: &[u8]) -> Self {          LittleEndian::read_u64(s)     }      }
impl FromSlice for f32 {     fn from_slice(s: &[u8]) -> Self {          LittleEndian::read_f32(s)     }      }
impl FromSlice for f64 {     fn from_slice(s: &[u8]) -> Self {          LittleEndian::read_f64(s)     }      }

pub trait ToSlice {
     fn to_slice(src: Self, dst: &mut [u8]);
}

impl ToSlice for u8 {fn to_slice(src: Self, dst: &mut [u8]) {         dst[0] = src;     } }
impl ToSlice for u16 {fn to_slice(src: Self, dst: &mut [u8]) {         LittleEndian::write_u16(dst, src)     } }
impl ToSlice for u32 {fn to_slice(src: Self, dst: &mut [u8]) {         LittleEndian::write_u32(dst, src)     } }
impl ToSlice for u64 {fn to_slice(src: Self, dst: &mut [u8]) {         LittleEndian::write_u64(dst, src)     } }
impl ToSlice for f32 {fn to_slice(src: Self, dst: &mut [u8]) {         LittleEndian::write_f32(dst, src)     } }
impl ToSlice for f64 {fn to_slice(src: Self, dst: &mut [u8]) {         LittleEndian::write_f64(dst, src)     } }
