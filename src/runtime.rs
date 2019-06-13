//TODO: make use of FuncInst in rt
use crate::instrs::*;
use crate::types::*;
use crate::util::AsPrimitive;
use std::convert::TryInto;
use std::error::Error;
use std::fmt;

use std::mem;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Shl, Shr, Sub};

#[derive(Debug, Clone, Copy)]
pub struct Trap;

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "it traps!")
    }
}

impl Error for Trap {}

fn assert_or_trap(cond: bool) -> Result<(), Trap> {
    if cond {
        Ok(())
    } else {
        Err(Trap {})
    }
}

pub type InvocationResult = Result<Option<Val>, Trap>;

/**
 * Successful result of one step.
 *
 * Finish may happen on `return` or end of execution.
 * End of execution <=> there is only 1 label in the valstack && the label's pc
 * equals to the label's instrs' len.
 */
enum StepNormalResult {
    Continue,
    Finish(Option<Val>),
}

type StepResult = Result<StepNormalResult, Trap>;


#[derive(Debug)]
struct MemInst {
    data: Vec<u8>,
    max: Option<usize>,
}

impl MemInst {
    fn new(mem: &Mem) -> MemInst {
        MemInst {
            data: vec![0; mem.limits.min * PAGE_SIZE],
            max: mem.limits.max,
        }
    }
}


#[derive(Debug)]
pub struct GlobalInst {
    mut_: Mut,
    val: Val,
}

impl GlobalInst {
    fn new(global: &Global) -> GlobalInst {
        GlobalInst {
            mut_: global.type_.mut_,
            val: global.init.const_val().unwrap(),
        }
    }
}

//TODO: currently only native functions
// need support imported functions
#[derive(Debug, Clone, Copy)]
pub struct FuncInst<'a> {
    type_: &'a Type,
    code: &'a Code,
}

impl<'a> FuncInst<'a> {
    pub fn new_locals(&'a self) -> Vec<Val> {
        self.code.locals.iter().map(Val::from).collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LabelCont {
    Finish,
    Loop,
}

/**
 * Label: instrs, continuation, arity, (pc)
 */
#[derive(Debug, Clone, Copy)]
struct Label<'a> {
    pc: Idx,
    expr: &'a Expr,
    continuation: LabelCont,
    // idx of first val in valstack of this frame
    // always <= valstack.len()
    val_idx: Idx,
}


#[derive(Debug)]
struct ValStack {
    stack: Vec<Val>,
}

impl ValStack {

    fn len(&self) -> usize {
        self.stack.len()
    }
    //TODO: refine it
    fn pop(&mut self) -> Result<Val, Trap> {
        match self.stack.pop() {
            Some(val) => Ok(val),
            _ => Err(Trap {}),
        }
    }

    fn tee(&self) -> Result<Val, Trap> {
        match self.stack.last() {
            Some(val) => Ok(*val),
            _ => Err(Trap {}),
        }
    }
    fn push(&mut self, val: Val) {
        self.stack.push(val);
    }

    fn return_value(&self, arity: usize) -> StepResult {
        use self::StepNormalResult::*;
        if arity == 0 {
            Ok(Finish(None))
        } else {
            assert!(arity == 1);
            // unwrap and wrap == assert
            Ok(Finish(Some(self.tee().unwrap())))
        }
    }

    fn unop<F, T>(&mut self, op: F) -> Result<(), Trap>
    where
        F: Fn(T) -> T,
        T: RawVal,
    {
        let val: T = self.pop()?.try_into()?;
        Ok(self.push(op(val).into()))
    }

    fn binop<F, T>(&mut self, op: F) -> Result<(), Trap>
    where
        F: Fn(T, T) -> T,
        T: RawVal,
    {
        let val2 = self.pop()?.try_into()?;
        let val1 = self.pop()?.try_into()?;
        Ok(self.push(op(val1, val2).into()))
    }

    fn binop_partial<T, U, F>(&mut self, op: F) -> Result<(), Trap>
    where
        F: Fn(U, U) -> Option<U>,
        T: RawVal + AsPrimitive<U>,
        U: Copy + AsPrimitive<T>,
    {
        let val2: T = self.pop()?.try_into()?;
        let val1: T = self.pop()?.try_into()?;
        Ok(self.push(op(val1.as_(), val2.as_()).ok_or(Trap {})?.as_().into()))
    }


    fn testop<F, T>(&mut self, op: F) -> Result<(), Trap>
    where
        F: Fn(T) -> bool,
        T: RawVal,
    {
        let val1 = self.pop()?.try_into()?;
        Ok(self.push((op(val1) as u32).into()))
    }

    fn relop<T, U, F>(&mut self, op: F) -> Result<(), Trap>
    where
        F: Fn(&U, &U) -> bool,
        T: RawVal + AsPrimitive<U>,
        U: Copy,
    {
        let val2: T = self.pop()?.try_into()?;
        let val1: T = self.pop()?.try_into()?;
        Ok(self.push((op(&val1.as_(), &val2.as_()) as u32).into()))
    }

    fn cvtop<T, U, F>(&mut self, op: F) -> Result<(), Trap>
    where
        F: Fn(T) -> U,
        T: RawVal + AsPrimitive<U>,
        U: RawVal,
    {
        let val: T = self.pop()?.try_into()?;
        Ok(self.push(op(val).into()))
    }
}

/**
 * Local Runtime of an invocation.
 *
 * A frame corresponds with a function invocation,
 * and consists of an arity and a set of locals.
 *
 *
 */
#[derive(Debug)]
pub struct Frame<'a> {
    valstack: ValStack,
    labelstack: Vec<Label<'a>>,
    arity: usize, // 0 or 1
    locals: Vec<Val>,
}

impl<'a> Frame<'a> {
    fn enter(&mut self, block: &'a Block, cont: LabelCont) {
        let label = Label {
            pc: 0,
            expr: &block.expr,
            continuation: cont,
            val_idx: self.valstack.len(),
        };
        self.labelstack.push(label);
    }


    fn br(&mut self, idx: usize) -> StepResult {
        use self::StepNormalResult::*;

        // as validated, cannot overflow
        let next_to_target = self.labelstack.len() - idx;
        let next_val_idx = self.labelstack[next_to_target].val_idx;
        self.labelstack.truncate(next_to_target);
        self.valstack.stack.truncate(next_val_idx);
        // now len(label) == next_to_target, so that last label *is* target
        Ok(Continue)
    }

}


#[derive(Debug)]
pub struct Runtime<'a> {
    funcs: Vec<FuncInst<'a>>,
    mem: MemInst,
    globals: Vec<GlobalInst>,
}


impl<'a> Runtime<'a> {

    fn instantiate_functions(module: &'a Module) -> Result<Vec<FuncInst<'a>>, Trap> {
        module
            .funcs
            .iter()
            .map(|f| {
                Ok(FuncInst {
                    type_: module.types.get(f.type_).ok_or(Trap {})?,
                    code: &f.code,
                })
            })
            .collect::<Result<Vec<FuncInst<'a>>, Trap>>()
    }

    fn instantiate(module: &'a Module) -> Result<Runtime<'a>, Trap> {
        let rt = Runtime {
            mem: MemInst::new(&module.mem),
            globals: module.globals.iter().map(GlobalInst::new).collect(),
            funcs: Runtime::instantiate_functions(module)?,
        };
        Ok(rt)
    }

    /**
     *
     * Load a value of type U from mem into stack as type T.
     *
     * where
     *      either T=U or
     *  T is unsigned integer +
     *  U is integer +
     *  sizeof(T) >= sizeof(U)
     * So that we could cast U to T with sign-extend (via `as` operator)
     *
     * e.g. i32.load_8s, shall be load<u32, i8>
     *
     */
    fn load<T, U>(&mut self, frame: &mut Frame, memarg: &Memarg) -> Result<(), Trap>
    where
        T: RawVal + Copy,
        U: FromSlice + AsPrimitive<T>,
    {
        let base: u32 = frame.valstack.pop()?.try_into()?;
        let eff: usize = (base + memarg.offset).try_into().unwrap();
        let tail = eff + mem::size_of::<U>();
        let data = &self.mem.data;
        assert_or_trap(data.len() < tail)?;
        let val: T = U::from_slice(&self.mem.data[eff..tail]).as_();
        frame.valstack.push(val.into());
        Ok(())
    }

    /**
     * Read a val from the stack of type T,
     * Wrap (modulo) it into unsigned type U,
     * and store into the mem.
     *
     * where
     *      either T=U or
     *      T: unsigned int
     *      U: unsigned int
     *      sizeof(T) >= sizeof(U)
     */
    fn store<T, U>(&mut self, frame: &mut Frame, memarg: &Memarg) -> Result<(), Trap>
    where
        T: RawVal + AsPrimitive<U>,
        U: Copy + ToSlice,
    {
        let base: u32 = frame.valstack.pop()?.try_into()?;
        let val: T = frame.valstack.pop()?.try_into()?;
        let val: U = val.as_();

        let eff = (base + memarg.offset) as usize;
        let tail = eff + mem::size_of::<U>();
        let data = &mut self.mem.data;
        assert_or_trap(data.len() >= tail)?;
        U::to_slice(val, &mut data[eff..tail]);
        Ok(())
    }

    fn new_frame<'b>(&self, func: FuncInst<'b>, args: &[Val]) -> Frame<'b>
    where
        'a: 'b,
    {
        let mut frame = Frame {
            labelstack: vec![Label {
                pc: 0,
                expr: &func.code.body,
                continuation: LabelCont::Finish,
                val_idx: 0,
            }],
            valstack: ValStack { stack: vec![] },
            arity: func.type_.arity(),
            locals: func.new_locals(), //all zero
        };

        let len = args.len();
        assert!(len == func.type_.args.len());
        frame.locals[0..len].copy_from_slice(args);

        frame
    }
    pub fn invoke<'b>(&mut self, func: FuncInst<'b>, args: &[Val]) -> InvocationResult
    where
        'a: 'b,
    {
        use self::StepNormalResult::*;
        let mut frame = self.new_frame(func, args);
        loop {
            match self.step(&mut frame) {
                Ok(Continue) => {}
                Ok(Finish(ret)) => {
                    return Ok(ret);
                }
                Err(trap) => {
                    return Err(trap);
                }
            }
        }
    }

    /**
     * Execute one instruction, by mutating local states in `frame` and global
     * states in `self.runtime`.
     *
     * Never panics, as long as validation passes.
     */
    fn step<'b>(&'b mut self, frame: &mut Frame) -> StepResult
    where
        'a: 'b,
    {
        use self::StepNormalResult::*;

        //TODO: seems nasty. How to make it clean?
        //e.g. some strange pattern matching?
        if frame.labelstack.is_empty() {
            // no more to execute
            return frame.valstack.return_value(frame.arity);
        }
        let label = frame.labelstack.last_mut().unwrap();
        let instr = label.expr.instrs.get(label.pc);
        if instr.is_none() {
            // this block comes to an end (as everything)
            // as spec, values are left untouched, but the label shall exit
            // as an optimization, we implement loop here
            match label.continuation {
                LabelCont::Finish => {
                    frame.labelstack.pop();
                }
                LabelCont::Loop => {
                    label.pc = 0;
                }
            }
            return Ok(Continue);
        }
        let instr = instr.unwrap();
        let stack = &mut frame.valstack;
        // anyway, we have an instr, and a stack now.
        debug!("step, stack {:?}, instr {:?}", stack, instr);
        label.pc += 1;

        match instr {
            Instr::Drop => {
                let _ = stack.pop()?;
            }
            Instr::Select => {
                let cond: u32 = stack.pop()?.try_into()?;
                let val2 = stack.pop()?;
                let val1 = stack.pop()?;
                let ret = if cond == 0 { val2 } else { val1 };
                stack.push(ret);
            }

            /* Variable Instructions */
            Instr::LocalGet(idx) => {
                let local = frame.locals[*idx];
                stack.push(local);
            }
            Instr::LocalSet(idx) => {
                let val = stack.pop()?;
                //TODO: assert type matches
                // or shall we do that in validation?
                frame.locals[*idx] = val;
            }
            Instr::LocalTee(idx) => {
                let val = stack.tee()?;
                frame.locals[*idx] = val;
            }
            /*
             * we should have an indirection of
             * globaladdr, but it seems to be unnecessery:
             * globaladdrs[i] == i always holds.
             * Eliminate that.
             */
            Instr::GlobalGet(idx) => {
                // for what time will Non-lexical lifetimes goes sub-stmt?
                let val = self.globals[*idx].val;
                stack.push(val);
            }
            Instr::GlobalSet(idx) => {
                let val = stack.pop().unwrap();
                let global = &mut self.globals[*idx];
                // Validation ensures that the global is, in fact, marked as mutable.
                // https://webassembly.github.io/spec/core/bikeshed/index.html#-hrefsyntax-instr-variablemathsfglobalsetx%E2%91%A0
                global.val = val;
            }
            Instr::Nop => {}
            Instr::Unreachable => return Err(Trap {}),

            Instr::Block(block) => {
                frame.enter(block, LabelCont::Finish);
            }
            Instr::Loop(block) => {
                frame.enter(block, LabelCont::Finish);
            }
            Instr::IfElse { then, else_ } => {
                let cond: u32 = stack.pop()?.try_into()?;
                if cond != 0 {
                    frame.enter(then, LabelCont::Finish);
                } else {
                    if let Some(else_) = else_ {
                        frame.enter(else_, LabelCont::Finish);
                    }
                }
            }
            Instr::Br(idx) => {
                return frame.br(*idx);
            }
            Instr::BrIf(idx) => {
                let cond: u32 = stack.pop()?.try_into()?;
                if cond != 0 {
                    return frame.br(*idx);
                } else {
                    /* definitely not a trap */
                    /* do nothing either */
                    /* continues */
                }
            }
            Instr::BrTable(args) => unimplemented!(),
            Instr::Return => return stack.return_value(frame.arity),
            Instr::Call(idx) => {
                let func = self.funcs[*idx];
                let arity = func.type_.arity();
                let len = stack.len();
                let args: Vec<Val> = stack
                    .stack
                    .drain(len - arity..) // last `arity` vals
                    .collect();

                // func.type_.ret could be None, so we save this match
                match (self.invoke(func, &args[..]), &func.type_.ret) {
                    (Ok(Some(val)), Some(ret_type)) => {
                        debug!("invoke, args {:?}, ret {:?}\n", &args, val);
                        assert_or_trap(val.matches(ret_type))?;
                        //succeeded, push ret value back to current stack
                        stack.push(val);
                    }
                    // also succeeded, but returned nothing
                    (Ok(None), None) => {}
                    _ => {
                        return Err(Trap {});
                    }
                }
            }

            Instr::CallIndirect(idx) => unimplemented!(),
            /* numerics */
            Instr::I32Load(memarg) => self.load::<u32, u32>(frame, memarg)?,
            Instr::I64Load(memarg) => self.load::<u64, u64>(frame, memarg)?,
            Instr::F32Load(memarg) => self.load::<f32, f32>(frame, memarg)?,
            Instr::F64Load(memarg) => self.load::<f64, f64>(frame, memarg)?,
            Instr::I32Load8S(memarg) => self.load::<u32, i8>(frame, memarg)?,
            Instr::I32Load8U(memarg) => self.load::<u32, u8>(frame, memarg)?,
            Instr::I32Load16S(memarg) => self.load::<u32, i16>(frame, memarg)?,
            Instr::I32Load16U(memarg) => self.load::<u32, u16>(frame, memarg)?,
            Instr::I64Load8S(memarg) => self.load::<u64, i8>(frame, memarg)?,
            Instr::I64Load8U(memarg) => self.load::<u64, u8>(frame, memarg)?,
            Instr::I64Load16S(memarg) => self.load::<u64, i16>(frame, memarg)?,
            Instr::I64Load16U(memarg) => self.load::<u64, u16>(frame, memarg)?,
            Instr::I64Load32S(memarg) => self.load::<u64, i32>(frame, memarg)?,
            Instr::I64Load32U(memarg) => self.load::<u64, u32>(frame, memarg)?,

            Instr::I32Store(memarg) => self.store::<u32, u32>(frame, memarg)?,
            Instr::I64Store(memarg) => self.store::<u64, u64>(frame, memarg)?,
            Instr::F32Store(memarg) => self.store::<f32, f32>(frame, memarg)?,
            Instr::F64Store(memarg) => self.store::<f64, f64>(frame, memarg)?,
            Instr::I32Store8(memarg) => self.store::<u32, u8>(frame, memarg)?,
            Instr::I32Store16(memarg) => self.store::<u32, u16>(frame, memarg)?,
            Instr::I64Store8(memarg) => self.store::<u64, u8>(frame, memarg)?,
            Instr::I64Store16(memarg) => self.store::<u64, u16>(frame, memarg)?,
            Instr::I64Store32(memarg) => self.store::<u64, u32>(frame, memarg)?,

            Instr::MemorySize => unimplemented!(),
            Instr::MemoryGrow => unimplemented!(),
            // consts
            Instr::I32Const(val) => stack.push((*val).into()),
            Instr::F32Const(val) => stack.push((*val).into()),
            Instr::I64Const(val) => stack.push((*val).into()),
            Instr::F64Const(val) => stack.push((*val).into()),

            Instr::I32Eqz => stack.testop(|i: u32| i == 0)?,
            Instr::I32Eq => stack.relop::<u32, _, _>(u32::eq)?,
            Instr::I32Ne => stack.relop::<u32, _, _>(u32::ne)?,
            Instr::I32LtS => stack.relop::<u32, _, _>(i32::lt)?,
            Instr::I32LtU => stack.relop::<u32, _, _>(u32::lt)?,
            Instr::I32GtS => stack.relop::<u32, _, _>(i32::gt)?,
            Instr::I32GtU => stack.relop::<u32, _, _>(u32::gt)?,
            Instr::I32LeS => stack.relop::<u32, _, _>(i32::le)?,
            Instr::I32LeU => stack.relop::<u32, _, _>(u32::le)?,
            Instr::I32GeS => stack.relop::<u32, _, _>(i32::ge)?,
            Instr::I32GeU => stack.relop::<u32, _, _>(u32::ge)?,

            Instr::I64Eqz => stack.testop(|i: u64| i == 0)?,
            Instr::I64Eq => stack.relop::<u64, _, _>(u64::eq)?,
            Instr::I64Ne => stack.relop::<u64, _, _>(u64::ne)?,
            Instr::I64LtS => stack.relop::<u64, _, _>(i64::lt)?,
            Instr::I64LtU => stack.relop::<u64, _, _>(u64::lt)?,
            Instr::I64GtS => stack.relop::<u64, _, _>(i64::gt)?,
            Instr::I64GtU => stack.relop::<u64, _, _>(u64::gt)?,
            Instr::I64LeS => stack.relop::<u64, _, _>(i64::le)?,
            Instr::I64LeU => stack.relop::<u64, _, _>(u64::le)?,
            Instr::I64GeS => stack.relop::<u64, _, _>(i64::ge)?,
            Instr::I64GeU => stack.relop::<u64, _, _>(u64::ge)?,

            Instr::F32Eq => stack.relop::<f32, _, _>(f32::eq)?,
            Instr::F32Ne => stack.relop::<f32, _, _>(f32::ne)?,
            Instr::F32Lt => stack.relop::<f32, _, _>(f32::lt)?,
            Instr::F32Gt => stack.relop::<f32, _, _>(f32::gt)?,
            Instr::F32Le => stack.relop::<f32, _, _>(f32::le)?,
            Instr::F32Ge => stack.relop::<f32, _, _>(f32::ge)?,

            Instr::F64Eq => stack.relop::<f64, _, _>(f64::eq)?,
            Instr::F64Ne => stack.relop::<f64, _, _>(f64::ne)?,
            Instr::F64Lt => stack.relop::<f64, _, _>(f64::lt)?,
            Instr::F64Gt => stack.relop::<f64, _, _>(f64::gt)?,
            Instr::F64Le => stack.relop::<f64, _, _>(f64::le)?,
            Instr::F64Ge => stack.relop::<f64, _, _>(f64::ge)?,


            Instr::I32Clz => stack.unop(u32::leading_zeros)?,
            Instr::I32Ctz => stack.unop(u32::trailing_zeros)?,
            Instr::I32Popcnt => stack.unop(u32::count_zeros)?,
            Instr::I32Add => stack.binop(u32::wrapping_add)?,
            Instr::I32Sub => stack.binop(u32::wrapping_sub)?,
            Instr::I32Mul => stack.binop(u32::wrapping_mul)?,
            Instr::I32DivS => stack.binop_partial::<u32, _, _>(i32::checked_div)?,
            Instr::I32DivU => stack.binop_partial::<u32, _, _>(u32::checked_div)?,
            Instr::I32RemS => stack.binop_partial::<u32, _, _>(i32::checked_rem)?,
            Instr::I32RemU => stack.binop_partial::<u32, _, _>(u32::checked_rem)?,
            Instr::I32And => stack.binop(u32::bitand)?,
            Instr::I32Or => stack.binop(u32::bitor)?,
            Instr::I32Xor => stack.binop(u32::bitxor)?,
            // 32 is magic but come on
            Instr::I32Shl => stack.binop(|i: u32, j| i.shl(j % 32))?,
            Instr::I32ShrS => stack.binop(|i: u32, j| i.shr(j % 32))?,
            //TODO: there may be some culprits in the signed modulo
            //spec says "j modulo 32", but rust use % as reminder (signed) not
            //modulus (always >= 0).
            //https://webassembly.github.io/spec/core/exec/numerics.html#op-ishr-s
            Instr::I32ShrU => stack.binop(|i, j| (i as i32).shr(j % 32) as u32)?,
            Instr::I32Rotl => stack.binop(u32::rotate_left)?,
            Instr::I32Rotr => stack.binop(u32::rotate_right)?,


            Instr::I64Clz => stack.unop(|i: u64| i.leading_zeros() as u64)?,
            Instr::I64Ctz => stack.unop(|i: u64| i.trailing_zeros() as u64)?,
            Instr::I64Popcnt => stack.unop(|i: u64| i.count_zeros() as u64)?,
            Instr::I64Add => stack.binop(u64::wrapping_add)?,
            Instr::I64Sub => stack.binop(u64::wrapping_sub)?,
            Instr::I64Mul => stack.binop(u64::wrapping_mul)?,
            Instr::I64DivS => stack.binop_partial::<u64, _, _>(i64::checked_div)?,
            Instr::I64DivU => stack.binop_partial::<u64, _, _>(u64::checked_div)?,
            Instr::I64RemS => stack.binop_partial::<u64, _, _>(i64::checked_rem)?,
            Instr::I64RemU => stack.binop_partial::<u64, _, _>(u64::checked_rem)?,
            Instr::I64And => stack.binop(u64::bitand)?,
            Instr::I64Or => stack.binop(u64::bitor)?,
            Instr::I64Xor => stack.binop(u64::bitxor)?,
            // 64 is magic but come on
            Instr::I64Shl => stack.binop(|i: u64, j| i.shl(j % 64))?,
            Instr::I64ShrS => stack.binop(|i: u64, j| i.shr(j % 64))?,
            //TODO: there may be some culprits in the signed modulo
            //spec says "j modulo 64", but rust use % as reminder (signed) not
            //modulus (always >= 0).
            //https://webassembly.github.io/spec/core/exec/numerics.html#op-ishr-s
            Instr::I64ShrU => stack.binop(|i, j| (i as i64).shr(j % 64) as u64)?,
            Instr::I64Rotl => stack.binop(|i: u64, j| i.rotate_left(j as u32))?,
            Instr::I64Rotr => stack.binop(|i: u64, j| i.rotate_left(j as u32))?,

            Instr::F32Abs => stack.unop(f32::abs)?,
            Instr::F32Neg => stack.unop(f32::neg)?,
            Instr::F32Ceil => stack.unop(f32::ceil)?,
            Instr::F32Floor => stack.unop(f32::floor)?,
            Instr::F32Trunc => stack.unop(f32::trunc)?,
            // f32::round may behave differently from f32.nearest...
            // https://webassembly.github.io/spec/core/exec/numerics.html#op-fnearest
            // https://doc.rust-lang.org/std/primitive.f32.html#method.round
            Instr::F32Nearest => stack.unop(f32::round)?,
            Instr::F32Sqrt => stack.unop(f32::sqrt)?,
            Instr::F32Add => stack.binop(f32::add)?,
            Instr::F32Sub => stack.binop(f32::sub)?,
            Instr::F32Mul => stack.binop(f32::mul)?,
            Instr::F32Div => stack.binop(f32::div)?,
            Instr::F32Min => stack.binop(f32::min)?,
            Instr::F32Max => stack.binop(f32::max)?,
            Instr::F32Copysign => stack.binop(f32::copysign)?,

            Instr::F64Abs => stack.unop(f64::abs)?,
            Instr::F64Neg => stack.unop(f64::neg)?,
            Instr::F64Ceil => stack.unop(f64::ceil)?,
            Instr::F64Floor => stack.unop(f64::floor)?,
            Instr::F64Trunc => stack.unop(f64::trunc)?,
            // f64::round may behave differently from f64.nearest...
            // https://webassembly.github.io/spec/core/exec/numerics.html#op-fnearest
            // https://doc.rust-lang.org/std/primitive.f64.html#method.round
            Instr::F64Nearest => stack.unop(f64::round)?,
            Instr::F64Sqrt => stack.unop(f64::sqrt)?,
            Instr::F64Add => stack.binop(f64::add)?,
            Instr::F64Sub => stack.binop(f64::sub)?,
            Instr::F64Mul => stack.binop(f64::mul)?,
            Instr::F64Div => stack.binop(f64::div)?,
            Instr::F64Min => stack.binop(f64::min)?,
            Instr::F64Max => stack.binop(f64::max)?,
            Instr::F64Copysign => stack.binop(f64::copysign)?,

            Instr::I32WrapI64 => stack.cvtop(|i: u64| i as u32)?,
            Instr::I32TruncF32S => stack.cvtop(|i: f32| (i as i32) as u32)?,
            Instr::I32TruncF32U => stack.cvtop(|i: f32| i as u32)?,
            Instr::I32TruncF64S => stack.cvtop(|i: f64| (i as i32) as u32)?,
            Instr::I32TruncF64U => stack.cvtop(|i: f64| i as u32)?,
            Instr::I64ExtendI32S => stack.cvtop(|i: u32| (i as i64) as u64)?,
            Instr::I64ExtendI32U => stack.cvtop(|i: u32| i as u64)?,
            Instr::I64TruncF32S => stack.cvtop(|i: f32| (i as i64) as u64)?,
            Instr::I64TruncF32U => stack.cvtop(|i: f32| i as u64)?,
            Instr::I64TruncF64S => stack.cvtop(|i: f64| (i as i64) as u64)?,
            Instr::I64TruncF64U => stack.cvtop(|i: f64| i as u64)?,
            Instr::F32ConvertI32S => stack.cvtop(|i: u32| i as f32)?,
            Instr::F32ConvertI32U => stack.cvtop(|i: u32| i as f32)?,
            Instr::F32ConvertI64S => stack.cvtop(|i: u64| i as f32)?,
            Instr::F32ConvertI64U => stack.cvtop(|i: u64| i as f32)?,
            Instr::F32DemoteF64 => stack.cvtop(|i: f64| i as f32)?,
            Instr::F64ConvertI32S => stack.cvtop(|i: u32| i as f64)?,
            Instr::F64ConvertI32U => stack.cvtop(|i: u32| i as f64)?,
            Instr::F64ConvertI64S => stack.cvtop(|i: u64| i as f64)?,
            Instr::F64ConvertI64U => stack.cvtop(|i: u64| i as f64)?,
            Instr::F64PromoteF32 => stack.cvtop(|i: f32| i as f64)?,
            Instr::I32ReinterpretF32 => stack.cvtop(f32::to_bits)?,
            Instr::I64ReinterpretF64 => stack.cvtop(f64::to_bits)?,
            Instr::F32ReinterpretI32 => stack.cvtop(f32::from_bits)?,
            Instr::F64ReinterpretI64 => stack.cvtop(f64::from_bits)?,
        };
        Ok(Continue)
    }
}
#[cfg(test)]
mod test {

    use super::*;

    fn helper(rt: &mut Runtime, arg: u64, expected: u64) {
        let ret: u64 = rt
            .invoke(rt.funcs[0], &[arg.into()])
            .expect("invocation succeed")
            .expect("return value")
            .try_into()
            .expect("got u64");
        assert_eq!(ret, expected);
    }

    #[test]
    fn invoke_factorial() -> Result<(), Trap> {
        // i64 -> i64
        let type_ = Type {
            args: vec![ValType::I64],
            ret: Some(ValType::I64),
        };
        let module = Module {
            types: vec![type_.clone()],
            funcs: vec![Function {
                type_: 0,
                code: Code {
                    locals: vec![ValType::I64],
                    body: Expr {
                        instrs: vec![
                            Instr::LocalGet(0),
                            Instr::I64Const(0),
                            Instr::I64Eq,
                            Instr::IfElse {
                                then: Block {
                                    type_: Some(ValType::I64),
                                    expr: Expr {
                                        instrs: vec![Instr::I64Const(1)],
                                    },
                                },
                                else_: Some(Block {
                                    type_: Some(ValType::I64),
                                    expr: Expr {
                                        instrs: vec![
                                            Instr::LocalGet(0),
                                            Instr::LocalGet(0),
                                            Instr::I64Const(1),
                                            Instr::I64Sub,
                                            Instr::Call(0),
                                            Instr::I64Mul,
                                        ],
                                    },
                                }),
                            },
                        ],
                    },
                },
            }],
            globals: vec![],
            mem: Mem::empty(),
        };

        let mut rt = Runtime::instantiate(&module)?;

        helper(&mut rt, 0, 1);
        helper(&mut rt, 1, 1);
        helper(&mut rt, 2, 2);
        helper(&mut rt, 3, 6);
        helper(&mut rt, 4, 24);
        Ok(())
    }
}