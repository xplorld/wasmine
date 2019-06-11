use crate::instrs::*;
use crate::types::*;
use crate::util::*;
use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::ops::BitAnd;


macro_rules! u2i {
    // for relop.
    // converts a function of `(&$i, &$i) -> $ret` into `(&$u, &$u) -> $ret`
    ($f: ident, $u: ty, $i: ty, $ret: ty) => {
        |u1: &$u, u2: &$u| <$i>::$f(&(*u1 as $i), &(*u2 as $i)) as $ret
    };

    // for binop_partial.
    // converts a function of `($i, $i) -> Option<$i>` into
    // `($u, $u) -> Option<$u>`
    ($f: ident, $u: ty, $i: ty) => {
        |u1: $u, u2: $u| {
            if let Some(uret) = <$i>::$f(u1 as $i, u2 as $i) {
                Some(uret as $u)
            } else {
                None
            }
        }
    };
}


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

#[derive(Debug)]
pub struct Runtime {
    mem: MemInst,
    globals: Vec<GlobalInst>,
}

/**
 * Label: instrs, continuation, arity, (pc)
 */
#[derive(Debug, Clone)]
struct Label<'a> {
    pc: Idx,
    block: &'a Block,
    // idx of first val in valstack of this frame
    // always <= valstack.len()
    val_idx: Idx,
}


#[derive(Debug)]
struct ValStack {
    stack: Vec<Val>,
}

impl ValStack {
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


    fn br(&mut self, idx: usize) -> StepResult {
        Err(Trap {})
        // use self::StepNormalResult::*;

        // let (pos, &Label { arity, .. }) = self.nth_label(idx).ok_or(Trap {})?;

        // let len = self.stack.len();
        // self.stack.drain(pos..len - arity);
        // Ok(Continue)
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

    fn binop_partial<F, T>(&mut self, op: F) -> Result<(), Trap>
    where
        F: Fn(T, T) -> Option<T>,
        T: RawVal,
    {
        let val2 = self.pop()?.try_into()?;
        let val1 = self.pop()?.try_into()?;
        Ok(self.push(op(val1, val2).ok_or(Trap {})?.into()))
    }


    fn testop<F, T>(&mut self, op: F) -> Result<(), Trap>
    where
        F: Fn(T) -> bool,
        T: RawVal,
    {
        let val1 = self.pop()?.try_into()?;
        Ok(self.push((op(val1) as u32).into()))
    }

    fn relop<F, T>(&mut self, op: F) -> Result<(), Trap>
    where
        F: Fn(&T, &T) -> bool,
        T: RawVal,
    {
        let val2 = self.pop()?.try_into()?;
        let val1 = self.pop()?.try_into()?;
        Ok(self.push((op(&val1, &val2) as u32).into()))
    }
}

/**
 * Local context of an invocation.
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

/**
 * Separate readonly insts from all states
 * so that we could borrow the inst and mutate runtime in the same tme.
 */
#[derive(Debug)]
pub struct Context<'a> {
    module: &'a Module,
    runtime: Runtime,
}


impl<'a> Context<'a> {
    fn new(module: &'a Module) -> Context<'a> {
        Context {
            module: module,
            runtime: Runtime {
                mem: MemInst::new(&module.mems),
                globals: module.globals.iter().map(GlobalInst::new).collect(),
            },
        }
    }

    fn new_frame(&self, func: &'a Function, args: &[Val]) -> Frame<'a> {
        let mut frame = Frame {
            labelstack: vec![Label {
                pc: 0,
                block: &func.body,
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
    pub fn invoke(&mut self, func: &'a Function, args: &[Val]) -> InvocationResult {
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
    fn step(&mut self, frame: &mut Frame) -> StepResult {

        use self::StepNormalResult::*;
        let label = frame.labelstack.last().unwrap();
        let instr = &label.block.instrs[label.pc];
        let stack = &mut frame.valstack;

        debug!("step, stack {:?}, instr {:?}", stack, instr);

        match instr {
            Instr::Block(block) => {
                let label = Label {
                    pc: 0,
                    block,
                    val_idx: stack.stack.len(),
                };
                frame.labelstack.push(label);
            }
            // consts
            Instr::I32Const(val) => stack.push((*val).into()),

            Instr::F32Const(val) => stack.push((*val).into()),
            Instr::I64Const(val) => stack.push((*val).into()),
            Instr::F64Const(val) => stack.push((*val).into()),

            //iunop
            Instr::I32Clz => stack.unop(u32::leading_zeros)?,
            Instr::I64Clz => stack.unop(|i: u64| i.leading_zeros() as u64)?,
            Instr::I32Ctz => stack.unop(u32::trailing_zeros)?,
            Instr::I64Ctz => stack.unop(|i: u64| i.trailing_zeros() as u64)?,
            Instr::I32Popcnt => stack.unop(u32::count_zeros)?,
            Instr::I64Popcnt => stack.unop(|i: u64| i.count_zeros() as u64)?,
            //ibinop
            Instr::I32Add => stack.binop(u32::wrapping_add)?,
            Instr::I64Add => stack.binop(u64::wrapping_add)?,
            Instr::I32Sub => stack.binop(u32::wrapping_sub)?,
            Instr::I64Sub => stack.binop(u64::wrapping_sub)?,
            Instr::I32Mul => stack.binop(u32::wrapping_mul)?,
            Instr::I64Mul => stack.binop(u64::wrapping_mul)?,
            Instr::I32Divu => stack.binop_partial(u32::checked_div)?,
            Instr::I64Divu => stack.binop_partial(u64::checked_div)?,
            Instr::I32Divs => stack.binop_partial(u2i!(checked_div, u32, i32))?,
            Instr::I64Divs => stack.binop_partial(u2i!(checked_div, u64, i64))?,
            Instr::I32Remu => stack.binop_partial(u32::checked_rem)?,
            Instr::I64Remu => stack.binop_partial(u64::checked_rem)?,
            Instr::I32Rems => stack.binop_partial(u2i!(checked_rem, u32, i32))?,
            Instr::I64Rems => stack.binop_partial(u2i!(checked_rem, u64, i64))?,
            Instr::I32And => stack.binop(u32::bitand)?,
            Instr::I64And => stack.binop(u64::bitand)?,
            Instr::I32Or => unimplemented!(),
            Instr::I64Or => unimplemented!(),
            Instr::I32Xor => unimplemented!(),
            Instr::I64Xor => unimplemented!(),
            Instr::I32Shl => unimplemented!(),
            Instr::I64Shl => unimplemented!(),
            Instr::I32Shru => unimplemented!(),
            Instr::I64Shru => unimplemented!(),
            Instr::I32Shrs => unimplemented!(),
            Instr::I64Shrs => unimplemented!(),
            Instr::I32Rotl => unimplemented!(),
            Instr::I64Rotl => unimplemented!(),
            Instr::I32Rotr => unimplemented!(),
            Instr::I64Rotr => unimplemented!(),
            //funop
            Instr::F32Abs => unimplemented!(),
            Instr::F64Abs => unimplemented!(),
            Instr::F32Neg => unimplemented!(),
            Instr::F64Neg => unimplemented!(),
            Instr::F32Sqrt => unimplemented!(),
            Instr::F64Sqrt => unimplemented!(),
            Instr::F32Ceil => unimplemented!(),
            Instr::F64Ceil => unimplemented!(),
            Instr::F32Floor => unimplemented!(),
            Instr::F64Floor => unimplemented!(),
            Instr::F32Trunc => unimplemented!(),
            Instr::F64Trunc => unimplemented!(),
            Instr::F32Nearest => unimplemented!(),
            Instr::F64Nearest => unimplemented!(),
            //fbinop
            Instr::F32Add => unimplemented!(),
            Instr::F64Add => unimplemented!(),
            Instr::F32Sub => unimplemented!(),
            Instr::F64Sub => unimplemented!(),
            Instr::F32Mul => unimplemented!(),
            Instr::F64Mul => unimplemented!(),
            Instr::F32Div => unimplemented!(),
            Instr::F64Div => unimplemented!(),
            Instr::F32Min => unimplemented!(),
            Instr::F64Min => unimplemented!(),
            Instr::F32Max => unimplemented!(),
            Instr::F64Max => unimplemented!(),
            Instr::F32Copysign => unimplemented!(),
            Instr::F64Copysign => unimplemented!(),
            //itestop
            Instr::I32Eqz => stack.testop(|i: u32| i == 0)?,
            Instr::I64Eqz => stack.testop(|i: u64| i == 0)?,
            //irelop
            Instr::I32Eq => stack.relop(u32::eq)?,
            Instr::I64Eq => stack.relop(u64::eq)?,
            Instr::I32Ne => stack.relop(u32::ne)?,
            Instr::I64Ne => stack.relop(u64::ne)?,
            Instr::I32Ltu => stack.relop(u32::lt)?,
            Instr::I64Ltu => stack.relop(u64::lt)?,
            Instr::I32Lts => stack.relop(u2i!(lt, u32, i32, bool))?,
            Instr::I64Lts => stack.relop(u2i!(lt, u64, i64, bool))?,
            Instr::I32Gtu => stack.relop(u32::gt)?,
            Instr::I64Gtu => stack.relop(u64::gt)?,
            Instr::I32Gts => stack.relop(u2i!(gt, u32, i32, bool))?,
            Instr::I64Gts => stack.relop(u2i!(gt, u64, i64, bool))?,
            Instr::I32Leu => unimplemented!(),
            Instr::I64Leu => unimplemented!(),
            Instr::I32Les => unimplemented!(),
            Instr::I64Les => unimplemented!(),
            Instr::I32Geu => unimplemented!(),
            Instr::I64Geu => unimplemented!(),
            Instr::I32Ges => unimplemented!(),
            Instr::I64Ges => unimplemented!(),
            //frelop
            Instr::F32Eq => unimplemented!(),
            Instr::F64Eq => unimplemented!(),
            Instr::F32Ne => unimplemented!(),
            Instr::F64Ne => unimplemented!(),
            Instr::F32Lt => unimplemented!(),
            Instr::F64Lt => unimplemented!(),
            Instr::F32Gt => unimplemented!(),
            Instr::F64Gt => unimplemented!(),
            Instr::F32Le => unimplemented!(),
            Instr::F64Le => unimplemented!(),
            Instr::F32Ge => unimplemented!(),
            Instr::F64Ge => unimplemented!(),
            /* Parametric Instructions */
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
                let val = self.runtime.globals[*idx].val;
                stack.push(val);
            }
            Instr::GlobalSet(idx) => {
                let val = stack.pop().unwrap();
                let global = &mut self.runtime.globals[*idx];
                // Validation ensures that the global is, in fact, marked as mutable.
                // https://webassembly.github.io/spec/core/bikeshed/index.html#-hrefsyntax-instr-variablemathsfglobalsetx%E2%91%A0
                global.val = val;
            }
            /*
             * For memory instructions, we always use the one
             * and only memory
             */
            Instr::I32Load(memarg) => {
                let base: u32 = stack.pop()?.try_into()?;
                let eff = (base + memarg.offset) as usize;
                let val = slice_to_u32(&self.runtime.mem.data[eff..eff + 4]);
                stack.push(Val::I32(val));
            }
            Instr::I32Store(memarg) => {
                let base: u32 = stack.pop()?.try_into()?;
                let val: u32 = stack.pop()?.try_into()?;

                let eff = (base + memarg.offset) as usize;
                u32_to_slice(val, &mut self.runtime.mem.data[eff..eff + 4]);
            }
            Instr::Nop => {}
            Instr::Unreachable => return Err(Trap {}),

            /*
            Instr::Label(label) => {
                frame.stack.push(*label);
            }
            Instr::If { not_taken, label } => {
                let cond: u32 = stack.pop()?.try_into()?;

                if cond != 0 {
                    // self.runtime.pc simply increments
                } else {
                    frame.pc = *not_taken;
                }
                frame.stack.push(*label);
            }
            Instr::End | Instr::Else => {
                // an End may exit the last block or the last frame (invocation).

                // find first label. If there is no label, return the function.
                // for a label, simply removes it entry and jumps to the
                // continuation.
                if let Some((pos, &Label { continuation, .. })) = frame.nth_label(0) {
                    frame.stack.remove(pos);
                    frame.pc = continuation;
                // outside match:
                // return Ok(Continue);
                } else {
                    return frame.return_value();
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
            Instr::Return => return frame.return_value(),
            Instr::Call(idx) => {
                let func = &self.module.funcs[*idx];
                let arity = func.type_.arity();
                let len = frame.stack.len();
                let args: Vec<Val> = frame
                    .stack
                    .drain(len - arity..len)
                    .into_iter()
                    .filter_map(|entry| match entry {
                        val => Some(val),
                        _ => None,
                    })
                    .collect();
                // unlikely (only on failure), so do not check beforehand.
                assert_or_trap(args.len() == arity)?;

                // func.type_.ret could be None, so we save this match
                match (self.invoke(func, &args[..]), &func.type_.ret) {
                    (Ok(Some(val)), Some(ret_type)) => {
                        print!("invoke, args {:?}, ret {:?}\n", &args, val);
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
            */
            Instr::CallIndirect(idx) => unimplemented!(),
            //TODO: remove this
            _ => unimplemented!(),

        };
        Ok(Continue)
    }
}
/*
#[cfg(test)]
mod test {

    use super::*;

    fn helper(ctx: &mut Context, arg: u64, expected: u64) {
        let ret: u64 = ctx
            .invoke(&ctx.module.funcs[0], &[arg.into()])
            .ok()
            .unwrap()
            .unwrap()
            .try_into()
            .ok()
            .unwrap();
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
                type_: type_.clone(),
                locals: vec![ValType::I64],
                body: Expr {
                    instrs: vec![
                        Instr::LocalGet(0),
                        Instr::I64Const(0),
                        Instr::I64Eq,
                        Instr::If {
                            not_taken: 6,
                            label: Label {
                                arity: 0,
                                continuation: 13,
                            },
                        },
                        Instr::I64Const(1),
                        Instr::Else,
                        Instr::LocalGet(0),
                        Instr::LocalGet(0),
                        Instr::I64Const(1),
                        Instr::I64Sub,
                        Instr::Call(0),
                        Instr::I64Mul,
                        Instr::End,
                        Instr::End,
                    ],
                },
            }],
            globals: vec![],
            mems: Mem {
                limits: Limits {
                    min: 0,
                    max: Some(0),
                },
            },
        };

        let mut ctx = Context::new(&module);

        // helper(&mut ctx, 0, 1);
        // helper(&mut ctx, 1, 1);
        helper(&mut ctx, 2, 2);
        helper(&mut ctx, 3, 6);
        helper(&mut ctx, 4, 24);
        Ok(())
    }
}
*/