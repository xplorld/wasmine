
use instrs::*;
use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use types::*;
use util::*;

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

enum StepNormalResult {
    Continue,
    Finish(Option<Val>),
}

type StepResult = Result<StepNormalResult, Trap>;


#[derive(Debug)]
enum StackEntry {
    Val(Val),
    Label(Label),
}

// I hate this block of boilerplate
impl StackEntry {

    fn is_val(&self) -> bool {
        match self {
            StackEntry::Val(_) => true,
            _ => false,
        }
    }

    fn is_label(&self) -> bool {
        match self {
            StackEntry::Label(_) => true,
            _ => false,
        }
    }
}

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
 * Local context of an invocation.
 */
#[derive(Debug)]
pub struct Frame<'a> {
    pc: Idx,
    insts: &'a Vec<Instr>,
    stack: Vec<StackEntry>,
    arity: usize, // 0 or 1
    locals: Vec<Val>,
}

impl<'a> Frame<'a> {
    //TODO: refine it
    fn pop(&mut self) -> Result<Val, Trap> {
        match self.stack.pop() {
            Some(StackEntry::Val(val)) => Ok(val),
            _ => Err(Trap {}),
        }
    }

    fn tee(&self) -> Result<Val, Trap> {
        match self.stack.last() {
            Some(StackEntry::Val(val)) => Ok(*val),
            _ => Err(Trap {}),
        }
    }
    fn push(&mut self, val: Val) {
        self.stack.push(StackEntry::Val(val));
    }

    fn nth_label(&self, n: usize) -> Option<(usize, &Label)> {
        self.stack
            .iter()
            .enumerate()
            .rev()
            .filter_map(|(pos, entry)| match entry {
                StackEntry::Label(label) => Some((pos, label)),
                _ => None,
            })
            .nth(n)
    }

    fn br(&mut self, idx: usize) -> StepResult {
        use self::StepNormalResult::*;

        let (pos, &Label { arity, .. }) = self.nth_label(idx).ok_or(Trap {})?;

        let len = self.stack.len();
        self.stack.drain(pos..len - arity);
        Ok(Continue)
    }

    fn return_value(&self) -> StepResult {
        use self::StepNormalResult::*;
        if self.arity == 0 {
            Ok(Finish(None))
        } else {
            assert!(self.arity == 1);
            // unwrap and wrap == assert
            Ok(Finish(Some(self.tee().unwrap())))
        }
    }

    /**
     * Beacause of function name, I can't think of a macro solution for this
     * boilerplate.
     */

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
        F: Fn(T, T) -> bool,
        T: RawVal,
    {
        let val2 = self.pop()?.try_into()?;
        let val1 = self.pop()?.try_into()?;
        Ok(self.push((op(val1, val2) as u32).into()))
    }
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
            pc: 0,
            insts: &func.body.instrs,
            stack: Vec::new(),
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
        let instr = &frame.insts[frame.pc];
        println!("step, stack {:?}, instr {:?}", frame.stack, instr);
        frame.pc += 1;
        match instr {
            // consts
            Instr::I32Const(val) => frame.push((*val).into()),
            Instr::F32Const(val) => frame.push((*val).into()),
            Instr::I64Const(val) => frame.push((*val).into()),
            Instr::F64Const(val) => frame.push((*val).into()),
            //
            Instr::I32Clz => frame.unop(u32::leading_zeros)?,
            Instr::I64Sub => frame.binop(u64::wrapping_sub)?,
            Instr::I64Mul => frame.binop(u64::wrapping_mul)?,
            Instr::I64Eq => frame.relop(|val1: u64, val2: u64| val1 == val2)?,
            Instr::Drop => {
                let _ = frame.pop()?;
            }
            Instr::Select => {
                let cond: u32 = frame.pop()?.try_into()?;
                let val2 = frame.pop()?;
                let val1 = frame.pop()?;
                let ret = if cond == 0 { val2 } else { val1 };
                frame.push(ret);
            }
            Instr::LocalGet(idx) => {
                let local = frame.locals[*idx];
                frame.push(local);
            }
            Instr::LocalSet(idx) => {
                let val = frame.pop()?;
                //TODO: assert type matches
                // or shall we do that in validation?
                frame.locals[*idx] = val;
            }
            Instr::LocalTee(idx) => {
                let val = frame.tee()?;
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
                frame.push(val);
            }
            Instr::GlobalSet(idx) => {
                let val = frame.pop().unwrap();
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
                let base: u32 = frame.pop()?.try_into()?;
                let eff = (base + memarg.offset) as usize;
                let val = slice_to_u32(&self.runtime.mem.data[eff..eff + 4]);
                frame.push(Val::I32(val));
            }
            Instr::I32Store(memarg) => {
                let base: u32 = frame.pop()?.try_into()?;
                let val: u32 = frame.pop()?.try_into()?;

                let eff = (base + memarg.offset) as usize;
                u32_to_slice(val, &mut self.runtime.mem.data[eff..eff + 4]);
            }
            Instr::Nop => {}
            Instr::Unreachable => return Err(Trap {}),
            Instr::Label(label) => {
                frame.stack.push(StackEntry::Label(*label));
            }
            Instr::If { not_taken, label } => {
                let cond: u32 = frame.pop()?.try_into()?;

                if cond != 0 {
                    // self.runtime.pc simply increments
                } else {
                    frame.pc = *not_taken;
                }
                frame.stack.push(StackEntry::Label(*label));
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
                let cond: u32 = frame.pop()?.try_into()?;
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
                        StackEntry::Val(val) => Some(val),
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
                        frame.push(val);
                    }
                    // also succeeded, but returned nothing
                    (Ok(None), None) => {}
                    _ => {
                        return Err(Trap {});
                    }
                }
            }
            Instr::CallIndirect(idx) => unimplemented!(),
        };
        Ok(Continue)
    }
}

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
        // let ret: u64 = ret.try_into().ok().unwrap();

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