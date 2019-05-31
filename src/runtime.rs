use types::*;
use util::*;

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

/**
 * Separate readonly insts from all states
 * so that we could borrow the inst and mutate runtime in the same tme.
 */
#[derive(Debug)]
pub struct Context<'a> {
    module: &'a Module,
    runtime: Runtime,
}

impl<'a> Frame<'a> {
    //TODO: refine it
    fn pop(&mut self) -> Option<Val> {
        match self.stack.pop() {
            Some(StackEntry::Val(val)) => Some(val),
            Some(other) => {
                self.stack.push(other);
                None
            }
            _ => None,
        }
    }

    fn tee(&self) -> Option<Val> {
        match self.stack.last() {
            Some(StackEntry::Val(val)) => Some(*val),
            _ => None,
        }
    }
    /**
     * return true.
     */
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
        if let Some((pos, &Label { arity, .. })) = self.nth_label(idx) {
            let len = self.stack.len();
            self.stack.drain(pos..len - arity);
            StepResult::Continue
        } else {
            StepResult::Complete(InvocationResult::Trap)
        }
    }

    fn return_value(&self) -> StepResult {
        if self.arity == 0 {
            StepResult::Complete(InvocationResult::Finish(None))
        } else {
            assert!(self.arity == 1);
            // unwrap and wrap == assert
            StepResult::Complete(InvocationResult::Finish(Some(self.tee().unwrap())))
        }
    }
}


#[derive(Debug)]
pub enum InvocationResult {
    Trap,
    Finish(Option<Val>), //arity 0 or 1
}

enum StepResult {
    Continue,
    Complete(InvocationResult),
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
        let mut frame = self.new_frame(func, args);
        loop {
            match self.step(&mut frame) {
                StepResult::Continue => {}
                StepResult::Complete(result) => {
                    print!("invoke, args {:?}, ret {:?}\n", &args, &result);
                    return result;
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
        let instr = &frame.insts[frame.pc];
        println!("step, stack {:?}, instr {:?}", frame.stack, instr);
        frame.pc += 1;
        match instr {
            Instr::I32Const(val) => {
                frame.push(Val::I32(*val));
                StepResult::Continue
            }
            Instr::F32Const(val) => {
                frame.push(Val::F32(*val));
                StepResult::Continue
            }
            Instr::I64Const(val) => {
                frame.push(Val::I64(*val));
                StepResult::Continue
            }
            Instr::F64Const(val) => {
                frame.push(Val::F64(*val));
                StepResult::Continue
            }
            Instr::I64Eq => {
                let val2 = frame.pop();
                let val1 = frame.pop();
                match (val2, val1) {
                    (Some(Val::I64(val2)), Some(Val::I64(val1))) => {
                        if val1 == val2 {
                            frame.push(Val::I32(1));
                        } else {
                            frame.push(Val::I32(0));
                        }
                        StepResult::Continue
                    }
                    _ => StepResult::Complete(InvocationResult::Trap),
                }
            }
            Instr::I64Sub => {
                let val2 = frame.pop();
                let val1 = frame.pop();
                match (val2, val1) {
                    (Some(Val::I64(val2)), Some(Val::I64(val1))) => {
                        frame.push(Val::I64(val1.wrapping_sub(val2)));
                        StepResult::Continue
                    }
                    _ => StepResult::Complete(InvocationResult::Trap),
                }
            }
            Instr::I64Mul => {
                let val2 = frame.pop();
                let val1 = frame.pop();
                match (val2, val1) {
                    (Some(Val::I64(val2)), Some(Val::I64(val1))) => {
                        frame.push(Val::I64(val1.wrapping_mul(val2)));
                        StepResult::Continue
                    }
                    _ => StepResult::Complete(InvocationResult::Trap),
                }
            }
            Instr::Drop => {
                if frame.pop().is_some() {
                    StepResult::Continue
                } else {
                    StepResult::Complete(InvocationResult::Trap)
                }
            }
            Instr::Select => {
                let cond = frame.pop();
                let val2 = frame.pop();
                let val1 = frame.pop();
                match (cond, val2, val1) {
                    (Some(Val::I32(cond)), Some(val2), Some(val1)) => {
                        frame.push(if cond == 0 { val2 } else { val1 });
                        StepResult::Continue
                    }
                    _ => StepResult::Complete(InvocationResult::Trap),
                }

            }
            Instr::LocalGet(idx) => {
                let local = frame.locals[*idx];
                frame.push(local);
                StepResult::Continue
            }
            Instr::LocalSet(idx) => {
                let val = frame.pop();
                match val {
                    Some(val) => {
                        //TODO: assert type matches
                        // or shall we do that in validation?
                        frame.locals[*idx] = val;
                        StepResult::Continue
                    }
                    None => StepResult::Complete(InvocationResult::Trap),
                }

            }
            Instr::LocalTee(idx) => {
                let val = frame.tee();
                match val {
                    Some(val) => {
                        frame.locals[*idx] = val;
                        StepResult::Continue
                    }
                    None => StepResult::Complete(InvocationResult::Trap),
                }
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
                StepResult::Continue
            }
            Instr::GlobalSet(idx) => {
                let val = frame.pop().unwrap();
                let global = &mut self.runtime.globals[*idx];
                // Validation ensures that the global is, in fact, marked as mutable.
                // https://webassembly.github.io/spec/core/bikeshed/index.html#-hrefsyntax-instr-variablemathsfglobalsetx%E2%91%A0
                global.val = val;
                StepResult::Continue
            }
            /*
             * For memory instructions, we always use the one
             * and only memory
             */
            Instr::I32Load(memarg) => match frame.pop() {
                Some(Val::I32(base)) => {
                    let eff = (base + memarg.offset) as usize;
                    let val = slice_to_u32(&self.runtime.mem.data[eff..eff + 4]);
                    frame.push(Val::I32(val));
                    StepResult::Continue
                }
                _ => StepResult::Complete(InvocationResult::Trap),
            },
            Instr::I32Store(memarg) => {
                let base = frame.pop();
                let val = frame.pop();
                match (base, val) {
                    (Some(Val::I32(base)), Some(Val::I32(val))) => {
                        let eff = (base + memarg.offset) as usize;
                        u32_to_slice(val, &mut self.runtime.mem.data[eff..eff + 4]);
                        StepResult::Continue
                    }
                    _ => StepResult::Complete(InvocationResult::Trap),
                }
            }
            Instr::Nop => StepResult::Continue,
            Instr::Unreachable => StepResult::Complete(InvocationResult::Trap),
            Instr::Label(label) => {
                frame.stack.push(StackEntry::Label(*label));
                StepResult::Continue
            }
            Instr::If { not_taken, label } => {
                if let Some(Val::I32(cond)) = frame.pop() {
                    if cond != 0 {
                        // self.runtime.pc simply increments
                    } else {
                        frame.pc = *not_taken;
                    }
                    frame.stack.push(StackEntry::Label(*label));
                    StepResult::Continue
                } else {
                    StepResult::Complete(InvocationResult::Trap)
                }
            }
            Instr::End | Instr::Else => {
                // an End may exit the last block or the last frame (invocation).

                // find first label. If there is no label, return the function.
                // for a label, simply removes it entry and jumps to the
                // continuation.
                if let Some((pos, &Label { continuation, .. })) = frame.nth_label(0) {
                    frame.stack.remove(pos);
                    frame.pc = continuation;
                    StepResult::Continue
                } else {
                    frame.return_value()
                }
            }
            Instr::Br(idx) => frame.br(*idx),
            Instr::BrIf(idx) => {
                let cond = frame.pop();
                if let Some(Val::I32(cond)) = cond {
                    if cond != 0 {
                        frame.br(*idx)
                    } else {
                        /* definitely not a trap */
                        /* do nothing either */
                        StepResult::Continue
                    }
                } else {
                    /* Oh traps */
                    StepResult::Complete(InvocationResult::Trap)
                }

            }
            Instr::BrTable(args) => unimplemented!(),
            Instr::Return => frame.return_value(),
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
                if args.len() != arity {
                    return StepResult::Complete(InvocationResult::Trap);
                }

                match (self.invoke(func, &args[..]), &func.type_.ret) {
                    (InvocationResult::Finish(Some(val)), Some(ret_type)) => {
                        print!("invoke, args {:?}, ret {:?}\n", &args, val);
                        if val.matches(ret_type) {
                            //succeeded, push ret value back to current stack
                            frame.push(val);
                            StepResult::Continue
                        } else {
                            StepResult::Complete(InvocationResult::Trap)
                        }
                    }
                    // also succeeded, but returned nothing
                    (InvocationResult::Finish(None), None) => StepResult::Continue,

                    _ => StepResult::Complete(InvocationResult::Trap),
                }
            }
            Instr::CallIndirect(idx) => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn invoke_factorial() {
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

        match ctx.invoke(&module.funcs[0], &[Val::I64(0)]) {
            InvocationResult::Finish(Some(Val::I64(ret))) => assert_eq!(ret, 1),
            _ => panic!("failed"),
        }

        match ctx.invoke(&module.funcs[0], &[Val::I64(1)]) {
            InvocationResult::Finish(Some(Val::I64(ret))) => assert_eq!(ret, 1),
            _ => panic!("failed"),
        }

        match ctx.invoke(&module.funcs[0], &[Val::I64(2)]) {
            InvocationResult::Finish(Some(Val::I64(ret))) => assert_eq!(ret, 2),
            _ => panic!("failed"),
        }

        match ctx.invoke(&module.funcs[0], &[Val::I64(3)]) {
            InvocationResult::Finish(Some(Val::I64(ret))) => assert_eq!(ret, 6),
            _ => panic!("failed"),
        }

        match ctx.invoke(&module.funcs[0], &[Val::I64(4)]) {
            InvocationResult::Finish(Some(Val::I64(ret))) => assert_eq!(ret, 24),
            _ => panic!("failed"),
        }
    }
}