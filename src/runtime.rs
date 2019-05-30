use types::*;

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
pub struct GlobalInst {
    mut_: Mut,
    val: Val,
}


/**
 * A VM is an instance of WASM runtme.
 */
pub struct Runtime {
    pc: Idx,
    stack: Vec<StackEntry>,
    mem: Vec<u8>,
    globals: Vec<GlobalInst>,
}

/**
 * Separate readonly insts from all states
 * so that we could borrow the inst and mutate runtime in the same tme.
 */
pub struct Context {
    insts: Vec<Instr>,
    frame: Frame,
    runtime: Runtime,
}

impl Runtime {

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
            StepResult::Trap
        }
    }
}

/**
 * TODO: use unsafe to eliminate copying.
 */
pub fn slice_to_u32(src: &[u8]) -> u32 {
    let mut arr: [u8; 4] = [0; 4];
    arr.copy_from_slice(src);
    u32::from_le_bytes(arr)
}

pub fn slice_to_u64(src: &[u8]) -> u64 {
    let mut arr: [u8; 8] = [0; 8];
    arr.copy_from_slice(src);
    u64::from_le_bytes(arr)
}

pub fn slice_to_f32(src: &[u8]) -> f32 {
    f32::from_bits(slice_to_u32(src))
}

pub fn slice_to_f64(src: &[u8]) -> f64 {
    f64::from_bits(slice_to_u64(src))
}

pub fn u32_to_slice(src: u32, dst: &mut [u8]) {
    dst.copy_from_slice(&src.to_le_bytes()[..])
}

pub fn u64_to_slice(src: u64, dst: &mut [u8]) {
    dst.copy_from_slice(&src.to_le_bytes()[..])
}

pub fn f32_to_slice(src: f32, dst: &mut [u8]) {
    u32_to_slice(src.to_bits(), dst)
}

pub fn f64_to_slice(src: f64, dst: &mut [u8]) {
    u64_to_slice(src.to_bits(), dst)
}

enum StepResult {
    Continue,
    Trap,
    Finish(Option<Val>), // arity 0 or 1
}

impl Context {
    fn return_value(&self) -> StepResult {
        if self.frame.arity == 0 {
            StepResult::Finish(None)
        } else {
            assert!(self.frame.arity == 1);
            // unwrap and wrap == assert
            StepResult::Finish(Some(self.runtime.tee().unwrap()))
        }
    }
}


/**
 * Execute one instruction, by mutating states in `ctx.runtime`.
 * A step results in two cases: succeed, or trap, marked True and False,
 * repectively. It is up to the caller to decide whether an external invocation
 * has ended or not.
 *
 * Never panics, as long as validation passes.
 *
 * TODO: separate Trap and (impossible) validation error.
 */
fn step(ctx: &mut Context) -> StepResult {
    let instr = &ctx.insts[ctx.runtime.pc];
    ctx.runtime.pc += 1;
    match instr {
        Instr::I32Const(val) => {
            ctx.runtime.push(Val::I32(*val));
            StepResult::Continue
        }
        Instr::F32Const(val) => {
            ctx.runtime.push(Val::F32(*val));
            StepResult::Continue
        }
        Instr::I64Const(val) => {
            ctx.runtime.push(Val::I64(*val));
            StepResult::Continue
        }
        Instr::F64Const(val) => {
            ctx.runtime.push(Val::F64(*val));
            StepResult::Continue
        }
        Instr::Drop => {
            if ctx.runtime.pop().is_some() {
                StepResult::Continue
            } else {
                StepResult::Trap
            }
        }
        Instr::Select => {
            let cond = ctx.runtime.pop();
            let val2 = ctx.runtime.pop();
            let val1 = ctx.runtime.pop();
            match (cond, val2, val1) {
                (Some(Val::I32(cond)), Some(val2), Some(val1)) => {
                    ctx.runtime.push(if cond == 0 { val2 } else { val1 });
                    StepResult::Continue
                }
                _ => StepResult::Trap,
            }
        }
        Instr::LocalGet(idx) => {
            let local = ctx.frame.locals[*idx];
            ctx.runtime.push(local);
            StepResult::Continue
        }
        Instr::LocalSet(idx) => {
            let val = ctx.runtime.pop();
            match val {
                Some(val) => {
                    ctx.frame.locals[*idx] = val;
                    StepResult::Continue
                }
                None => StepResult::Trap,
            }
        }
        Instr::LocalTee(idx) => {
            let val = ctx.runtime.tee();
            match val {
                Some(val) => {
                    ctx.frame.locals[*idx] = val;
                    StepResult::Continue
                }
                None => StepResult::Trap,
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
            let val = ctx.runtime.globals[*idx].val;
            ctx.runtime.push(val);
            StepResult::Continue
        }
        Instr::GlobalSet(idx) => {
            let val = ctx.runtime.pop().unwrap();
            let global = &mut ctx.runtime.globals[*idx];
            // Validation ensures that the global is, in fact, marked as mutable.
            // https://webassembly.github.io/spec/core/bikeshed/index.html#-hrefsyntax-instr-variablemathsfglobalsetx%E2%91%A0
            global.val = val;
            StepResult::Continue
        }
        /*
         * For memory instructions, we always use the one
         * and only memory
         */
        Instr::I32Load(memarg) => match ctx.runtime.pop() {
            Some(Val::I32(base)) => {
                let eff = (base + memarg.offset) as usize;
                let val = slice_to_u32(&ctx.runtime.mem[eff..eff + 4]);
                ctx.runtime.push(Val::I32(val));
                StepResult::Continue
            }
            _ => StepResult::Trap,
        },
        Instr::I32Store(memarg) => {
            let base = ctx.runtime.pop();
            let val = ctx.runtime.pop();
            match (base, val) {
                (Some(Val::I32(base)), Some(Val::I32(val))) => {
                    let eff = (base + memarg.offset) as usize;
                    u32_to_slice(val, &mut ctx.runtime.mem[eff..eff + 4]);
                    StepResult::Continue
                }
                _ => StepResult::Trap,
            }
        }
        Instr::Nop => StepResult::Continue,
        Instr::Unreachable => StepResult::Trap,
        Instr::Label(label) => {
            ctx.runtime.stack.push(StackEntry::Label(*label));
            StepResult::Continue
        }
        Instr::IfElse { not_taken, label } => {
            if let Some(Val::I32(cond)) = ctx.runtime.pop() {
                if cond != 0 {
                    // ctx.runtime.pc simply increments
                } else {
                    ctx.runtime.pc = *not_taken;
                }
                ctx.runtime.stack.push(StackEntry::Label(*label));
                StepResult::Continue
            } else {
                StepResult::Trap
            }
        }
        Instr::End => {
            // an End may exit the last block or the last ctx.frame (invocation).

            // find first label. If there is no labels, return the function.
            // for a label, simply removes it entry and jumps to the
            // continuation.
            if let Some((pos, &Label { continuation, .. })) = ctx.runtime.nth_label(0) {
                ctx.runtime.stack.remove(pos);
                ctx.runtime.pc = continuation;
                StepResult::Continue
            } else {
                ctx.return_value()
            }
        }
        Instr::Br(idx) => ctx.runtime.br(*idx),
        Instr::BrIf(idx) => {
            let cond = ctx.runtime.pop();
            if let Some(Val::I32(cond)) = cond {
                if cond != 0 {
                    ctx.runtime.br(*idx)
                } else {
                    /* definitely not a trap */
                    /* do nothing either */
                    StepResult::Continue
                }
            } else {
                /* Oh traps */
                StepResult::Trap
            }
        }
        Instr::BrTable(args) => unimplemented!(),
        Instr::Return => ctx.return_value(),
        Instr::Call(idx) => unimplemented!(),
        Instr::CallIndirect(idx) => unimplemented!(),
    }
}
