use types::*;

#[derive(Debug)]
enum StackEntry {
    Val(Val),
    Label(Label),
    Frame(Frame),
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

    fn is_frame(&self) -> bool {
        match self {
            StackEntry::Frame(_) => true,
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
    runtime: Runtime,
}

impl Runtime {
    /**
     * By spec, during execution there are always at least 1 frames in the stack
     * so there are no possibilities to return a `None`.
     */
    fn current_frame(&mut self) -> &mut Frame {
        self.stack
            .iter_mut()
            .rev()
            .filter_map(|entry| {
                if let StackEntry::Frame(frame) = entry {
                    Some(frame)
                } else {
                    None
                }
            })
            .next()
            .unwrap()
    }

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

    fn tee(&mut self) -> Option<Val> {
        match self.stack.last() {
            Some(StackEntry::Val(val)) => Some(*val),
            _ => None,
        }
    }
    /**
     * return true.
     */
    fn push(&mut self, val: Val) -> bool {
        self.stack.push(StackEntry::Val(val));
        true
    }

    fn nth_label(&self, n: usize) -> Option<usize> {
        self.stack
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, entry)| entry.is_label())
            .nth(n)
            .map(|(idx, _)| idx)
    }

    fn nth_frame(&self, n: usize) -> Option<usize> {
        self.stack
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, entry)| entry.is_frame())
            .nth(n)
            .map(|(idx, _)| idx)
    }

    fn br(&mut self, idx: usize) -> bool {
        if let Some(pos) = self.nth_label(idx) {
            if let StackEntry::Label(label) = self.stack[pos] {
                let len = self.stack.len();
                let arity = label.arity;
                self.stack.drain(pos..len - arity);
                return true;
            }
        }
        false
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


/**
 * Execute one instruction, by mutating states in `rt`.
 * A step results in two cases: succeed, or trap, marked True and False,
 * repectively. It is up to the caller to decide whether an external invocation
 * has ended or not.
 *
 * Never panics, as long as validation passes.
 *
 * TODO: separate Trap and (impossible) validation error.
 */
pub fn step(ctx: &mut Context) -> bool {
    let rt = &mut ctx.runtime;
    let instr = &ctx.insts[rt.pc];
    rt.pc += 1;
    match instr {
        Instr::I32Const(val) => rt.push(Val::I32 { i: *val }),
        Instr::F32Const(val) => rt.push(Val::F32 { f: *val }),
        Instr::I64Const(val) => rt.push(Val::I64 { i: *val }),
        Instr::F64Const(val) => rt.push(Val::F64 { f: *val }),
        Instr::Drop => rt.pop().is_some(),
        Instr::Select => {
            let cond = rt.pop();
            let val2 = rt.pop();
            let val1 = rt.pop();
            match (cond, val2, val1) {
                (Some(Val::I32 { i: cond }), Some(val2), Some(val1)) => {
                    rt.push(if cond == 0 { val2 } else { val1 })
                }
                _ => false,
            }
        }
        Instr::LocalGet(idx) => {
            let local = rt.current_frame().locals[*idx];
            rt.push(local)
        }
        Instr::LocalSet(idx) => {
            let val = rt.pop();
            let locals = &mut rt.current_frame().locals;
            match val {
                Some(val) => {
                    locals[*idx] = val;
                    true
                }
                None => false,
            }
        }
        Instr::LocalTee(idx) => {
            let val = rt.tee();
            let locals = &mut rt.current_frame().locals;
            match val {
                Some(val) => {
                    locals[*idx] = val;
                    true
                }
                None => false,
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
            let val = rt.globals[*idx].val;
            rt.push(val)
        }
        Instr::GlobalSet(idx) => {
            let val = rt.pop().unwrap();
            let global = &mut rt.globals[*idx];
            // Validation ensures that the global is, in fact, marked as mutable.
            // https://webassembly.github.io/spec/core/bikeshed/index.html#-hrefsyntax-instr-variablemathsfglobalsetx%E2%91%A0
            global.val = val;
            true
        }
        /*
         * For memory instructions, we always use the one
         * and only memory
         */
        Instr::I32Load(memarg) => match rt.pop() {
            Some(Val::I32 { i: base }) => {
                let eff = (base + memarg.offset) as usize;
                let val = slice_to_u32(&rt.mem[eff..eff + 4]);
                rt.push(Val::I32 { i: val })
            }
            _ => false,
        },
        Instr::I32Store(memarg) => {
            let base = rt.pop();
            let val = rt.pop();
            match (base, val) {
                (Some(Val::I32 { i: base }), Some(Val::I32 { i: val })) => {
                    let eff = (base + memarg.offset) as usize;
                    u32_to_slice(val, &mut rt.mem[eff..eff + 4]);
                    true
                }
                _ => false,
            }
        }
        Instr::Nop => true,
        Instr::Unreachable => false,
        Instr::Label(label) => {
            rt.stack.push(StackEntry::Label(*label));
            true
        }
        Instr::IfElse { not_taken, label } => {
            if let Some(Val::I32 { i: cond }) = rt.pop() {
                if cond != 0 {
                    // rt.pc simply increments
                } else {
                    rt.pc = *not_taken;
                }
                rt.stack.push(StackEntry::Label(*label));
                true
            } else {
                false
            }
        }
        Instr::End => {
            // an End may exit the last block or the last frame (invocation).

            // w.r.t. Validation, there are always Frame in the bottom
            let pos = rt.stack.iter().rposition(|entry| !entry.is_val()).unwrap();
            match rt.stack[pos] {
                // In the frame case, keep the top #arity values, and
                // pop all other values & labels above the frame.
                // https://webassembly.github.io/spec/core/bikeshed/index.html#returning-from-a-function%E2%91%A0
                StackEntry::Frame(Frame { arity, locals: _ }) => {
                    let end = rt.stack.len() - arity;
                    rt.stack.drain(pos..end);
                    true
                }
                // In the label case, keep all values above the label, and jump
                // to its continuation.
                // https://webassembly.github.io/spec/core/bikeshed/index.html#exiting--hrefsyntax-instrmathitinstrast-with-label--l
                StackEntry::Label(label) => {
                    rt.stack.remove(pos);
                    rt.pc = label.continuation;
                    true
                }
                _ => false, // not possible
            }
        }
        Instr::Br(idx) => rt.br(*idx),
        Instr::BrIf(idx) => {
            let cond = rt.pop();
            if let Some(Val::I32 { i: cond }) = cond {
                if cond != 0 {
                    rt.br(*idx)
                } else {
                    /* definitely not a trap */
                    true
                }
            } else {
                /* Oh traps */
                false
            }
        }
        Instr::BrTable(args) => unimplemented!(),
        Instr::Return => unimplemented!(),
        Instr::Call(idx) => unimplemented!(),
        Instr::CallIndirect(idx) => unimplemented!(),
    }
}
