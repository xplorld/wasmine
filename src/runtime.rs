use types::*;

#[derive(Debug)]
enum StackEntry {
    Val(Val),
    Label(Label),
    Frame(Frame),
}

impl StackEntry {
    fn is_val(&self) -> bool {
        match self {
            StackEntry::Val(_) => true,
            _ => false
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
    insts: Vec<Instr>,
    pc: Idx,
    stack: Vec<StackEntry>,
    mem: Vec<u8>, 
    globals: Vec<GlobalInst>,
}

impl Runtime {
    /**
     * By spec, during execution there are always at least 1 frames in the stack
     * so there are no possibilities to return a `None`.
     */
    fn current_frame(&self) -> &mut Frame {
        for entry in self.stack.iter_mut().rev() {
            match entry {
                StackEntry::Frame(frame) => {
                    return frame;
                }
                _ => {}
            }
        }
        panic!("impossible")
    }

    fn pop(&self) -> Option<Val> {
        match self.stack.pop() {
            Some(StackEntry::Val(val)) => Some(val),
            Some(other) => {self.stack.push(other); None}
            _ => None
        }
    }
    /**
     * return true.
     */
    fn push(&self, val: Val) -> bool {
        self.stack.push(StackEntry::Val(val));
        true
    }
}

/**
 * TODO: use unsafe to eliminate copying.
 */
pub fn slice_to_u32(src: &[u8]) -> u32 {
    let arr: [u8;4];
    arr.copy_from_slice(src);
    u32::from_le_bytes(arr)
}

pub fn slice_to_u64(src: &[u8]) -> u64 {
    let arr: [u8;8];
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
pub fn step(rt: &mut Runtime) -> bool {
    let instr: Instr = rt.insts[rt.pc];
    rt.pc+=1;
    match instr {
        Instr::I32Const(val) => rt.push(Val::I32 { i: val }),
        Instr::F32Const(val) => rt.push(Val::F32 { f: val }),
        Instr::I64Const(val) => rt.push(Val::I64 { i: val }),
        Instr::F64Const(val) => rt.push(Val::F64 { f: val }),
        Instr::Drop => {
            rt.pop() != None
        }
        Instr::Select => {
            let cond = rt.pop();
            let val2 = rt.pop();
            let val1 = rt.pop();
            match (cond, val2, val1) {
                (Some(Val::I32 { i: cond }), Some(val2), Some(val1)) => 
                    rt.push(if cond == 0 { val2 } else {val1}),
                _ => false
            }
        }
        Instr::LocalGet(idx) => {
            let locals = rt.current_frame().locals;
            rt.push(locals[idx])
        }
        Instr::LocalSet(idx) => {
            let locals = rt.current_frame().locals;
            match rt.pop() {
                Some(val) => {locals[idx] = val; true}
                None => false
            }
        }
        Instr::LocalTee(idx) => {
             let locals = rt.current_frame().locals;
            match rt.pop() {
                Some(val) => {locals[idx] = val; rt.push(val)}
                None => false
            }
        }
        /*
         * we should have an indirection of 
         * globaladdr, but it seems to be unnecessery:
         * globaladdrs[i] == i always holds.
         * Eliminate that. 
         */
        Instr::GlobalGet(idx) => {
            rt.push(rt.globals[idx].val)
        }
        Instr::GlobalSet(idx) => {
            let global = rt.globals[idx];
            // Validation ensures that the global is, in fact, marked as mutable.
            // https://webassembly.github.io/spec/core/bikeshed/index.html#-hrefsyntax-instr-variablemathsfglobalsetx%E2%91%A0
            global.val = rt.pop().unwrap();
            true
        }
        /*
         * For memory instructions, we always use the one
         * and only memory
         */
        Instr::I32Load(memarg) => {
            match rt.pop() {
                Some(Val::I32 {i:base}) => {
                    let eff = (base + memarg.offset) as usize;
                    let val = slice_to_u32(&rt.mem[eff..eff+4]);
                    rt.push(Val::I32 {i:val})
                }
                _ => false
            }
        }
        Instr::I32Store(memarg) => {
            let base = rt.pop();
            let val = rt.pop();
            match (base, val) {
                (Some(Val::I32 {i:base}), Some(Val::I32{i:val})) => {
                    let eff = (base + memarg.offset) as usize;
                    u32_to_slice(val, &mut rt.mem[eff..eff+4]);
                    true
                }
                _ => false
            }
        }
        Instr::Nop => true,
        Instr::Unreachable => false,
        Instr::Label(label) => {
            rt.stack.push(StackEntry::Label(label));
            true
        }
        Instr::IfElse {not_taken, label} => {
            if let Some(Val::I32 {i:cond}) = rt.pop() {
                if cond != 0 {
                    // rt.pc simply increments
                } else {
                    rt.pc = not_taken;
                }
               rt.stack.push(StackEntry::Label(label));
               true
            } else {
               false
            }
        },
        Instr::End => {
            // an End may exit the last block or the last frame (invocation).

            // w.r.t. Validation, there are always Frame in the bottom
            let pos = rt.stack.iter()
                        .rposition(|&entry| !entry.is_val())
                        .unwrap();
            
                match rt.stack[pos] {
                    // In the frame case, keep the top #arity values, and 
                    // pop all other values & labels above the frame.
                    // https://webassembly.github.io/spec/core/bikeshed/index.html#returning-from-a-function%E2%91%A0
                    StackEntry::Frame(frame) => {
                        match frame.arity {
                            0 => {
                                rt.stack.truncate(pos);
                                true
                            }
                            1 => {
                                let ret = rt.stack.pop().unwrap();
                                rt.stack.truncate(pos);
                                rt.stack.push(ret);
                                true
                            }
                            _ => false
                        }
                    },
                    // In the label case, keep all values above the label, and jump
                    // to its continuation.
                    // https://webassembly.github.io/spec/core/bikeshed/index.html#exiting--hrefsyntax-instrmathitinstrast-with-label--l
                    StackEntry::Label(label) => {
                        rt.stack.remove(pos);
                        rt.pc = label.continuation;
                        true
                    },
                    _ => false // not possible
                }
        },
        Instr::Br(idx) => {
            //TODO
        }
    }
}
