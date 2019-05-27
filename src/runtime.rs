use types::*;

enum StackEntry {
    Val(Val),
    Label(Label),
    Frame {
        locals: Vec<Val>,
        /* ModuleInst, which is always the same */
    },
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
    fn current_frame_locals(&self) -> Option<&mut Vec<Val>> {
        for entry in self.stack.iter_mut().rev() {
            match entry {
                StackEntry::Frame { locals: val } => {
                    return Option::Some(val);
                }
                _ => {}
            }
        }
        Option::None
    }

    fn pop_value_or_panic(&self) -> Val {
        if let Some(StackEntry::Val(val)) = self.stack.pop() {
            return val;
        } else {
            panic!("encountered drop but there's no value on top of stack")
        }
    }

    fn push_value(&self, val: Val) {
        self.stack.push(StackEntry::Val(val));
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


// TODO: return a boolean to indicate succeed/trap
// so that the caller have a chance to report the trap instead of
// simply panic'd
pub fn step(rt: &mut Runtime) {
    let instr: Instr = rt.insts[rt.pc];
    rt.pc+=1;
    match instr {
        Instr::I32Const(val) => rt.push_value(Val::I32 { i: val }),
        Instr::F32Const(val) => rt.push_value(Val::F32 { f: val }),
        Instr::I64Const(val) => rt.push_value(Val::I64 { i: val }),
        Instr::F64Const(val) => rt.push_value(Val::F64 { f: val }),
        Instr::Drop => {
            rt.pop_value_or_panic();
        }
        Instr::Select => {
            let cond = rt.pop_value_or_panic();
            let val2 = rt.pop_value_or_panic();
            let val1 = rt.pop_value_or_panic();
            if let Val::I32 { i: val } = cond {
                if val == 0 {
                    rt.push_value(val2);
                } else {
                    rt.push_value(val1);
                }
            }
        }
        Instr::LocalGet(idx) => {
            let locals = rt.current_frame_locals().unwrap();
            rt.push_value(locals[idx]);
        }
        Instr::LocalSet(idx) => {
            let locals = rt.current_frame_locals().unwrap();
            let val = rt.pop_value_or_panic();
            locals[idx] = val;
        }
        Instr::LocalTee(idx) => {
            let locals = rt.current_frame_locals().unwrap();
            let val = rt.pop_value_or_panic();
            rt.push_value(val);
            locals[idx] = val;
        }
        /*
         * we should have an indirection of 
         * globaladdr, but it seems to be unnecessery:
         * globaladdrs[i] == i always holds.
         * Eliminate that. 
         */
        Instr::GlobalGet(idx) => {
            rt.push_value(rt.globals[idx].val);
        }
        Instr::GlobalSet(idx) => {
            let global = rt.globals[idx];
            if let Mut::Var = global.mut_ {
                panic!("global {} cannot be written", idx);
            }
            global.val = rt.pop_value_or_panic();
        }
        /*
         * For memory instructions, we always use the one
         * and only memory
         */
        Instr::I32Load(memarg) => {
            if let Val::I32 {i:base} = rt.pop_value_or_panic() {
                let eff = (base + memarg.offset) as usize;
                let val = slice_to_u32(&rt.mem[eff..eff+4]);
                rt.push_value(Val::I32 {i:val});
            } else {
                panic!("top of stack is not value of i32");
            }
        }
        Instr::I32Store(memarg) => {
            if let Val::I32 {i:base} = rt.pop_value_or_panic() {
                if let Val::I32{i:val} = rt.pop_value_or_panic() {
                    let eff = (base + memarg.offset) as usize;
                    u32_to_slice(val, &mut rt.mem[eff..eff+4]);
                }
            } 
            panic!("top of stack is not value of i32");
        }
        Instr::Nop => {}
        Instr::Unreachable => {panic!("unreachable")}
        Instr::Label(label) => {
            rt.stack.push(StackEntry::Label(label));
        },
        Instr::IfElse {not_taken, label} => {
            if let Val::I32 {i:cond} = rt.pop_value_or_panic() {
                if cond != 0 {
                    // rt.pc simply increments
                } else {
                    rt.pc = not_taken;
                }
               rt.stack.push(StackEntry::Label(label)); 
            } else {
                panic!("stack top value type mismatch, expected i32");
            }
        },
        Instr::End => {
            // finds last Label, pops it and jumps to its continuation
            let pos = rt.stack.iter().rposition(|&entry| {
                if let StackEntry::Label(_) = entry {
                    return true;
                } else {
                    return false;
                }
            }).unwrap();
            if let StackEntry::Label(label) = rt.stack[pos] {
                rt.stack.remove(pos);
                rt.pc = label.continuation;
            } else {
                panic!("impossible, why isn't there a enum variant \
                comparing solution?");
            }
        },
        Instr::Br(idx) => {
            //TODO
        }
    }
}
