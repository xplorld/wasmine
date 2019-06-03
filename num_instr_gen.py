ops = {"iunop": ["clz", "ctz", "popcnt"],
       "ibinop": ["add", "sub", "mul", "div_sx", "rem_sx", "and",
                  "or", "xor", "shl", "shr_sx", "rotl", "rotr", ],
       "funop": ["abs", "neg", "sqrt", "ceil", "floor", "trunc", "nearest", ],
       "fbinop": ["add", "sub", "mul", "div", "min", "max", "copysign", ],
       "itestop": ["eqz", ],
       "irelop": ["eq", "ne", "lt_sx", "gt_sx", "le_sx", "ge_sx", ],
       "frelop": ["eq", "ne", "lt", "gt", "le", "ge", ],
       }


def inst_sx(insts):
    for inst in insts:
        if inst.endswith("_sx"):
            yield inst[:-3] + "U"
            yield inst[:-3] + "S"
        else:
            yield inst


def inst_names():
    for typ, insts in ops.items():
        yield f"//{typ}"
        for inst in inst_sx(insts):
            if typ[0] == 'i':
                yield f"I32{inst.capitalize()}"
                yield f"I64{inst.capitalize()}"
            else:
                yield f"F32{inst.capitalize()}"
                yield f"F64{inst.capitalize()}"


def defines():
    for name in inst_names():
        print(f"{name},")


def impls():
    for name in inst_names():
        if name.startswith("//"):
            print(name)
        else:
            print(f"Instr::{name} => unimplemented!(),")


impls()
