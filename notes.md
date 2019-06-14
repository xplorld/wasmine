# Table, Element and `call_indirect`

There may be 0 or 1 table, consisting of a vector of FuncRef, which is the function address (index of func in the instantiated module).

- Module.table: defines type of this table (only valid type is `anyfunc`)
- Module.elem: defines initial values in the table. If some index in the table is uninitialized, it contains None and `trap`s in invocation.
- Runtime.table: the one and only table (`Vec<Idx>`)
- `call_indirect(idx)`: `let funcidx = rt.table[stack.pop() as u32]; match(types[typeidx], funcidx.type_)?; invoke(funcidx);`
    - if `call_indirect` encounters uninitialized table entry, `trap`s.
    - if `call_indirect` encounters func of mismatched type, `trap`s.
    - `call_indirect` could only call functions in the `funcs` space (defined or imported).

JS API can change size of the runtime table & entries in the table, but WASM functions can not. Wasmine does not support that (for now).