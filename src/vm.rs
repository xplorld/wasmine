mod wasmine {
    /**
     * A VM is an instance of WASM runtme.
     */
    struct VM {
        module: Module,
        stack: Vec<ValType>,
    }
}
