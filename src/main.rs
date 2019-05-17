mod types;

fn main() {
    println!("Hello, world!");
    let v = types::Val::I32 {i : 2};
    println!("{:?}", v);
}
