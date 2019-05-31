
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