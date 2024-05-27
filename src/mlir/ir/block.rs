use crate::support::binding::{
    impl_owned_mlir_value, impl_unowned_mlir_value, OwnedMlirValue, UnownedMlirValue,
};

use std::marker::PhantomData;

use mlir_sys::MlirBlock;

#[repr(transparent)]
#[derive(Debug)]
pub struct Block<'c> {
    raw: MlirBlock,
    _context: PhantomData<&'c ()>,
}

impl_owned_mlir_value!(context_ref, Block, MlirBlock);

#[repr(transparent)]
#[derive(Debug)]
pub struct BlockRef<'c> {
    _context: PhantomData<&'c ()>,
}

impl_unowned_mlir_value!(context_ref, Block, BlockRef, MlirBlock);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn no_owned_block_ref() {
        let _block_ref = BlockRef {
            _context: PhantomData,
        };
    }
}
