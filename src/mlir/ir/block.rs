use crate::{
    ir::{LocationRef, Operation, OperationRef, RegionRef, TypeRef, ValueRef},
    support::{
        binding::{
            impl_owned_mlir_value, impl_unowned_mlir_value, OwnedMlirValue, UnownedMlirValue,
        },
        string_reader::StringReader,
    },
};

use std::{
    fmt::{Display, Formatter},
    marker::PhantomData,
    mem::forget,
};

use mlir_sys::{
    mlirBlockAppendOwnedOperation, mlirBlockCreate, mlirBlockDestroy, mlirBlockEqual,
    mlirBlockGetArgument, mlirBlockGetFirstOperation, mlirBlockGetNextInRegion,
    mlirBlockGetNumArguments, mlirBlockGetParentOperation, mlirBlockGetParentRegion,
    mlirBlockGetTerminator, mlirBlockInsertOwnedOperationAfter, mlirBlockPrint,
    MlirBlock, MlirLocation, MlirType,
};

/// [Block] wraps the `mlir::Block` class, which represents a block of operations in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirBlockAppendOwnedOperation`
/// - `mlirBlockCreate`
/// - `mlirBlockDestroy`
/// - `mlirBlockEqual`
/// - `mlirBlockGetArgument`
/// - `mlirBlockGetFirstOperation`
/// - `mlirBlockGetNextInRegion`
/// - `mlirBlockGetNumArguments`
/// - `mlirBlockGetParentOperation`
/// - `mlirBlockGetParentRegion`
/// - `mlirBlockGetTerminator`
/// - `mlirBlockPrint`
///
/// The following bindings are not used/supported:
/// - `mlirBlockAddArgument`
/// - `mlirBlockArgumentGetArgNumber`
/// - `mlirBlockArgumentGetOwner`
/// - `mlirBlockArgumentSetType`
/// - `mlirBlockDetach`
/// - `mlirBlockInsertArgument`
/// - `mlirBlockInsertOwnedOperationAfter`
/// - `mlirBlockInsertOwnedOperationBefore`
/// - `mlirBlockInsertOwnedOperation`
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Block<'c> {
    raw: MlirBlock,
    _context: PhantomData<&'c ()>,
}

impl_owned_mlir_value!(context_ref, Block, MlirBlock);

impl<'c> Block<'c> {
    /// Creates a new block.
    ///
    /// # Arguments
    /// * `args` - The types and locations of the input arguments to the block.
    ///
    /// # Returns
    /// Returns a new [Block] instance.
    pub fn new(args: &[(&'c TypeRef, &'c LocationRef)]) -> Block<'c> {
        let types = args.iter().map(|&(t, _)| t).collect::<Vec<_>>();
        let locs = args.iter().map(|&(_, l)| l).collect::<Vec<_>>();
        unsafe {
            Self::from_raw(mlirBlockCreate(
                args.len() as isize,
                types.as_ptr() as *const MlirType,
                locs.as_ptr() as *const MlirLocation,
            ))
        }
    }
}

impl<'c> Drop for Block<'c> {
    fn drop(&mut self) {
        unsafe { mlirBlockDestroy(self.raw) }
    }
}

impl<'c> PartialEq for Block<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirBlockEqual(self.raw, other.raw) }
    }
}

impl<'c> PartialEq<BlockRef<'c>> for Block<'c> {
    fn eq(&self, other: &BlockRef<'c>) -> bool {
        unsafe { mlirBlockEqual(self.raw, other.to_raw()) }
    }
}

impl<'c> Eq for Block<'c> {}

#[repr(transparent)]
#[derive(Debug)]
pub struct BlockRef<'c> {
    _context: PhantomData<&'c ()>,
}

impl_unowned_mlir_value!(context_ref, Block, BlockRef, MlirBlock);

impl<'c> BlockRef<'c> {
    /// Appends the given operation to the block.
    ///
    /// # Arguments
    /// * `operation` - The operation to append to the block.
    ///
    /// # Returns
    /// Returns a reference to the appended operation owned by the block.
    pub fn append_operation<'a>(&'a self, operation: Operation<'c>) -> &'a OperationRef<'c> {
        let operation_ref = unsafe { OperationRef::from_raw(operation.to_raw()) };
        unsafe { mlirBlockAppendOwnedOperation(self.to_raw(), operation.to_raw()) };
        forget(operation);
        operation_ref
    }

    /// # Returns
    /// Returns the number of arguments the block has.
    pub fn num_arguments(&self) -> isize {
        unsafe { mlirBlockGetNumArguments(self.to_raw()) }
    }

    /// Gets the argument at the provided index, verifying that the index is within bounds.
    ///
    /// # Arguments
    /// * `index` - The index of the argument to get.
    ///
    /// # Returns
    /// Returns the argument at the provided index.
    pub fn argument(&self, idx: isize) -> &ValueRef<'c> {
        if idx < 0 || idx >= self.num_arguments() {
            panic!("Argument index {} out of bounds", idx);
        }
        unsafe { ValueRef::from_raw(mlirBlockGetArgument(self.to_raw(), idx)) }
    }

    /// # Returns
    /// If the block is nested within a region, returns the parent region.
    pub fn parent_region(&self) -> Option<&RegionRef<'c>> {
        unsafe { RegionRef::try_from_raw(mlirBlockGetParentRegion(self.to_raw())) }
    }

    /// # Returns
    /// If the block is nested within an operation, returns the parent operation.
    pub fn parent_operation(&self) -> Option<&OperationRef<'c>> {
        unsafe { OperationRef::try_from_raw(mlirBlockGetParentOperation(self.to_raw())) }
    }

    /// # Returns
    /// If this block is in a region and there is another block after it in the region,
    /// returns that block.
    pub fn next_in_parent_region(&self) -> Option<&BlockRef<'c>> {
        unsafe { Self::try_from_raw(mlirBlockGetNextInRegion(self.to_raw())) }
    }

    /// # Returns
    /// Returns the first operation of the region, if it has one.
    pub fn first_operation(&self) -> Option<&OperationRef<'c>> {
        unsafe { OperationRef::try_from_raw(mlirBlockGetFirstOperation(self.to_raw())) }
    }

    /// # Returns
    /// Returns the terminating operation of the region, if it has one.
    pub fn terminator(&self) -> Option<&OperationRef<'c>> {
        unsafe { OperationRef::try_from_raw(mlirBlockGetTerminator(self.to_raw())) }
    }

    pub fn insert_operation_after(&self, reference: &OperationRef, operation: Operation) {
        let operation_ref = unsafe { OperationRef::from_raw(operation.to_raw()) };
        unsafe {
            mlirBlockInsertOwnedOperationAfter(
                self.to_raw(),
                reference.to_raw(),
                operation_ref.to_raw(),
            )
        };
        forget(operation);
    }
}

impl<'c> PartialEq for BlockRef<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirBlockEqual(self.to_raw(), other.to_raw()) }
    }
}

impl<'c> PartialEq<Block<'c>> for BlockRef<'c> {
    fn eq(&self, other: &Block<'c>) -> bool {
        unsafe { mlirBlockEqual(self.to_raw(), other.to_raw()) }
    }
}

impl<'c> Eq for BlockRef<'c> {}

impl<'c> Display for BlockRef<'c> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut reader = StringReader::new(f);
        unsafe { mlirBlockPrint(self.to_raw(), reader.callback(), reader.as_raw_mut()) }
        Ok(())
    }
}

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
