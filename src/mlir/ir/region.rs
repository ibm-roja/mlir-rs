use crate::{
    ir::{Block, BlockRef},
    support::binding::{
        impl_owned_mlir_value, impl_unowned_mlir_value, OwnedMlirValue, UnownedMlirValue,
    },
    ContextRef,
};

use std::marker::PhantomData;

use mlir_sys::{
    mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirRegionDestroy, mlirRegionEqual,
    mlirRegionGetFirstBlock, mlirRegionGetNextInOperation, MlirRegion,
};

/// [Region] wraps the `mlir::Region` class, which represents a region of blocks in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirRegionAppendOwnedBlock`
/// - `mlirRegionCreate`
/// - `mlirRegionDestroy`
/// - `mlirRegionEqual`
/// - `mlirRegionGetFirstBlock`
/// - `mlirRegionGetNextInOperation`
///
/// The following bindings are not used/supported:
/// - `mlirRegionInsertOwnedBlock`
/// - `mlirRegionInsertOwnedBlockAfter`
/// - `mlirRegionInsertOwnedBlockBefore`
/// - `mlirRegionTakeBody`
#[repr(transparent)]
#[derive(Debug)]
pub struct Region<'c> {
    raw: MlirRegion,
    _context: PhantomData<&'c ()>,
}

impl_owned_mlir_value!(context_ref, Region, MlirRegion);

impl<'c> Region<'c> {
    /// Creates a new region.
    ///
    /// # Arguments
    /// * `context` - The context that should be associated with the region.
    ///
    /// # Returns
    /// Returns a new [Region] instance.
    pub fn new(_context: &'c ContextRef) -> Region<'c> {
        unsafe { Self::from_raw(mlirRegionCreate()) }
    }
}

impl<'c> Drop for Region<'c> {
    fn drop(&mut self) {
        unsafe { mlirRegionDestroy(self.raw) }
    }
}

impl<'c> PartialEq for Region<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl<'c> PartialEq<RegionRef<'c>> for Region<'c> {
    fn eq(&self, other: &RegionRef<'c>) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.to_raw()) }
    }
}

impl<'c> Eq for Region<'c> {}

/// [RegionRef] is a reference to an instance of the `mlir::Region` class, which represents a region
/// of blocks in the MLIR IR.
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct RegionRef<'c> {
    _context: PhantomData<&'c ()>,
}

impl_unowned_mlir_value!(context_ref, Region, RegionRef, MlirRegion);

impl<'c> RegionRef<'c> {
    /// Appends the given block to the region.
    ///
    /// # Arguments
    /// * `block` - The block to append to the region.
    ///
    /// # Returns
    /// Returns a reference to the appended block owned by the region.
    pub fn append_block<'a>(&'a self, block: Block<'c>) -> &'a BlockRef<'c> {
        let block_ref = unsafe { BlockRef::from_raw(block.to_raw()) };
        unsafe { mlirRegionAppendOwnedBlock(self.to_raw(), block_ref.to_raw()) };
        block_ref
    }

    /// # Returns
    /// Returns the first block of the region, if it has one.
    pub fn first_block(&self) -> Option<&BlockRef<'c>> {
        unsafe { BlockRef::try_from_raw(mlirRegionGetFirstBlock(self.to_raw())) }
    }

    /// # Returns
    /// If this region is in an operation and there is another region after it in the operation,
    /// returns that region.
    pub fn next_in_parent_operation(&self) -> Option<&RegionRef<'c>> {
        unsafe { Self::try_from_raw(mlirRegionGetNextInOperation(self.to_raw())) }
    }
}

impl<'c> PartialEq for RegionRef<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.to_raw(), other.to_raw()) }
    }
}

impl<'c> PartialEq<Region<'c>> for RegionRef<'c> {
    fn eq(&self, other: &Region<'c>) -> bool {
        unsafe { mlirRegionEqual(self.to_raw(), other.to_raw()) }
    }
}

impl<'c> Eq for RegionRef<'c> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn no_owned_region_ref() {
        let _region_ref = RegionRef {
            _context: PhantomData,
        };
    }
}
