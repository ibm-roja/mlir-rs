use crate::support::binding::{
    impl_owned_mlir_value, impl_unowned_mlir_value, OwnedMlirValue, UnownedMlirValue,
};

use std::marker::PhantomData;

use mlir_sys::MlirRegion;

#[repr(transparent)]
#[derive(Debug)]
pub struct Region<'c> {
    raw: MlirRegion,
    _context: PhantomData<&'c ()>,
}

impl_owned_mlir_value!(context_ref, Region, MlirRegion);

#[repr(transparent)]
#[derive(Debug)]
pub struct RegionRef<'c> {
    _context: PhantomData<&'c ()>,
}

impl_unowned_mlir_value!(context_ref, Region, RegionRef, MlirRegion);

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
