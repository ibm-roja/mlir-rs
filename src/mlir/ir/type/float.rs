use super::impl_type_variant;
use crate::{
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef,
};

use std::marker::PhantomData;

use mlir_sys::{mlirF32TypeGet, mlirF64TypeGet, mlirTypeIsAF32, mlirTypeIsAF64, MlirType};

/// [FloatTypeRef] is a reference to an instance of the `mlir::FloatType` class, which represents a
/// floating-point type in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirF32TypeGet`
/// - `mlirF64TypeGet`
/// - `mlirTypeIsAF32`
/// - `mlirTypeIsAF64`
///
/// The following bindings are not used/supported:
/// - `mlirF16TypeGet`
/// - `mlirFloat16GetTypeID`
/// - `mlirFloat32GetTypeID`
/// - `mlirFloat64GetTypeID`
/// - `mlirTypeIsAF16`
/// - (functions for other variants such as BF16, TF32, ...)
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct FloatTypeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

/// Checks if the given raw MLIR type is a `f32` or `f64` type.
///
/// # Safety
/// The given raw MLIR type must point to a valid MLIR type instance.
///
/// # Arguments
/// * `ty` - The raw MLIR type to check.
///
/// # Returns
/// Returns whether the type is a floating-point type.
unsafe fn mlir_type_is_a_float(ty: MlirType) -> bool {
    mlirTypeIsAF32(ty) || mlirTypeIsAF64(ty)
}

impl_unowned_mlir_value!(no_refs, FloatTypeRef, MlirType);
impl_type_variant!(FloatTypeRef, mlir_type_is_a_float);

impl FloatTypeRef {
    /// Constructs a new 32-bit floating point type.
    ///
    /// # Arguments
    /// * `context` - The context that should own the type.
    ///
    /// # Returns
    /// Returns a reference to a new [FloatTypeRef] instance.
    pub fn new_f32(context: &ContextRef) -> &Self {
        unsafe { Self::from_raw(mlirF32TypeGet(context.to_raw())) }
    }

    /// Constructs a new 64-bit floating point type.
    ///
    /// # Arguments
    /// * `context` - The context that should own the type.
    ///
    /// # Returns
    /// Returns a reference to a new [FloatTypeRef] instance.
    pub fn new_f64(context: &ContextRef) -> &Self {
        unsafe { Self::from_raw(mlirF64TypeGet(context.to_raw())) }
    }

    /// # Returns
    /// Returns the bitwidth of the floating point type.
    pub fn bitwidth(&self) -> u32 {
        if unsafe { mlirTypeIsAF32(self.to_raw()) } {
            32
        } else if unsafe { mlirTypeIsAF64(self.to_raw()) } {
            64
        } else {
            unreachable!("Invalid floating point type.")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::TypeRef, Context};

    #[test]
    fn bitwidth() {
        let context = Context::new(None, false);
        let f32_type = FloatTypeRef::new_f32(&context);
        let f64_type = FloatTypeRef::new_f64(&context);
        assert_eq!(f32_type.bitwidth(), 32);
        assert_eq!(f64_type.bitwidth(), 64);
    }

    #[test]
    fn from_type() {
        let context = Context::new(None, false);
        let erased_f32_type = TypeRef::parse(&context, "f32").unwrap();
        let erased_f64_type = TypeRef::parse(&context, "f64").unwrap();
        let erased_i32_type = TypeRef::parse(&context, "i32").unwrap();
        assert!(FloatTypeRef::try_from_type(erased_f32_type).is_some());
        assert!(FloatTypeRef::try_from_type(erased_f64_type).is_some());
        assert!(FloatTypeRef::try_from_type(erased_i32_type).is_none());
    }

    #[test]
    #[should_panic]
    fn no_owned_float_type_ref() {
        let _float_type_ref = FloatTypeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
