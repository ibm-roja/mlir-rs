use super::impl_type_variant;
use crate::{
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef,
};

use std::marker::PhantomData;

use mlir_sys::{
    mlirIntegerTypeGet, mlirIntegerTypeGetWidth, mlirIntegerTypeIsSigned,
    mlirIntegerTypeIsSignless, mlirIntegerTypeIsUnsigned, mlirIntegerTypeSignedGet,
    mlirIntegerTypeUnsignedGet, mlirTypeIsAInteger, MlirType,
};

/// [IntegerTypeRef] is a reference to an instance of the `mlir::IntegerType` class, which
/// represents an n-bit integer type in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirIntegerTypeGetWidth`
/// - `mlirIntegerTypeGet`
/// - `mlirIntegerTypeIsSigned`
/// - `mlirIntegerTypeIsSignless`
/// - `mlirIntegerTypeIsUnsigned`
/// - `mlirIntegerTypeSignedGet`
/// - `mlirIntegerTypeUnsignedGet`
/// - `mlirTypeIsAInteger`
///
/// The following bindings are not used/supported:
/// - `mlirIntegerTypeGetTypeID`
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct IntegerTypeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(IntegerTypeRef, MlirType);
impl_type_variant!(IntegerTypeRef, mlirTypeIsAInteger);

impl IntegerTypeRef {
    /// Constructs a new signless integer type with the provided bitwidth.
    ///
    /// # Arguments
    /// * `context` - The context that should own the integer type.
    /// * `bitwidth` - The bitwidth of the integer type.
    ///
    /// # Returns
    /// Returns a reference to a new [IntegerTypeRef] instance.
    pub fn new_signless(context: &ContextRef, bitwidth: u32) -> &Self {
        unsafe { Self::from_raw(mlirIntegerTypeGet(context.to_raw(), bitwidth)) }
    }

    /// Constructs a new signed integer type with the provided bitwidth.
    ///
    /// # Arguments
    /// * `context` - The context that should own the integer type.
    /// * `bitwidth` - The bitwidth of the integer type.
    ///
    /// # Returns
    /// Returns a reference to a new [IntegerTypeRef] instance.
    pub fn new_signed(context: &ContextRef, bitwidth: u32) -> &Self {
        unsafe { Self::from_raw(mlirIntegerTypeSignedGet(context.to_raw(), bitwidth)) }
    }

    /// Constructs a new unsigned integer type with the provided bitwidth.
    ///
    /// # Arguments
    /// * `context` - The context that should own the integer type.
    /// * `bitwidth` - The bitwidth of the integer type.
    ///
    /// # Returns
    /// Returns a reference to a new [IntegerTypeRef] instance.
    pub fn new_unsigned(context: &ContextRef, bitwidth: u32) -> &Self {
        unsafe { Self::from_raw(mlirIntegerTypeUnsignedGet(context.to_raw(), bitwidth)) }
    }

    /// # Returns
    /// Returns the bitwidth of the integer type.
    pub fn bitwidth(&self) -> u32 {
        unsafe { mlirIntegerTypeGetWidth(self.to_raw()) }
    }

    /// # Returns
    /// Returns whether the integer type is signless.
    pub fn is_signless(&self) -> bool {
        unsafe { mlirIntegerTypeIsSignless(self.to_raw()) }
    }

    /// # Returns
    /// Returns whether the integer type is signed.
    pub fn is_signed(&self) -> bool {
        unsafe { mlirIntegerTypeIsSigned(self.to_raw()) }
    }

    /// # Returns
    /// Returns whether the integer type is unsigned.
    pub fn is_unsigned(&self) -> bool {
        unsafe { mlirIntegerTypeIsUnsigned(self.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::TypeRef, Context};

    #[test]
    fn dialect() {
        let context = Context::new(None, false);
        let integer_type = IntegerTypeRef::new_signless(&context, 32);
        assert_eq!(integer_type.dialect().namespace(), "builtin");
    }

    #[test]
    fn from_type() {
        let context = Context::new(None, false);
        let erased_integer_type = TypeRef::parse(&context, "i32").unwrap();
        let erased_unit_type = TypeRef::parse(&context, "none").unwrap();
        assert!(IntegerTypeRef::try_from_type(erased_integer_type).is_some());
        assert!(IntegerTypeRef::try_from_type(erased_unit_type).is_none());
    }

    #[test]
    fn bitwidth() {
        let context = Context::new(None, false);
        assert_eq!(IntegerTypeRef::new_signless(&context, 16).bitwidth(), 16);
        assert_eq!(IntegerTypeRef::new_signless(&context, 32).bitwidth(), 32);
        assert_eq!(IntegerTypeRef::new_signless(&context, 64).bitwidth(), 64);
        assert_eq!(IntegerTypeRef::new_signless(&context, 128).bitwidth(), 128);
    }

    #[test]
    fn signless() {
        let context = Context::new(None, false);
        let ty = IntegerTypeRef::new_signless(&context, 64);
        assert!(ty.is_signless());
        assert!(!ty.is_signed());
        assert!(!ty.is_unsigned());
    }

    #[test]
    fn signed() {
        let context = Context::new(None, false);
        let ty = IntegerTypeRef::new_signed(&context, 64);
        assert!(!ty.is_signless());
        assert!(ty.is_signed());
        assert!(!ty.is_unsigned());
    }

    #[test]
    fn unsigned() {
        let context = Context::new(None, false);
        let ty = IntegerTypeRef::new_unsigned(&context, 64);
        assert!(!ty.is_signless());
        assert!(!ty.is_signed());
        assert!(ty.is_unsigned());
    }

    #[test]
    #[should_panic]
    fn no_owned_integer_type_ref() {
        let _integer_type_ref = IntegerTypeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
