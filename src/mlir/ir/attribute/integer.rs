use super::impl_attribute_variant;
use crate::{
    ir::IntegerTypeRef,
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
};

use std::marker::PhantomData;

use mlir_sys::{
    mlirAttributeIsAInteger, mlirIntegerAttrGet, mlirIntegerAttrGetValueSInt,
    mlirIntegerAttrGetValueUInt, MlirAttribute,
};

/// [IntegerAttributeRef] is a reference to an instance of the `mlir::IntegerAttr` class, which
/// represents a constant integer value in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirAttributeIsAInteger`
/// - `mlirIntegerAttrGet`
/// - `mlirIntegerAttrGetValueSInt`
/// - `mlirIntegerAttrGetValueUInt`
///
/// The following bindings are not used/supported:
/// - `mlirIntegerAttrGetValueInt`
/// - `mlirIntegerAttrGetTypeID`
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct IntegerAttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(IntegerAttributeRef, MlirAttribute);
impl_attribute_variant!(IntegerAttributeRef, mlirAttributeIsAInteger);

impl IntegerAttributeRef {
    /// Constructs a new integer attribute of the specified type with the provided value. The
    /// attribute is owned by the same context that owns its type.
    ///
    /// # Arguments
    /// * `ty` - The type of the integer attribute.
    /// * `value` - The integer value to hold in the attribute.
    ///
    /// # Returns
    /// Returns a reference to a new [IntegerAttributeRef] instance.
    pub fn new(ty: &IntegerTypeRef, value: i64) -> &Self {
        unsafe { Self::from_raw(mlirIntegerAttrGet(ty.to_raw(), value)) }
    }

    /// # Returns
    /// Returns the integer value held by the attribute as a signed 64-bit integer.
    pub fn value_signed(&self) -> i64 {
        unsafe { mlirIntegerAttrGetValueSInt(self.to_raw()) }
    }

    /// # Returns
    /// Returns the integer value held by the attribute as an unsigned 64-bit integer.
    pub fn value_unsigned(&self) -> u64 {
        unsafe { mlirIntegerAttrGetValueUInt(self.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::AttributeRef, Context};

    #[test]
    fn value() {
        let context = Context::new(None, false);
        let integer_type = IntegerTypeRef::new_signless(&context, 32);
        let integer_attribute = IntegerAttributeRef::new(integer_type, 42);
        assert_eq!(integer_attribute.value_signed(), 42);
        assert_eq!(integer_attribute.value_unsigned(), 42);
    }

    #[test]
    fn from_attribute() {
        let context = Context::new(None, false);
        let erased_integer_attribute = AttributeRef::parse(&context, "42 : i32").unwrap();
        let integer_attribute =
            IntegerAttributeRef::try_from_attribute(erased_integer_attribute).unwrap();
        assert_eq!(integer_attribute.value_signed(), 42);
    }

    #[test]
    #[should_panic]
    fn no_owned_integer_attribute_ref() {
        let _integer_attribute_ref = IntegerAttributeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
