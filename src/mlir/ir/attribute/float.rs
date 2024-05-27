use super::impl_attribute_variant;
use crate::{
    ir::FloatTypeRef,
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef,
};

use std::marker::PhantomData;

use mlir_sys::{
    mlirAttributeIsAFloat, mlirFloatAttrDoubleGet, mlirFloatAttrGetValueDouble, MlirAttribute,
};

/// [FloatAttributeRef] is a reference to an instance of the `mlir::FloatAttr` class, which
/// represents a constant floating-point value in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirAttributeIsAFloat`
/// - `mlirFloatAttrDoubleGet`
/// - `mlirFloatAttrGetValueDouble`
///
/// The following bindings are not used/supported:
/// - `mlirFloatAttrDoubleGetChecked`
/// - `mlirFloatAttrGetTypeID`
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct FloatAttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(no_refs, FloatAttributeRef, MlirAttribute);
impl_attribute_variant!(FloatAttributeRef, mlirAttributeIsAFloat);

impl FloatAttributeRef {
    /// Constructs a new float attribute of the specified type with the provided value.
    ///
    /// # Arguments
    /// * `ty` - The type of the float attribute.
    /// * `value` - The float value to hold in the attribute.
    ///
    /// # Returns
    /// Returns a reference to a new [FloatAttributeRef] instance.
    pub fn new<'a>(context: &'a ContextRef, ty: &'a FloatTypeRef, value: f64) -> &'a Self {
        unsafe { Self::from_raw(mlirFloatAttrDoubleGet(context.to_raw(), ty.to_raw(), value)) }
    }

    /// # Returns
    /// Returns the value held by the attribute as a 64-bit float.
    pub fn value(&self) -> f64 {
        unsafe { mlirFloatAttrGetValueDouble(self.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::AttributeRef, Context};

    #[test]
    fn value() {
        let context = Context::new(None, false);
        let float_type = FloatTypeRef::new_f64(&context);
        let float_attribute = FloatAttributeRef::new(&context, float_type, 6.29);
        assert_eq!(float_attribute.value(), 6.29);
    }

    #[test]
    fn from_attribute() {
        let context = Context::new(None, false);
        let erased_float_attribute = AttributeRef::parse(&context, "3.14 : f64").unwrap();
        let erased_integer_attribute = AttributeRef::parse(&context, "3 : i64").unwrap();
        assert!(FloatAttributeRef::try_from_attribute(erased_float_attribute).is_some());
        assert!(FloatAttributeRef::try_from_attribute(erased_integer_attribute).is_none());
    }

    #[test]
    #[should_panic]
    fn no_owned_float_attribute_ref() {
        let _float_attribute_ref = FloatAttributeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
