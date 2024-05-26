use super::impl_attribute_variant;
use crate::{
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef,
};

use std::{marker::PhantomData, os::raw::c_int};

use mlir_sys::{mlirAttributeIsABool, mlirBoolAttrGet, mlirBoolAttrGetValue, MlirAttribute};

/// [BoolAttributeRef] is a reference to an instance of the `mlir::BoolAttr` class, which represents
/// a constant boolean value in the MLIR IR.
///
/// All relevant bindings into the MLIR C API are used/supported:
/// - `mlirAttributeIsABool`
/// - `mlirBoolAttrGet`
/// - `mlirBoolAttrGetValue`
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct BoolAttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(BoolAttributeRef, MlirAttribute);
impl_attribute_variant!(BoolAttributeRef, mlirAttributeIsABool);

impl BoolAttributeRef {
    /// Constructs a new boolean attribute with the provided value.
    ///
    /// # Arguments
    /// * `context` - The context that should own the attribute.
    /// * `value` - The boolean value to hold in the attribute.
    ///
    /// # Returns
    /// Returns a reference to a new [BoolAttributeRef] instance.
    pub fn new(context: &ContextRef, value: bool) -> &Self {
        unsafe { Self::from_raw(mlirBoolAttrGet(context.to_raw(), value as c_int)) }
    }

    /// # Returns
    /// Returns the boolean value held by the attribute.
    pub fn value(&self) -> bool {
        unsafe { mlirBoolAttrGetValue(self.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::AttributeRef, Context};

    #[test]
    fn value() {
        let context = Context::new(None, false);
        let attr = BoolAttributeRef::new(&context, true);
        assert!(attr.value());
    }

    #[test]
    fn from_attribute() {
        let context = Context::new(None, false);
        let erased_bool_attribute = AttributeRef::parse(&context, "true").unwrap();
        let erased_int_attribute = AttributeRef::parse(&context, "64 : i32").unwrap();
        assert!(BoolAttributeRef::try_from_attribute(erased_bool_attribute).is_some());
        assert!(BoolAttributeRef::try_from_attribute(erased_int_attribute).is_none());
    }

    #[test]
    #[should_panic]
    fn no_owned_bool_attribute_ref() {
        let _bool_attribute_ref = BoolAttributeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
