use super::impl_attribute_variant;
use crate::{
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef, StringRef,
};

use std::marker::PhantomData;

use mlir_sys::{mlirAttributeIsAString, mlirStringAttrGet, mlirStringAttrGetValue, MlirAttribute};

/// [StringAttributeRef] is a reference to an instance of the `mlir::StringAttr` class, which
/// represents a constant string value in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirAttributeIsAString`
/// - `mlirStringAttrGet`
/// - `mlirStringAttrGetValue`
///
/// The following bindings are not used/supported:
/// - `mlirStringAttrGetTypeID`
/// - `mlirStringAttrTypedGet`
#[repr(transparent)]
#[derive(Debug)]
pub struct StringAttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(StringAttributeRef, MlirAttribute);
impl_attribute_variant!(StringAttributeRef, mlirAttributeIsAString);

impl StringAttributeRef {
    /// Constructs a new string attribute with the provided value.
    ///
    /// # Arguments
    /// * `context` - The context that should own the attribute.
    /// * `value` - The string value to hold in the attribute.
    ///
    /// # Returns
    /// Returns a reference to a new [StringAttributeRef] instance.
    pub fn new<'a>(context: &'a ContextRef, value: &str) -> &'a Self {
        unsafe {
            Self::from_raw(mlirStringAttrGet(
                context.to_raw(),
                StringRef::from(&value).to_raw(),
            ))
        }
    }

    /// # Returns
    /// Returns the string value held by the attribute.
    pub fn value(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirStringAttrGetValue(self.to_raw())) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::AttributeRef, Context};

    #[test]
    fn value() {
        let context = Context::new(None, false);
        let attr = StringAttributeRef::new(&context, "hello");
        assert_eq!(attr.value(), "hello");
    }

    #[test]
    fn from_attribute() {
        let context = Context::new(None, false);
        let erased_attribute = AttributeRef::parse(&context, r#""hello, world""#).unwrap();
        let string_attribute = StringAttributeRef::try_from_attribute(erased_attribute).unwrap();
        assert_eq!(string_attribute.value(), "hello, world");
        assert!(StringAttributeRef::try_from_attribute(
            AttributeRef::parse(&context, "64 : i32").unwrap()
        )
        .is_none());
    }

    #[test]
    #[should_panic]
    fn no_owned_string_attribute_ref() {
        let _string_attribute_ref = StringAttributeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
