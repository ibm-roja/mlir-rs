use crate::{
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef, StringRef,
};

use std::marker::PhantomData;

use mlir_sys::{
    mlirIdentifierEqual, mlirIdentifierGet, mlirIdentifierGetContext, mlirIdentifierStr,
    MlirIdentifier,
};

/// [IdentifierRef] wraps the raw `MlirIdentifier` type from the MLIR C API, which represents a
/// string used as an identifier in the MLIR IR.
///
/// All relevant bindings into the MLIR C API are used/supported:
/// - `mlirIdentifierGet`
/// - `mlirIdentifierGetContext`
/// - `mlirIdentifierStr`
/// - `mlirIdentifierEqual`
#[repr(transparent)]
#[derive(Debug)]
pub struct IdentifierRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(IdentifierRef, MlirIdentifier);

impl IdentifierRef {
    /// Constructs a new [IdentifierRef] from the provided string.
    ///
    /// # Arguments
    /// * `context` - The context that should own the identifier.
    /// * `string` - The string to use as the identifier.
    ///
    /// # Returns
    /// Returns a new [IdentifierRef] reference.
    pub fn new<'a>(context: &'a ContextRef, string: &str) -> &'a Self {
        unsafe {
            Self::from_raw(mlirIdentifierGet(
                context.to_raw(),
                StringRef::from(&string).to_raw(),
            ))
        }
    }

    /// # Returns
    /// Returns a reference to the context that owns the identifier.
    pub fn context(&self) -> &ContextRef {
        unsafe { ContextRef::from_raw(mlirIdentifierGetContext(self.to_raw())) }
    }

    /// # Returns
    /// Returns the string value of the identifier.
    pub fn value(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirIdentifierStr(self.to_raw())) }
    }
}

impl PartialEq for IdentifierRef {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirIdentifierEqual(self.to_raw(), other.to_raw()) }
    }
}

impl Eq for IdentifierRef {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn context() {
        let context = Context::new(None, false);
        let identifier = IdentifierRef::new(&context, "test");
        assert_eq!(identifier.context(), &context);
    }

    #[test]
    fn value() {
        let context = Context::new(None, false);
        let identifier = IdentifierRef::new(&context, "test");
        assert_eq!(identifier.value(), "test");
    }

    #[test]
    fn compare_identifiers() {
        let context = Context::new(None, false);
        let identifier1 = IdentifierRef::new(&context, "test");
        let identifier2 = IdentifierRef::new(&context, "test");
        let identifier3 = IdentifierRef::new(&context, "other");
        assert_eq!(identifier1, identifier1);
        assert_eq!(identifier1, identifier2);
        assert_ne!(identifier1, identifier3);
        assert_ne!(identifier2, identifier3);
        assert_eq!(identifier3, identifier3);
    }

    #[test]
    #[should_panic]
    fn no_owned_identifier_ref() {
        let _identifier_ref = IdentifierRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
