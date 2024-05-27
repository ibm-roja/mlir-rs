use super::impl_type_variant;
use crate::{
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef,
};

use std::marker::PhantomData;

use mlir_sys::{mlirNoneTypeGet, mlirTypeIsANone, MlirType};

/// [NoneTypeRef] is a reference to an instance of the `mlir::NoneType` class, which represents a
/// unit type in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirNoneTypeGet`
/// - `mlirTypeIsANone`
///
/// The following bindings are not used/supported:
/// - `mlirNoneTypeGetTypeID`
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct NoneTypeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(no_refs, NoneTypeRef, MlirType);
impl_type_variant!(NoneTypeRef, mlirTypeIsANone);

impl NoneTypeRef {
    /// Constructs a new none type.
    ///
    /// # Arguments
    /// * `context` - The context that should own the none type.
    ///
    /// # Returns
    /// Returns a reference to a new [NoneTypeRef] instance.
    pub fn new(context: &ContextRef) -> &Self {
        unsafe { Self::from_raw(mlirNoneTypeGet(context.to_raw())) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::TypeRef, Context};

    #[test]
    fn dialect() {
        let context = Context::new(None, false);
        let none_type = NoneTypeRef::new(&context);
        assert_eq!(none_type.dialect().namespace(), "builtin");
    }

    #[test]
    fn from_type() {
        let context = Context::new(None, false);
        let erased_unit_type = TypeRef::parse(&context, "none").unwrap();
        let erased_integer_type = TypeRef::parse(&context, "i32").unwrap();
        assert!(NoneTypeRef::try_from_type(erased_unit_type).is_some());
        assert!(NoneTypeRef::try_from_type(erased_integer_type).is_none());
    }

    #[test]
    #[should_panic]
    fn no_owned_none_type_ref() {
        let _none_type_ref = NoneTypeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
