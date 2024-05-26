use crate::{
    binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef, StringRef,
};

use std::marker::PhantomData;

use mlir_sys::{mlirDialectEqual, mlirDialectGetContext, mlirDialectGetNamespace, MlirDialect};

/// [DialectRef] is a reference to an instance of the `mlir::Dialect` class, which holds MLIR
/// operations, types, and attributes, as well as the behaviour associated with the whole group.
///
/// All relevant bindings into the MLIR C API are used/supported:
/// - `mlirDialectGetContext`
/// - `mlirDialectEqual`
/// - `mlirDialectGetNamespace`
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct DialectRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(DialectRef, MlirDialect);

impl DialectRef {
    /// # Returns
    /// Returns a reference to the context that owns the dialect.
    pub fn context(&self) -> &ContextRef {
        unsafe { ContextRef::from_raw(mlirDialectGetContext(self.to_raw())) }
    }

    /// # Returns
    /// Returns the namespace of the dialect.
    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectGetNamespace(self.to_raw())) }
    }
}

impl PartialEq for DialectRef {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirDialectEqual(self.to_raw(), other.to_raw()) }
    }
}

impl Eq for DialectRef {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{binding::OwnedMlirValue, Context, DialectHandle};

    use mlir_sys::{mlirGetDialectHandle__arith__, mlirGetDialectHandle__func__};

    #[test]
    fn context() {
        let dialect_handle = unsafe { DialectHandle::from_raw(mlirGetDialectHandle__func__()) };
        let context = Context::new(None, false);
        let dialect = dialect_handle.load_into_context(&context);
        assert_eq!(dialect.context(), &context);
    }

    #[test]
    fn namespace() {
        let dialect_handle = unsafe { DialectHandle::from_raw(mlirGetDialectHandle__func__()) };
        let context = Context::new(None, false);
        let dialect = dialect_handle.load_into_context(&context);
        assert_eq!(dialect.namespace(), "func");
    }

    #[test]
    fn compare_dialects() {
        let dialect_handle_func =
            unsafe { DialectHandle::from_raw(mlirGetDialectHandle__func__()) };
        let dialect_handle_arith =
            unsafe { DialectHandle::from_raw(mlirGetDialectHandle__arith__()) };
        let context = Context::new(None, false);
        let dialect_func = dialect_handle_func.load_into_context(&context);
        let dialect_arith = dialect_handle_arith.load_into_context(&context);
        assert_eq!(dialect_func, dialect_func);
        assert_ne!(dialect_func, dialect_arith);
        assert_ne!(dialect_arith, dialect_func);
        assert_eq!(dialect_arith, dialect_arith);
    }

    #[test]
    #[should_panic]
    fn no_owned_dialect_ref() {
        let _dialect_ref = DialectRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
