use crate::{ContextRef, StringRef};

use std::{marker::PhantomData, mem::transmute};

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

impl DialectRef {
    /// Constructs a reference to a [DialectRef] from the provided raw [MlirDialect] value.
    ///
    /// # Safety
    /// The caller of this function is responsible for associating the reference to the [DialectRef]
    /// with a lifetime that is bound to its owner, and ensuring that the provided raw [MlirDialect]
    /// value is valid.
    ///
    /// # Arguments
    /// * `context` - The raw [MlirDialect] value.
    ///
    /// # Returns
    /// Returns a new [DialectRef] instance.
    pub unsafe fn from_raw<'a>(dialect: MlirDialect) -> &'a Self {
        transmute(dialect)
    }

    /// # Returns
    /// Returns the reference to the [DialectRef] as an [MlirDialect].
    pub fn to_raw(&self) -> MlirDialect {
        unsafe { transmute(self) }
    }

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

impl Drop for DialectRef {
    fn drop(&mut self) {
        panic!("Owned instances of DialectRef should never be created!")
    }
}

impl PartialEq for DialectRef {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirDialectEqual(self.to_raw(), other.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, DialectHandle};

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
