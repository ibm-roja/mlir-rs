use crate::{
    support::binding::{impl_owned_mlir_value, OwnedMlirValue, UnownedMlirValue},
    ContextRef, DialectRef, DialectRegistryRef, StringRef,
};

use mlir_sys::{
    mlirDialectHandleGetNamespace, mlirDialectHandleInsertDialect, mlirDialectHandleLoadDialect,
    mlirDialectHandleRegisterDialect, MlirDialectHandle,
};

/// [DialectHandle] wraps the raw `MlirDialectHandle` type from the MLIR C API, which points to the
/// registration hooks for a dialect.
///
/// All relevant bindings into the MLIR C API are used/supported:
/// - `mlirDialectHandleGetNamespace`
/// - `mlirDialectHandleInsertDialect`
/// - `mlirDialectHandleRegisterDialect`
/// - `mlirDialectHandleLoadDialect`
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct DialectHandle {
    raw: MlirDialectHandle,
}

impl_owned_mlir_value!(no_refs, DialectHandle, MlirDialectHandle);

impl DialectHandle {
    /// # Returns
    /// Returns the namespace of the dialect.
    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectHandleGetNamespace(self.to_raw())) }
    }

    /// Inserts the dialect into the provided dialect registry.
    ///
    /// # Arguments
    /// * `dialect_registry` - The [DialectRegistryRef] to insert the dialect into.
    pub fn insert_into_registry(&self, dialect_registry_ref: &DialectRegistryRef) {
        unsafe {
            mlirDialectHandleInsertDialect(self.to_raw(), dialect_registry_ref.to_raw());
        }
    }

    /// Registers the dialect with the provided context.
    ///
    /// # Arguments
    /// * `context` - The [ContextRef] to register the dialect with.
    pub fn register_with_context(&self, context: &ContextRef) {
        unsafe {
            mlirDialectHandleRegisterDialect(self.to_raw(), context.to_raw());
        }
    }

    /// Loads the dialect into the provided context.
    ///
    /// # Arguments
    /// * `context` - The [ContextRef] to load the dialect into.
    ///
    /// # Returns
    /// Returns a reference to the dialect that was loaded.
    pub fn load_into_context<'a>(&self, context: &'a ContextRef) -> &'a DialectRef {
        unsafe {
            DialectRef::from_raw(mlirDialectHandleLoadDialect(
                self.to_raw(),
                context.to_raw(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, DialectRegistry};

    use mlir_sys::mlirGetDialectHandle__func__;

    #[test]
    fn namespace() {
        let dialect_handle = unsafe { DialectHandle::from_raw(mlirGetDialectHandle__func__()) };
        assert_eq!(dialect_handle.namespace(), "func");
    }

    #[test]
    fn insert_into_registry() {
        let dialect_handle = unsafe { DialectHandle::from_raw(mlirGetDialectHandle__func__()) };
        let dialect_registry = DialectRegistry::default();
        dialect_handle.insert_into_registry(&dialect_registry);

        let context = Context::new(Some(&dialect_registry), false);
        assert_eq!(context.num_registered_dialects(), 2);
        assert_eq!(context.num_loaded_dialects(), 1);
    }

    #[test]
    fn register_with_context() {
        let dialect_handle = unsafe { DialectHandle::from_raw(mlirGetDialectHandle__func__()) };
        let context = Context::new(None, false);
        dialect_handle.register_with_context(&context);
        assert_eq!(context.num_registered_dialects(), 2);
        assert_eq!(context.num_loaded_dialects(), 1);
    }

    #[test]
    fn load_into_context() {
        let dialect_handle = unsafe { DialectHandle::from_raw(mlirGetDialectHandle__func__()) };
        let context = Context::new(None, false);
        let dialect = dialect_handle.load_into_context(&context);
        assert_eq!(context.num_registered_dialects(), 1);
        assert_eq!(context.num_loaded_dialects(), 2);
        assert_eq!(dialect.namespace(), "func");
    }
}
