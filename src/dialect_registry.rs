use crate::binding::{impl_owned_mlir_value, impl_unowned_mlir_value, UnownedMlirValue};

use std::{marker::PhantomData, ops::Deref};

use mlir_sys::{
    mlirDialectRegistryCreate, mlirDialectRegistryDestroy, mlirRegisterAllDialects,
    MlirDialectRegistry,
};

/// [DialectRegistry] wraps the `mlir::DialectRegistry` class, which holds mappings from dialect
/// namespaces to the constructors for their matching dialect.
///
/// All relevant bindings into the MLIR C API are used/supported:
/// - `mlirDialectRegistryCreate`
/// - `mlirDialectRegistryDestroy`
/// - `mlirRegisterAllDialects`
#[repr(transparent)]
#[derive(Debug)]
pub struct DialectRegistry {
    raw: MlirDialectRegistry,
}

impl_owned_mlir_value!(DialectRegistry, MlirDialectRegistry);

impl Default for DialectRegistry {
    fn default() -> Self {
        Self {
            raw: unsafe { mlirDialectRegistryCreate() },
        }
    }
}

impl Drop for DialectRegistry {
    fn drop(&mut self) {
        unsafe { mlirDialectRegistryDestroy(self.raw) }
    }
}

/// [DialectRegistryRef] is a reference to an instance of the `mlir::DialectRegistry` class.
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct DialectRegistryRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(DialectRegistry, DialectRegistryRef, MlirDialectRegistry);

impl DialectRegistryRef {
    /// Registers all dialects known to MLIR with the dialect registry.
    pub fn register_all_dialects(&self) {
        unsafe { mlirRegisterAllDialects(self.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn empty_registry() {
        let dialect_registry = DialectRegistry::default();
        let context = Context::new(Some(&dialect_registry), false);
        assert_eq!(context.num_registered_dialects(), 1);
    }

    #[test]
    fn all_available_dialects() {
        let dialect_registry = DialectRegistry::default();
        dialect_registry.register_all_dialects();
        let context = Context::new(Some(&dialect_registry), false);
        assert_eq!(context.num_registered_dialects(), 42);
    }

    #[test]
    #[should_panic]
    fn no_owned_registry_ref() {
        let _registry_ref = DialectRegistryRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
