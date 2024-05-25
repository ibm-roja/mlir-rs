use std::{marker::PhantomData, mem::transmute, ops::Deref};

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

impl DialectRegistry {
    /// # Returns
    /// Returns the raw [MlirDialectRegistry] value.
    pub fn to_raw(&self) -> MlirDialectRegistry {
        self.raw
    }
}

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

impl Deref for DialectRegistry {
    type Target = DialectRegistryRef;

    fn deref(&self) -> &Self::Target {
        unsafe { DialectRegistryRef::from_raw(self.raw) }
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

impl DialectRegistryRef {
    /// Constructs a reference to a [DialectRegistryRef] from the provided raw [MlirDialectRegistry]
    /// value.
    ///
    /// # Safety
    /// The caller of this function is responsible for associating the reference to the
    /// [DialectRegistryRef] with a lifetime that is bound to its owner.
    ///
    /// # Arguments
    /// * `dialect_registry` - The raw [MlirDialectRegistry] value.
    ///
    /// # Returns
    /// Returns a new [DialectRegistryRef] instance.
    pub unsafe fn from_raw<'a>(dialect_registry: MlirDialectRegistry) -> &'a Self {
        transmute(dialect_registry)
    }

    /// # Returns
    /// Returns the reference to the [DialectRegistryRef] as an [MlirDialectRegistry].
    pub fn to_raw(&self) -> MlirDialectRegistry {
        unsafe { transmute(self) }
    }

    /// Registers all dialects known to MLIR with the dialect registry.
    pub fn register_all_dialects(&self) {
        unsafe { mlirRegisterAllDialects(self.to_raw()) }
    }
}

impl Drop for DialectRegistryRef {
    fn drop(&mut self) {
        panic!("Owned instances of DialectRegistryRef should never be created!")
    }
}
