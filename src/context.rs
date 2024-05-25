use crate::{DialectRef, DialectRegistryRef, StringRef};

use std::{marker::PhantomData, mem::transmute, ops::Deref};

use mlir_sys::{
    mlirContextAppendDialectRegistry, mlirContextCreateWithRegistry,
    mlirContextCreateWithThreading, mlirContextDestroy, mlirContextEnableMultithreading,
    mlirContextEqual, mlirContextGetAllowUnregisteredDialects, mlirContextGetNumLoadedDialects,
    mlirContextGetNumRegisteredDialects, mlirContextGetOrLoadDialect,
    mlirContextIsRegisteredOperation, mlirContextLoadAllAvailableDialects,
    mlirContextSetAllowUnregisteredDialects, MlirContext,
};

/// [Context] wraps the `mlir::MLIRContext` class, the top-level object for a collection of MLIR
/// operations.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirContextCreateWithThreading`
/// - `mlirContextCreateWithRegistry`
/// - `mlirContextEqual`
/// - `mlirContextDestroy`
/// - `mlirContextSetAllowUnregisteredDialects`
/// - `mlirContextGetAllowUnregisteredDialects`
/// - `mlirContextGetNumRegisteredDialects`
/// - `mlirContextAppendDialectRegistry`
/// - `mlirContextGetNumLoadedDialects`
/// - `mlirContextGetOrLoadDialect`
/// - `mlirContextEnableMultithreading`
/// - `mlirContextLoadAllAvailableDialects`
/// - `mlirContextIsRegisteredOperation`
///
/// The following bindings are not used/supported:
/// - `mlirContextCreate`
/// - `mlirContextSetThreadPool`
/// - `mlirContextAttachDiagnosticHandler`
/// - `mlirContextDetachDiagnosticHandler`
#[repr(transparent)]
#[derive(Debug)]
pub struct Context {
    raw: MlirContext,
}

impl Context {
    /// Creates a new MLIR Context.
    ///
    /// # Arguments
    /// * `dialect_registry` - An optional reference to a [DialectRegistryRef] from which to preload
    ///    dialects.
    /// * `threading_enabled` - Whether to enable multithreading.
    ///
    /// # Returns
    /// Returns a new [Context] instance.
    pub fn new(dialect_registry: Option<&DialectRegistryRef>, threading_enabled: bool) -> Self {
        let raw_context = match dialect_registry {
            Some(dialect_registry) => unsafe {
                mlirContextCreateWithRegistry(dialect_registry.to_raw(), threading_enabled)
            },
            None => unsafe { mlirContextCreateWithThreading(threading_enabled) },
        };
        Self { raw: raw_context }
    }

    /// # Returns
    /// Returns the raw [MlirContext] value.
    pub fn to_raw(&self) -> MlirContext {
        self.raw
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.raw, other.raw) }
    }
}

impl PartialEq<ContextRef> for Context {
    fn eq(&self, other: &ContextRef) -> bool {
        unsafe { mlirContextEqual(self.raw, other.to_raw()) }
    }
}

impl Eq for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { mlirContextDestroy(self.raw) }
    }
}

impl Deref for Context {
    type Target = ContextRef;

    fn deref(&self) -> &Self::Target {
        unsafe { ContextRef::from_raw(self.raw) }
    }
}

/// [ContextRef] is a reference to an instance of the `mlir::MLIRContext` class.
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct ContextRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl ContextRef {
    /// Constructs a reference to a [ContextRef] from the provided raw [MlirContext] value.
    ///
    /// # Safety
    /// The caller of this function is responsible for associating the reference to the [ContextRef]
    /// with a lifetime that is bound to its owner.
    ///
    /// # Arguments
    /// * `context` - The raw [MlirContext] value.
    ///
    /// # Returns
    /// Returns a new [ContextRef] instance.
    pub unsafe fn from_raw<'a>(context: MlirContext) -> &'a Self {
        transmute(context)
    }

    /// # Returns
    /// Returns the reference to the [ContextRef] as an [MlirContext].
    pub fn to_raw(&self) -> MlirContext {
        unsafe { transmute(self) }
    }

    /// # Returns
    /// Returns whether the context allows unregistered dialects.
    pub fn allows_unregistered_dialects(&self) -> bool {
        unsafe { mlirContextGetAllowUnregisteredDialects(self.to_raw()) }
    }

    /// Sets whether the context allows unregistered dialects.
    ///
    /// # Arguments
    /// * `allow` - Whether to allow unregistered dialects.
    pub fn set_allow_unregistered_dialects(&self, allow: bool) {
        unsafe { mlirContextSetAllowUnregisteredDialects(self.to_raw(), allow) }
    }

    /// # Returns
    /// Returns the number of dialects registered with the context.
    pub fn num_registered_dialects(&self) -> isize {
        unsafe { mlirContextGetNumRegisteredDialects(self.to_raw()) }
    }

    /// Appends the contents of the provided dialect registry to the registry associated with the
    /// context.
    ///
    /// # Arguments
    /// * `dialect_registry` - A reference to the dialect registry to append.
    pub fn append_dialect_registry(&self, dialect_registry: &DialectRegistryRef) {
        unsafe { mlirContextAppendDialectRegistry(self.to_raw(), dialect_registry.to_raw()) }
    }

    /// # Returns
    /// Returns the number of dialects loaded by the context.
    pub fn num_loaded_dialects(&self) -> isize {
        unsafe { mlirContextGetNumLoadedDialects(self.to_raw()) }
    }

    /// Gets the dialect with the given name, loading it if it has not been already.
    ///
    /// # Arguments
    /// * `dialect_name` - The name of the dialect to get or load.
    ///
    /// # Returns
    /// Returns the dialect with the given name.
    pub fn get_or_load_dialect(
        &self,
        dialect_name: impl for<'a> Into<StringRef<'a>>,
    ) -> Option<&DialectRef> {
        let dialect_name = dialect_name.into().to_raw();
        let raw_dialect = unsafe { mlirContextGetOrLoadDialect(self.to_raw(), dialect_name) };
        if raw_dialect.ptr.is_null() {
            None
        } else {
            Some(unsafe { DialectRef::from_raw(raw_dialect) })
        }
    }

    /// Sets the threading mode, enabling or disabling multithreading.
    ///
    /// # Arguments
    /// * `threading_enabled` - Whether to enable multithreading.
    pub fn set_threading_enabled(&self, threading_enabled: bool) {
        unsafe { mlirContextEnableMultithreading(self.to_raw(), threading_enabled) }
    }

    /// Eagerly loads all dialects registered with the context, making them available for use in IR
    /// construction.
    pub fn load_all_available_dialects(&self) {
        unsafe { mlirContextLoadAllAvailableDialects(self.to_raw()) }
    }

    /// Checks if the operation with the provided fully-qualified name (e.g. 'dialect.operation') is
    /// registered with, and its dialect is loaded in, the context.
    ///
    /// # Arguments
    /// * `operation_name` - The fully-qualified name of the operation to check.
    ///
    /// # Returns
    /// Returns whether the operation is registered and its dialect is loaded.
    pub fn is_operation_registered(
        &self,
        operation_name: impl for<'a> Into<StringRef<'a>>,
    ) -> bool {
        let operation_name = operation_name.into().to_raw();
        unsafe { mlirContextIsRegisteredOperation(self.to_raw(), operation_name) }
    }
}

impl PartialEq for ContextRef {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.to_raw(), other.to_raw()) }
    }
}

impl PartialEq<Context> for ContextRef {
    fn eq(&self, other: &Context) -> bool {
        unsafe { mlirContextEqual(self.to_raw(), other.raw) }
    }
}

impl Eq for ContextRef {}
