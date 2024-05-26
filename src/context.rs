use crate::{
    binding::{impl_owned_mlir_value, impl_unowned_mlir_value, UnownedMlirValue},
    DialectRef, DialectRegistryRef, StringRef,
};

use std::{marker::PhantomData, ops::Deref};

use mlir_sys::{
    mlirContextAppendDialectRegistry, mlirContextCreateWithRegistry,
    mlirContextCreateWithThreading, mlirContextEnableMultithreading, mlirContextEqual,
    mlirContextGetAllowUnregisteredDialects, mlirContextGetNumLoadedDialects,
    mlirContextGetNumRegisteredDialects, mlirContextGetOrLoadDialect,
    mlirContextIsRegisteredOperation, mlirContextLoadAllAvailableDialects,
    mlirContextSetAllowUnregisteredDialects, MlirContext,
};

/// [Context] wraps the `mlir::MLIRContext` class, the top-level object for a collection of MLIR
/// operations.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirContextAppendDialectRegistry`
/// - `mlirContextCreateWithRegistry`
/// - `mlirContextCreateWithThreading`
/// - `mlirContextDestroy`
/// - `mlirContextEnableMultithreading`
/// - `mlirContextEqual`
/// - `mlirContextGetAllowUnregisteredDialects`
/// - `mlirContextGetNumLoadedDialects`
/// - `mlirContextGetNumRegisteredDialects`
/// - `mlirContextGetOrLoadDialect`
/// - `mlirContextIsRegisteredOperation`
/// - `mlirContextLoadAllAvailableDialects`
/// - `mlirContextSetAllowUnregisteredDialects`
///
/// The following bindings are not used/supported:
/// - `mlirContextAttachDiagnosticHandler`
/// - `mlirContextCreate`
/// - `mlirContextDetachDiagnosticHandler`
/// - `mlirContextSetThreadPool`
#[repr(transparent)]
#[derive(Debug)]
pub struct Context {
    raw: MlirContext,
}

impl_owned_mlir_value!(Context, MlirContext);

impl Context {
    /// Creates a new MLIR Context.
    ///
    /// # Arguments
    /// * `dialect_registry` - An optional reference to a [DialectRegistryRef] from which to
    /// pre-register dialects.
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

impl_unowned_mlir_value!(Context, ContextRef, MlirContext);

impl ContextRef {
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
    pub fn get_or_load_dialect<'a>(
        &self,
        dialect_name: impl Into<StringRef<'a>>,
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
    pub fn is_operation_registered<'a>(&self, operation_name: impl Into<StringRef<'a>>) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DialectRegistry;

    #[test]
    fn new() {
        let context = Context::new(None, false);
        assert_eq!(context.num_registered_dialects(), 1);
        assert_eq!(context.num_loaded_dialects(), 1);
    }

    #[test]
    fn new_with_registry() {
        let dialect_registry = DialectRegistry::default();
        dialect_registry.register_all_dialects();
        let context = Context::new(Some(&dialect_registry), false);
        assert_eq!(context.num_registered_dialects(), 42);
        assert_eq!(context.num_loaded_dialects(), 1);
    }

    #[test]
    fn allows_unregistered_dialects() {
        let context = Context::new(None, false);
        assert!(!context.allows_unregistered_dialects());
        context.set_allow_unregistered_dialects(true);
        assert!(context.allows_unregistered_dialects());
    }

    #[test]
    fn append_dialect_registry() {
        let context = Context::new(None, false);
        let dialect_registry = DialectRegistry::default();
        dialect_registry.register_all_dialects();
        context.append_dialect_registry(&dialect_registry);
        assert_eq!(context.num_registered_dialects(), 42);
        assert_eq!(context.num_loaded_dialects(), 1);
    }

    #[test]
    fn load_dialects() {
        let dialect_registry = DialectRegistry::default();
        dialect_registry.register_all_dialects();
        let context = Context::new(Some(&dialect_registry), false);
        context.load_all_available_dialects();
        assert_eq!(context.num_loaded_dialects(), 42);
    }

    #[test]
    fn get_dialect() {
        let context = Context::new(None, false);
        assert_eq!(
            context.get_or_load_dialect(&"builtin").unwrap().namespace(),
            "builtin"
        );
        assert!(context.get_or_load_dialect(&"func").is_none());

        let dialect_registry = DialectRegistry::default();
        dialect_registry.register_all_dialects();
        context.append_dialect_registry(&dialect_registry);
        assert_eq!(
            context.get_or_load_dialect(&"func").unwrap().namespace(),
            "func"
        );
    }

    #[test]
    fn is_registered_operation() {
        let context = Context::new(None, false);
        assert!(context.is_operation_registered(&"builtin.module"));
        assert!(!context.is_operation_registered(&"func.func"));

        let dialect_registry = DialectRegistry::default();
        dialect_registry.register_all_dialects();
        context.append_dialect_registry(&dialect_registry);
        assert!(!context.is_operation_registered(&"func.func"));

        context.load_all_available_dialects();
        assert!(context.is_operation_registered(&"func.func"));
    }

    #[test]
    fn compare_contexts() {
        let context1 = Context::new(None, false);
        let context2 = Context::new(None, false);

        assert_eq!(context1, context1);
        assert_eq!(context1.deref(), &context1);
        assert_eq!(&context1, context1.deref());
        assert_eq!(context1.deref(), context1.deref());

        assert_ne!(context1, context2);
        assert_ne!(context1.deref(), &context2);
        assert_ne!(&context1, context2.deref());
        assert_ne!(context1.deref(), context2.deref());

        assert_ne!(context2, context1);
        assert_ne!(context2.deref(), &context1);
        assert_ne!(&context2, context1.deref());
        assert_ne!(context2.deref(), context1.deref());

        assert_eq!(context2, context2);
        assert_eq!(context2.deref(), &context2);
        assert_eq!(&context2, context2.deref());
        assert_eq!(context2.deref(), context2.deref());
    }

    #[test]
    #[should_panic]
    fn no_owned_context_ref() {
        let _context_ref = ContextRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
