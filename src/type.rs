use crate::{
    binding::{impl_unowned_mlir_value, UnownedMlirValue},
    string_reader::StringReader,
    ContextRef, DialectRef, StringRef,
};

use std::{
    fmt::{Display, Formatter},
    marker::PhantomData,
};

use mlir_sys::{
    mlirNoneTypeGet, mlirTypeEqual, mlirTypeGetContext, mlirTypeGetDialect, mlirTypeParseGet,
    mlirTypePrint, MlirType,
};

/// [TypeRef] is a reference to an instance of the `mlir::Type` class, which represents a type in
/// the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirTypeEqual`
/// - `mlirTypeGetContext`
/// - `mlirTypeGetDialect`
/// - `mlirTypeNoneGet`
/// - `mlirTypeParseGet`
/// - `mlirTypePrint`
///
/// The following bindings are not used/supported:
/// - `mlirTypeDump`
/// - `mlirTypeGetTypeID`
#[repr(transparent)]
#[derive(Debug)]
pub struct TypeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(TypeRef, MlirType);

impl TypeRef {
    // TODO: Should this be in TypeRef?
    pub fn none(context: &ContextRef) -> &Self {
        unsafe { Self::from_raw(mlirNoneTypeGet(context.to_raw())) }
    }

    /// Attempts to parse a type from the provided string.
    ///
    /// # Arguments
    /// * `context` - The context that should own the type.
    /// * `ty` - The string to parse the type from.
    ///
    /// # Returns
    /// Returns a new [TypeRef] reference if the type could be parsed, otherwise `None`.
    pub fn parse<'a>(context: &'a ContextRef, ty: &str) -> Option<&'a Self> {
        unsafe {
            Self::try_from_raw(mlirTypeParseGet(
                context.to_raw(),
                StringRef::from(&ty).to_raw(),
            ))
        }
    }

    /// # Returns
    /// Returns the context that owns this type.
    pub fn context(&self) -> &ContextRef {
        unsafe { ContextRef::from_raw(mlirTypeGetContext(self.to_raw())) }
    }

    /// # Returns
    /// Returns the dialect of the type.
    pub fn dialect(&self) -> &DialectRef {
        unsafe { DialectRef::from_raw(mlirTypeGetDialect(self.to_raw())) }
    }
}

impl PartialEq for TypeRef {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirTypeEqual(self.to_raw(), other.to_raw()) }
    }
}

impl Eq for TypeRef {}

impl Display for TypeRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut reader = StringReader::new(f);
        unsafe { mlirTypePrint(self.to_raw(), reader.callback(), reader.as_raw_mut()) }
        Ok(())
    }
}
