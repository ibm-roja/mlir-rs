use crate::{
    support::{
        binding::{impl_unowned_mlir_value, UnownedMlirValue},
        string_reader::StringReader,
    },
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
/// - `mlirNoneTypeGet`
/// - `mlirTypeEqual`
/// - `mlirTypeGetContext`
/// - `mlirTypeGetDialect`
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn parse() {
        let context = Context::new(None, false);
        assert!(TypeRef::parse(&context, "i32").is_some());
        assert!(TypeRef::parse(&context, "f32").is_some());
        assert!(TypeRef::parse(&context, "index").is_some());
        assert!(TypeRef::parse(&context, "z").is_none());
    }

    #[test]
    fn context() {
        let context = Context::new(None, false);
        let ty = TypeRef::parse(&context, "i32").unwrap();
        assert_eq!(ty.context(), &context);
    }

    #[test]
    fn dialect() {
        let context = Context::new(None, false);
        let ty = TypeRef::parse(&context, "i32").unwrap();
        assert_eq!(ty.dialect().namespace(), "builtin");
    }

    #[test]
    fn compare_types() {
        let context = Context::new(None, false);
        let type1 = TypeRef::parse(&context, "i32").unwrap();
        let type2 = TypeRef::parse(&context, "i32").unwrap();
        let type3 = TypeRef::parse(&context, "f32").unwrap();
        assert_eq!(type1, type1);
        assert_eq!(type1, type2);
        assert_eq!(type2, type2);
        assert_ne!(type1, type3);
        assert_ne!(type3, type1);
        assert_ne!(type2, type3);
        assert_ne!(type3, type2);
    }

    #[test]
    #[should_panic]
    fn no_owned_type_ref() {
        let _type_ref = TypeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
