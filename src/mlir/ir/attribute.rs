mod bool;
mod dense_i32;
mod float;
mod integer;
mod string;

pub use self::{bool::*, dense_i32::*, float::*, integer::*, string::*};
use crate::{
    ir::{IdentifierRef, TypeRef},
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
    mlirAttributeEqual, mlirAttributeGetContext, mlirAttributeGetDialect, mlirAttributeGetType,
    mlirAttributeParseGet, mlirAttributePrint, MlirAttribute, MlirIdentifier, MlirNamedAttribute,
};

/// [AttributeRef] is a reference to an instance of the `mlir::Attribute` class, which represents a
/// constant value in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirAttributeEqual`
/// - `mlirAttributeGetContext`
/// - `mlirAttributeGetDialect`
/// - `mlirAttributeGetType`
/// - `mlirAttributeParseGet`
/// - `mlirAttributePrint`
///
/// The following bindings are not used/supported:
/// - `mlirAttributeDump`
/// - `mlirAttributeGetNull`
/// - `mlirAttributeGetTypeID`
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct AttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(no_refs, AttributeRef, MlirAttribute);

impl AttributeRef {
    /// Attempts to parse an attribute from the provided string.
    ///
    /// # Arguments
    /// * `context` - The context that should own the attribute.
    /// * `attribute` - The string to parse the attribute from.
    ///
    /// # Returns
    /// Returns a new [AttributeRef] reference if the attribute could be parsed, otherwise `None`.
    pub fn parse<'a>(context: &'a ContextRef, attribute: &str) -> Option<&'a Self> {
        unsafe {
            Self::try_from_raw(mlirAttributeParseGet(
                context.to_raw(),
                StringRef::from(&attribute).to_raw(),
            ))
        }
    }

    /// # Returns
    /// Returns a reference to the context that owns the attribute.
    pub fn context(&self) -> &ContextRef {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.to_raw())) }
    }

    /// # Returns
    /// Returns the type of the attribute.
    pub fn r#type(&self) -> &TypeRef {
        unsafe { TypeRef::from_raw(mlirAttributeGetType(self.to_raw())) }
    }

    /// # Returns
    /// Returns the dialect of the attribute.
    pub fn dialect(&self) -> &DialectRef {
        unsafe { DialectRef::from_raw(mlirAttributeGetDialect(self.to_raw())) }
    }

    /// # Returns
    /// Returns a named attribute with the provided name.
    pub fn with_name(&self, name: &str) -> NamedAttribute {
        let identifier = IdentifierRef::new(self.context(), name);
        unsafe { NamedAttribute::from_raw(identifier.to_raw(), self.to_raw()) }
    }
}

impl PartialEq for AttributeRef {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAttributeEqual(self.to_raw(), other.to_raw()) }
    }
}

impl Eq for AttributeRef {}

impl Display for AttributeRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut reader = StringReader::new(f);
        unsafe { mlirAttributePrint(self.to_raw(), reader.callback(), reader.as_raw_mut()) }
        Ok(())
    }
}

/// [NamedAttribute] holds a name-attribute pair.
#[repr(C)]
#[derive(Debug)]
pub struct NamedAttribute<'a> {
    name: &'a IdentifierRef,
    attribute: &'a AttributeRef,
}

impl<'a> NamedAttribute<'a> {
    /// Constructs a new [NamedAttribute] from the provided raw identifier and attribute.
    ///
    /// # Safety
    /// The caller of this function is responsible for associating the lifetime on the resulting
    /// value with the lifetime of the owner of the raw values, and ensuring that the provided raw
    /// values are valid.
    ///
    /// # Arguments
    /// * `raw_ident` - The raw identifier representing the name of the attribute.
    /// * `raw_attr` - The raw attribute.
    ///
    /// # Returns
    /// Returns a new [NamedAttribute] instance.
    pub unsafe fn from_raw(
        raw_ident: MlirIdentifier,
        raw_attr: MlirAttribute,
    ) -> NamedAttribute<'a> {
        NamedAttribute {
            name: IdentifierRef::from_raw(raw_ident),
            attribute: AttributeRef::from_raw(raw_attr),
        }
    }

    /// # Returns
    /// Returns the [NamedAttribute] as a raw [MlirNamedAttribute] value.
    pub fn to_raw(&self) -> MlirNamedAttribute {
        MlirNamedAttribute {
            name: self.name.to_raw(),
            attribute: self.attribute.to_raw(),
        }
    }

    /// # Returns
    /// Returns the name associated with the attribute.
    pub fn name(&self) -> &'a IdentifierRef {
        self.name
    }

    /// # Returns
    /// Returns the attribute value.
    pub fn attribute(&self) -> &'a AttributeRef {
        self.attribute
    }
}

macro_rules! impl_attribute_variant {
    ($variant_type:ident, $verify_fn:ident) => {
        impl $variant_type {
            pub fn try_from_attribute(attr: &$crate::ir::AttributeRef) -> Option<&Self> {
                let types_match = unsafe { $verify_fn(attr.to_raw()) };
                if types_match {
                    Some(unsafe { $variant_type::from_raw(attr.to_raw()) })
                } else {
                    None
                }
            }

            pub fn as_attribute(&self) -> &$crate::ir::AttributeRef {
                unsafe { $crate::ir::AttributeRef::from_raw(self.to_raw()) }
            }
        }

        impl std::ops::Deref for $variant_type {
            type Target = $crate::ir::AttributeRef;

            fn deref(&self) -> &Self::Target {
                self.as_attribute()
            }
        }
    };
}

use impl_attribute_variant;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::NoneTypeRef, Context};

    #[test]
    fn parse_attribute() {
        let context = Context::new(None, false);
        assert!(AttributeRef::parse(&context, "unit").is_some());
        assert!(AttributeRef::parse(&context, "i32").is_some());
        assert!(AttributeRef::parse(&context, r#""foo""#).is_some());
        assert!(AttributeRef::parse(&context, r#"z"#).is_none());
    }

    #[test]
    fn context() {
        let context = Context::new(None, false);
        let attribute = AttributeRef::parse(&context, "unit").unwrap();
        assert_eq!(attribute.context(), &context);
    }

    #[test]
    fn r#type() {
        let context = Context::new(None, false);
        let attribute = AttributeRef::parse(&context, "unit").unwrap();
        assert_eq!(attribute.r#type(), NoneTypeRef::new(&context).as_type());
    }

    #[test]
    fn dialect() {
        let context = Context::new(None, false);
        let attribute = AttributeRef::parse(&context, "unit").unwrap();
        assert_eq!(attribute.dialect().namespace(), "builtin");
    }

    #[test]
    fn with_name() {
        let context = Context::new(None, false);
        let attribute = AttributeRef::parse(&context, "unit").unwrap();
        let named_attribute = attribute.with_name("example");
        assert_eq!(named_attribute.name().value(), "example");
        assert_eq!(
            named_attribute.attribute(),
            AttributeRef::parse(&context, "unit").unwrap()
        );
    }

    #[test]
    fn compare_attributes() {
        let context = Context::new(None, false);
        let attribute1 = AttributeRef::parse(&context, "unit").unwrap();
        let attribute2 = AttributeRef::parse(&context, "unit").unwrap();
        let attribute3 = AttributeRef::parse(&context, "i32").unwrap();
        assert_eq!(attribute1, attribute1);
        assert_eq!(attribute1, attribute2);
        assert_eq!(attribute2, attribute2);
        assert_ne!(attribute1, attribute3);
        assert_ne!(attribute3, attribute1);
        assert_ne!(attribute2, attribute3);
        assert_ne!(attribute3, attribute2);
    }

    #[test]
    #[should_panic]
    fn no_owned_attribute_ref() {
        let _attribute_ref = AttributeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
