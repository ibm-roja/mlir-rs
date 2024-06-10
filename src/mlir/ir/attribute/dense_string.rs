use std::{marker::PhantomData};

use mlir_sys::{
    MlirAttribute, mlirAttributeGetContext, mlirAttributeGetNull, mlirAttributeIsADenseElements,
    mlirDenseElementsAttrGetStringValue, mlirDenseElementsAttrStringGet,
    mlirElementsAttrGetNumElements, MlirStringRef, mlirRankedTensorTypeGet
};

use crate::{
    ContextRef,
    ir::{
        attribute::impl_attribute_variant,
        IdentifierRef, NamedAttribute, TypeRef
    }, StringRef,
    support::binding::impl_unowned_mlir_value, UnownedMlirValue
};

/// [`StringBoolAttributeRef`] is a reference to an instance of the `mlir::DenseStringElementsAttr`, which
/// represents a constant array of strings in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirAttributeIsADenseElements`
/// - `mlirElementsAttrGetNumElements`
/// - `mlirDenseElementsAttrGetStringValue`
/// - `mlirDenseElementsAttrStringGet`
#[repr(transparent)]
#[derive(Debug)]
pub struct DenseStringAttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
    raw: MlirAttribute,
}

impl_unowned_mlir_value!(no_refs, DenseStringAttributeRef, MlirAttribute);
impl_attribute_variant!(DenseStringAttributeRef, mlirAttributeIsADenseElements);

impl DenseStringAttributeRef {
    /// Constructs a new dense array of strings attribute with the provided values.
    ///
    /// # Arguments
    /// * `context` - The context that should own the attribute.
    /// * `values` - The strings to hold in the attribute.
    /// * `string_type` - The string type to be parsed to an MlirType.
    ///
    /// # Returns
    /// Returns a reference to a new [`DenseStringAttributeRef`] instance.
    pub fn new<'a>(
        context: &ContextRef,
        values: &[StringRef],
        string_type: &str,
    ) -> &'a DenseStringAttributeRef {
        let shape: [i64; 1] = [values.len() as i64];

        let shaped_type = unsafe {
            mlirRankedTensorTypeGet(1, shape.as_ptr(), {
                TypeRef::parse(&context, string_type).unwrap().to_raw()
            }, mlirAttributeGetNull())
        };

        unsafe{
        Self::from_raw(
            mlirDenseElementsAttrStringGet(
                shaped_type,
                values.len() as isize,
                values.as_ptr() as *mut MlirStringRef,
            ))
        }

    }

    /// # Returns
    /// Returns the length of the array.
    pub fn len(&self) -> isize {
        unsafe { mlirElementsAttrGetNumElements(self.to_raw()) as isize }
    }

    /// # Returns
    /// Returns whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the element at the provided index, verifying that the index is within bounds.
    ///
    /// # Arguments
    /// * `index` - The index of the element to get.
    ///
    /// # Returns
    /// Returns the StringRef at the provided index.
    pub fn get(&self, index: isize) -> StringRef {
        assert!(index < self.len().try_into().unwrap());
        unsafe {
            let string_ref = mlirDenseElementsAttrGetStringValue(self.to_raw(), index);
            StringRef::from_raw(string_ref)
        }
    }

    /// # Returns
    /// Returns a named attribute with the provided name.
    pub fn with_name(&self, name: &str) -> NamedAttribute {
        let identifier = IdentifierRef::new(&self.context(), name);
        unsafe { NamedAttribute::from_raw(identifier.to_raw(), self.to_raw()) }
    }

    /// # Returns
    /// Returns a reference to the context that owns the attribute.
    pub fn context(&self) -> &ContextRef {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.to_raw())) }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use crate::{ir::{DenseStringAttributeRef, LocationRef, OperationBuilder}, StringRef};

    #[test]
    fn test_compile_return_dense_string_attribute() {
        let context = crate::Context::new(None, false);
        context.set_allow_unregistered_dialects(true);

        let strings = [
            CString::new("hello").unwrap(),
            CString::new("world").unwrap(),
            CString::new("foo").unwrap()
        ];

        let string_refs = strings
            .iter()
            .map(|s| StringRef::from_cstring(s))
            .collect::<Vec<StringRef>>();

        let attr = DenseStringAttributeRef::new(&context, string_refs.as_slice(), "!dialect.string");

        let loc = LocationRef::new_unknown(&context);
        let op = OperationBuilder::new("dialect.op1", loc)
            .add_attributes( &[attr.with_name("dense string")])
            .build()
            .unwrap();

        println!("{}", *op);
    }

    #[test]
    fn test_get_dense_string_attributes(){
        let context = crate::Context::new(None, false);
        context.set_allow_unregistered_dialects(true);

        let strings = [
            CString::new("hello").unwrap(),
            CString::new("world").unwrap(),
            CString::new("foo").unwrap()
        ];

        let string_refs = strings
            .iter()
            .map(|s| StringRef::from_cstring(s))
            .collect::<Vec<StringRef>>();

        let attr = DenseStringAttributeRef::new(&context, string_refs.as_slice(), "!dialect.string");

        let loc = LocationRef::new_unknown(&context);
        let op = OperationBuilder::new("dialect.op1", loc)
            .add_attributes( &[attr.with_name("dense string")])
            .build()
            .unwrap();

        let attribute_ref = op.attribute("dense string").unwrap();
        let dense_attribute = DenseStringAttributeRef::try_from_attribute(attribute_ref).unwrap();

        assert_eq!(dense_attribute.len(), 3);
        assert!(dense_attribute.get(0).eq(&string_refs[0]));
        assert!(dense_attribute.get(1).eq(&string_refs[1]));
        assert!(dense_attribute.get(2).eq(&string_refs[2]));
    }
}
