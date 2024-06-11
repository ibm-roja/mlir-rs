use std::marker::PhantomData;

use mlir_sys::{
    MlirAttribute, mlirAttributeGetNull, mlirAttributeIsADenseElements,
    mlirDenseElementsAttrGetStringValue, mlirDenseElementsAttrStringGet,
    mlirElementsAttrGetNumElements, mlirRankedTensorTypeGet, MlirStringRef,
};

use crate::{
    ir::{attribute::impl_attribute_variant, TypeRef},
    support::binding::impl_unowned_mlir_value,
    StringRef, UnownedMlirValue,
};

/// [`DenseStringAttributeRef`] is a reference to an instance of the `mlir::DenseStringElementsAttr`, which
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
    /// * `string_type` - The type of each string element in the array.
    ///
    /// # Returns
    /// Returns a reference to a new [`DenseStringAttributeRef`] instance.
    pub fn new<'a>(
        values: &[StringRef],
        string_type: &'a TypeRef,
    ) -> &'a DenseStringAttributeRef {
        let shape: [i64; 1] = [values.len() as i64];

        let shaped_type = unsafe {
            mlirRankedTensorTypeGet(
                1,
                shape.as_ptr(),
                string_type.to_raw(),
                mlirAttributeGetNull(),
            )
        };

        unsafe {
            Self::from_raw(mlirDenseElementsAttrStringGet(
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
        assert!(index < self.len());
        unsafe {
            let string_ref = mlirDenseElementsAttrGetStringValue(self.to_raw(), index);
            StringRef::from_raw(string_ref)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Context,
        ir::{DenseStringAttributeRef, LocationRef, OperationBuilder, TypeRef},
        StringRef,
    };
    use std::ffi::CString;

    #[test]
    fn test_compile_return_dense_string_attribute() {
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);

        let strings = [
            CString::new("hello").unwrap(),
            CString::new("world").unwrap(),
            CString::new("foo").unwrap(),
        ];

        let string_refs = strings
            .iter()
            .map(|s| StringRef::from_cstring(s))
            .collect::<Vec<StringRef>>();

        let attr =
            DenseStringAttributeRef::new(string_refs.as_slice(), TypeRef::parse(&context, "!dialect.string").unwrap());

        let loc = LocationRef::new_unknown(&context);
        let op = OperationBuilder::new("dialect.op1", loc)
            .add_attributes(&[attr.with_name("dense string")])
            .build()
            .unwrap();

        println!("{}", *op);
    }

    #[test]
    fn test_get_dense_string_attributes() {
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);

        let strings = [
            CString::new("hello").unwrap(),
            CString::new("world").unwrap(),
            CString::new("foo").unwrap(),
        ];

        let string_refs = strings
            .iter()
            .map(|s| StringRef::from_cstring(s))
            .collect::<Vec<StringRef>>();

        let attr =
            DenseStringAttributeRef::new(string_refs.as_slice(), TypeRef::parse(&context, "!dialect.string").unwrap());

        let loc = LocationRef::new_unknown(&context);
        let op = OperationBuilder::new("dialect.op1", loc)
            .add_attributes(&[attr.with_name("dense string")])
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
