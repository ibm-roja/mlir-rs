use std::{ffi::CString, marker::PhantomData};

use mlir_sys::{
    mlirAttributeGetNull, mlirAttributeIsADenseElements, mlirDenseElementsAttrGetStringValue,
    mlirDenseElementsAttrStringGet, mlirElementsAttrGetNumElements, mlirRankedTensorTypeGet,
    MlirAttribute, MlirStringRef,
};

use crate::{
    ir::{attribute::impl_attribute_variant, TypeRef},
    support::binding::impl_unowned_mlir_value,
    StringRef, UnownedMlirValue,
};

/// [`DenseStringAttributeRef`] is a reference to an instance of the `mlir::DenseStringElementsAttr`
/// class, which represents a constant array of strings in the MLIR IR.
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
    /// * `values` - The strings to hold in the attribute.
    /// * `string_type` - The type of each string element in the array.
    ///
    /// # Returns
    /// Returns a reference to a new [`DenseStringAttributeRef`] instance.
    pub fn new<'a>(
        values: &[impl AsRef<str>],
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

        let null_terminated_values: Vec<CString> = values
            .iter()
            .map(|v| CString::new(v.as_ref()).expect("Failed to convert value to CString"))
            .collect();
        let string_refs: Vec<StringRef> = null_terminated_values
            .iter()
            .map(StringRef::from_cstring)
            .collect();

        unsafe {
            Self::from_raw(mlirDenseElementsAttrStringGet(
                shaped_type,
                string_refs.len() as isize,
                string_refs.as_ptr() as *mut MlirStringRef,
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
        ir::{DenseStringAttributeRef, Operation, TypeRef},
        Context,
    };

    #[test]
    fn get_elements_len() {
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let values = ["hello", "world", "foo"];
        let attr = DenseStringAttributeRef::new(
            &values,
            TypeRef::parse(&context, "!dialect.string").unwrap(),
        );
        assert_eq!(attr.len(), 3);
        assert!(!attr.is_empty());
    }

    #[test]
    fn get_values() {
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let values = ["hello", "world", "foo"];
        let attr = DenseStringAttributeRef::new(
            &values,
            TypeRef::parse(&context, "!dialect.string").unwrap(),
        );
        assert_eq!(attr.get(0), "hello");
        assert_eq!(attr.get(1), "world");
        assert_eq!(attr.get(2), "foo");
    }

    #[test]
    fn empty_array() {
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let values: [&str; 0] = [];
        let attr = DenseStringAttributeRef::new(
            &values,
            TypeRef::parse(&context, "!dialect.string").unwrap(),
        );
        assert_eq!(attr.len(), 0);
        assert!(attr.is_empty());
    }

    #[test]
    fn parse_from_operation() {
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);

        let op = Operation::parse(
            &context,
            r#""dialect.op1"() {dense_string = dense<["hello", "world", "foo"]> : tensor<3x!dialect.string>} : () -> ()"#,
            "dense_string_test",
        ).unwrap();
        let attr_ref =
            DenseStringAttributeRef::try_from_attribute(op.attribute("dense_string").unwrap())
                .unwrap();

        assert_eq!(attr_ref.len(), 3);
        assert_eq!(attr_ref.get(0), "hello");
        assert_eq!(attr_ref.get(1), "world");
        assert_eq!(attr_ref.get(2), "foo");
    }
}
