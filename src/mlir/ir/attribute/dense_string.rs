use crate::{ContextRef, StringRef, UnownedMlirValue};

use std::{marker::PhantomData};

use crate::ir::TypeRef;
use mlir_sys::{
    mlirAttributeGetNull,
    mlirDenseElementsAttrGetStringValue, mlirDenseElementsAttrStringGet,
    mlirElementsAttrGetNumElements, mlirRankedTensorTypeGet, MlirAttribute,
    MlirStringRef,
};

/// [`StringBoolAttributeRef`] is a reference to an instance of the `mlir::DenseStringElementsAttr`, which
/// represents a constant array of strings in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
///
/// The following bindings are not used/supported:
#[repr(transparent)]
#[derive(Debug)]
pub struct DenseStringAttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
    raw: MlirAttribute,
}

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
        context: &'a ContextRef,
        values: &[StringRef],
        string_type: &str,
    ) -> DenseStringAttributeRef {
        let shape: [i64; 1] = [values.len() as i64];

        let shaped_type = unsafe {
            mlirRankedTensorTypeGet(1, shape.as_ptr(), {
                TypeRef::parse(context, string_type).unwrap().to_raw()
            }, mlirAttributeGetNull())
        };

        let dense_elements_attr = unsafe {
            mlirDenseElementsAttrStringGet(
                shaped_type,
                values.len() as isize,
                values.as_ptr() as *mut MlirStringRef,
            )
        };

        unsafe{Self::from_raw(dense_elements_attr) }
    }

    /// # Returns
    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        (unsafe { mlirElementsAttrGetNumElements(self.raw) }) as usize
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
    /// Returns the string at the provided index.
    pub fn get(&self, index: isize) -> StringRef {
        assert!(index < self.len().try_into().unwrap());
        unsafe {
            let string_ref = mlirDenseElementsAttrGetStringValue(self.to_raw(), index);
            StringRef::from_raw(string_ref)
        }
    }

    pub unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            _prevent_external_instantiation: Default::default(),
            raw,
        }
    }

    pub unsafe fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}
