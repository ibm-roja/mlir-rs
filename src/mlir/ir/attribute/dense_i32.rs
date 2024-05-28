use super::impl_attribute_variant;
use crate::{
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef,
};

use std::marker::PhantomData;

use mlir_sys::{
    mlirAttributeIsADenseI32Array, mlirDenseArrayGetNumElements, mlirDenseI32ArrayGet,
    mlirDenseI32ArrayGetElement, MlirAttribute,
};

/// [DenseI32AttributeRef] is a reference to an instance of the `mlir::DenseI32ArrayAttr`, which
/// represents a constant array of 32-bit integers in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirAttributeIsADenseI32Array`
/// - `mlirDenseArrayGetNumElements`
/// - `mlirDenseI32ArrayGetElement`
/// - `mlirDenseI32ArrayGet`
///
/// The following bindings are not used/supported:
/// - `mlirDenseArrayAttrGetTypeID`
#[repr(transparent)]
#[derive(Debug)]
pub struct DenseI32AttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(no_refs, DenseI32AttributeRef, MlirAttribute);
impl_attribute_variant!(DenseI32AttributeRef, mlirAttributeIsADenseI32Array);

impl DenseI32AttributeRef {
    /// Constructs a new dense array of 32-bit integers attribute with the provided values.
    ///
    /// # Arguments
    /// * `context` - The context that should own the attribute.
    /// * `values` - The integers to hold in the attribute.
    ///
    /// # Returns
    /// Returns a reference to a new [DenseI32AttributeRef] instance.
    pub fn new<'a>(context: &'a ContextRef, values: &[i32]) -> &'a Self {
        unsafe {
            Self::from_raw(mlirDenseI32ArrayGet(
                context.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// # Returns
    /// Returns the length of the array.
    pub fn len(&self) -> isize {
        unsafe { mlirDenseArrayGetNumElements(self.to_raw()) }
    }

    /// # Returns
    /// Returns whether the array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the element at the provided index, verifying that the index is within bounds.
    ///
    /// # Arguments
    /// * `index` - The index of the element to get.
    ///
    /// # Returns
    /// Returns the index at the provided index.
    pub fn get(&self, index: isize) -> i32 {
        assert!(index < self.len());
        unsafe { mlirDenseI32ArrayGetElement(self.to_raw(), index) }
    }
}
