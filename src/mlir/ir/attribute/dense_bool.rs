use super::impl_attribute_variant;
use crate::{support::binding::impl_unowned_mlir_value, ContextRef, UnownedMlirValue};

use std::{marker::PhantomData, os::raw::c_int};

use mlir_sys::{
    mlirAttributeIsADenseBoolArray, mlirDenseArrayGetNumElements, mlirDenseBoolArrayGet,
    mlirDenseBoolArrayGetElement, MlirAttribute,
};

/// [`DenseBoolAttributeRef`] is a reference to an instance of the `mlir::DenseBoolArrayAttr`, which
/// represents a constant array of booleans in the MLIR IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirAttributeIsADenseBoolArray`
/// - `mlirDenseArrayGetNumElements`
/// - `mlirDenseBoolArrayGetElement`
/// - `mlirDenseBoolArrayGet`
///
/// The following bindings are not used/supported:
/// - `mlirDenseArrayAttrGetTypeID`
#[repr(transparent)]
#[derive(Debug)]
pub struct DenseBoolAttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(no_refs, DenseBoolAttributeRef, MlirAttribute);
impl_attribute_variant!(DenseBoolAttributeRef, mlirAttributeIsADenseBoolArray);

impl DenseBoolAttributeRef {
    /// Constructs a new dense array of booleans attribute with the provided values.
    ///
    /// # Arguments
    /// * `context` - The context that should own the attribute.
    /// * `values` - The booleans to hold in the attribute.
    ///
    /// # Returns
    /// Returns a reference to a new [`DenseBoolAttributeRef`] instance.
    pub fn new<'a>(context: &'a ContextRef, values: &[bool]) -> &'a Self {
        let int_values: Vec<c_int> = values.iter().map(|&b| if b { 1 } else { 0 }).collect();
        unsafe {
            Self::from_raw(mlirDenseBoolArrayGet(
                context.to_raw(),
                values.len() as isize,
                int_values.as_ptr(),
            ))
        }
    }

    /// # Returns
    /// Returns the length of the array.
    pub fn len(&self) -> isize {
        unsafe { mlirDenseArrayGetNumElements(self.to_raw()) }
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
    /// Returns the boolean at the provided index.
    pub fn get(&self, index: isize) -> bool {
        assert!(index < self.len());
        unsafe { mlirDenseBoolArrayGetElement(self.to_raw(), index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn get_elements_len() {
        let context = Context::new(None, false);
        let values = [true, false, true];
        let attr = DenseBoolAttributeRef::new(&context, &values);
        assert_eq!(attr.len(), 3);
        assert!(!attr.is_empty());
    }

    #[test]
    fn get_value() {
        let context = Context::new(None, false);
        let values = [true, false, true];
        let attr = DenseBoolAttributeRef::new(&context, &values);
        assert!(attr.get(0));
        assert!(!attr.get(1));
        assert!(attr.get(2));
    }

    #[test]
    fn get_value_out_of_bounds() {
        let context = Context::new(None, false);
        let values = [true, false, true];
        let attr = DenseBoolAttributeRef::new(&context, &values);
        assert!(std::panic::catch_unwind(|| {
            attr.get(3);
        })
        .is_err());
    }

    #[test]
    #[should_panic]
    fn no_owned_dense_bool_attribute_ref() {
        let _dense_bool_attribute_ref = DenseBoolAttributeRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
