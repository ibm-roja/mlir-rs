use super::impl_attribute_variant;
use crate::{
    ir::TypeRef,
    support::binding::{impl_unowned_mlir_value, UnownedMlirValue},
    ContextRef, StringRef,
};

use std::marker::PhantomData;

use mlir_sys::{
    mlirAttributeIsADenseElements, mlirDenseElementsAttrStringGet, mlirShapedTypeGetDimSize,
    MlirAttribute, MlirStringRef,
};

#[repr(transparent)]
#[derive(Debug)]
pub struct DenseElementsAttributeRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(no_refs, DenseElementsAttributeRef, MlirAttribute);
impl_attribute_variant!(DenseElementsAttributeRef, mlirAttributeIsADenseElements);

impl DenseElementsAttributeRef {
    pub fn new_strings<'a>(ty: &'a TypeRef, values: &[StringRef]) -> &'a Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrStringGet(
                ty.to_raw(),
                values.len() as isize,
                values.as_ptr() as *mut MlirStringRef,
            ))
        }
    }
}
