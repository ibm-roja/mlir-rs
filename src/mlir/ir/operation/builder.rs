use crate::{
    ir::{LocationRef, NamedAttribute, TypeRef, ValueRef},
    StringRef, UnownedMlirValue,
};

use std::marker::PhantomData;

use mlir_sys::{
    mlirOperationStateAddAttributes, mlirOperationStateAddOperands, mlirOperationStateAddResults,
    mlirOperationStateGet, MlirNamedAttribute, MlirOperationState, MlirType, MlirValue,
};

pub struct OperationBuilder<'a> {
    state: MlirOperationState,
    _context: PhantomData<&'a ()>,
}

impl<'a> OperationBuilder<'a> {
    pub fn new(name: &str, location: &'a LocationRef) -> OperationBuilder<'a> {
        Self {
            state: unsafe {
                mlirOperationStateGet(StringRef::from(&name).to_raw(), location.to_raw())
            },
            _context: PhantomData,
        }
    }

    pub fn add_results(mut self, types: &[&'a TypeRef]) -> Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.state as *mut MlirOperationState,
                types.len() as isize,
                types.as_ptr() as *const MlirType,
            );
        }
        self
    }

    pub fn add_operands(mut self, operands: &[&'a ValueRef]) -> Self {
        unsafe {
            mlirOperationStateAddOperands(
                &mut self.state as *mut MlirOperationState,
                operands.len() as isize,
                operands.as_ptr() as *const MlirValue,
            );
        }
        self
    }

    // TODO: Add support for regions & successors.

    pub fn add_attributes(mut self, attributes: &[NamedAttribute<'a>]) -> Self {
        unsafe {
            mlirOperationStateAddAttributes(
                &mut self.state as *mut MlirOperationState,
                attributes.len() as isize,
                attributes.as_ptr() as *const MlirNamedAttribute,
            );
        }
        self
    }

    // TODO: Add support for enabling type inference.
}
