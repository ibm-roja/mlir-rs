use crate::ir::{OperationRef, ValueRef};
use crate::support::binding::impl_owned_mlir_value;
use crate::{OwnedMlirValue, UnownedMlirValue};
use mlir_sys::{
    mlirOpOperandGetOwner, mlirOpOperandGetValue, mlirOpOperandIsNull,
    MlirOpOperand,
};
use std::marker::PhantomData;

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct OpOperandRef<'c> {
    raw: MlirOpOperand,
    _context: PhantomData<&'c ()>,
}

impl_owned_mlir_value!(context_ref, OpOperandRef, MlirOpOperand);

/// ['OpOperandRef'] is a wrapper to an instance of the `mlir::OpOperand` class, which represents an
/// operand of an operation in the MLIR IR.
impl<'c> OpOperandRef<'c> {

    /// # Returns
    /// Returns true if the operand is null.
    pub fn is_null(&self) -> bool {
        unsafe { mlirOpOperandIsNull(self.raw) }
    }

    /// # Returns
    /// Returns the MlirValue of the operand.
    pub fn get_value(&self) -> &ValueRef<'c> {
        unsafe { ValueRef::from_raw(mlirOpOperandGetValue(self.to_raw())) }
    }

    /// # Returns
    /// Returns the operation that owns the operand.
    pub fn get_owner(&self) -> &OperationRef {
        unsafe { OperationRef::from_raw(mlirOpOperandGetOwner(self.to_raw())) }
    }
}
