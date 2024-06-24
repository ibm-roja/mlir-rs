use crate::ir::{OperationRef, ValueRef};
use crate::support::binding::impl_owned_mlir_value;
use crate::{OwnedMlirValue, UnownedMlirValue};
use mlir_sys::{
    mlirOpOperandGetNextUse, mlirOpOperandGetOwner, mlirOpOperandGetValue, mlirOpOperandIsNull,
    MlirOpOperand
};
use std::marker::PhantomData;

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct OpOperandRef<'c> {
    raw: MlirOpOperand,
    _context: PhantomData<&'c ()>,
}

impl_owned_mlir_value!(context_ref, OpOperandRef, MlirOpOperand);

impl<'c> OpOperandRef<'c> {
    pub fn is_null(&self) -> bool {
        unsafe { mlirOpOperandIsNull(self.raw) }
    }

    pub fn get_value(&self) -> &ValueRef<'c> {
        unsafe { ValueRef::from_raw(mlirOpOperandGetValue(self.to_raw())) }
    }

    pub fn get_owner(&self) -> &OperationRef {
        unsafe { OperationRef::from_raw(mlirOpOperandGetOwner(self.to_raw())) }
    }

    pub fn next_use(&self) -> Option<OpOperandRef<'c>> {
        let next_raw = unsafe { mlirOpOperandGetNextUse(self.to_raw()) };
        if unsafe { OpOperandRef::from_raw(next_raw).is_null() } {
            None
        } else {
            Some(unsafe { OpOperandRef::from_raw(next_raw) })
        }
    }
}
