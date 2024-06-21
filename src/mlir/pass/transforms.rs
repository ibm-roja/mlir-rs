use mlir_sys::{mlirCreateTransformsCSE, mlirCreateTransformsCanonicalizer, mlirCreateTransformsSymbolDCE, MlirPass, MlirOperationWalkCallback, mlirOperationWalk};
use crate::ir::{Operation, OperationRef};
use crate::{OwnedMlirValue, UnownedMlirValue};

pub fn create_canonicalization_pass() -> MlirPass {
    unsafe { mlirCreateTransformsCanonicalizer() }
}

pub fn create_cse_pass() -> MlirPass {
    unsafe { mlirCreateTransformsCSE() }
}

pub fn create_dce_pass() -> MlirPass {
    unsafe { mlirCreateTransformsSymbolDCE() }
}

pub fn walk(op: &OperationRef, callback: MlirOperationWalkCallback, user_data: *mut std::ffi::c_void) {
    // 1 == postorder
    unsafe { mlirOperationWalk(op.to_raw(), callback, user_data, 1) }
}
