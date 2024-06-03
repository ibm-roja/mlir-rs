use mlir_sys::{mlirCreateTransformsCanonicalizer, mlirCreateTransformsSymbolDCE, MlirPass};

pub fn create_dce_pass() -> MlirPass {
    unsafe { mlirCreateTransformsSymbolDCE() }
}

pub fn create_canonicalization_pass() -> MlirPass {
    unsafe { mlirCreateTransformsCanonicalizer() }
}
