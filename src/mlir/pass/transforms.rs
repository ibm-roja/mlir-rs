use mlir_sys::{
    mlirCreateTransformsCSE, mlirCreateTransformsCanonicalizer, mlirCreateTransformsSymbolDCE,
    MlirPass,
};

pub fn create_canonicalization_pass() -> MlirPass {
    unsafe { mlirCreateTransformsCanonicalizer() }
}

pub fn create_cse_pass() -> MlirPass {
    unsafe { mlirCreateTransformsCSE() }
}

pub fn create_dce_pass() -> MlirPass {
    unsafe { mlirCreateTransformsSymbolDCE() }
}
