use mlir_sys::{mlirCreateTransformsSymbolDCE, MlirPass};

pub fn create_dce_pass() -> MlirPass {
    unsafe { mlirCreateTransformsSymbolDCE() }
}