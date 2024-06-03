mod pass_manager;
mod transforms;

pub use self::{pass_manager::*, transforms::*};

use mlir_sys::MlirPass;

/// A Pass in MLIR is a transformation or analysis that can be applied to a module.
/// They must be added to a PassManager to be executed.
pub struct Pass{
    raw: MlirPass
}

impl Pass{
    pub fn new(raw: MlirPass) -> Self {
        Self { raw }
    }

    pub fn to_raw(&self) -> MlirPass {
        self.raw
    }

    pub fn from_raw(raw: MlirPass) -> Self {
        Self { raw }
    }
}
