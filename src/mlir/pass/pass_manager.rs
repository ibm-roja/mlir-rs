use super::Pass;

use crate::{ir::Operation, Context, ContextRef, LogicalResult, OwnedMlirValue, UnownedMlirValue};

use std::marker::PhantomData;

use mlir_sys::{mlirPassManagerAddOwnedPass, mlirPassManagerRunOnOp, MlirPassManager};

/// A PassManager is the top-level entry point for managing a set of optimization passes over a module.
/// It is responsible for scheduling and running the passes.
pub struct PassManager {
    raw: MlirPassManager,
    _context: PhantomData<Context>,
}

impl PassManager {
    /// Creates a new pass manager.
    pub fn new(context: &ContextRef) -> Self {
        let raw = unsafe { mlir_sys::mlirPassManagerCreate(context.to_raw()) };
        Self {
            raw,
            _context: PhantomData,
        }
    }

    /// Destroys the pass manager.
    pub fn destroy(&self) {
        unsafe { mlir_sys::mlirPassManagerDestroy(self.raw) }
    }

    /// Adds a pass to the pass manager.
    /// This pass will be applied to the module when the pass manager is run.
    pub fn add_pass(&self, pass: Pass) {
        unsafe { mlirPassManagerAddOwnedPass(self.raw, pass.to_raw()) }
    }

    /// Runs the passes added to the pass manager against any module
    pub fn run(&self, op: &Operation) {
        let result =
            unsafe { LogicalResult::from_raw(mlirPassManagerRunOnOp(self.raw, op.to_raw())) };

        assert!(match result {
            LogicalResult::Success => true,
            LogicalResult::Failure => {
                panic!("Failed to run pass manager on operation");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::pass::transforms;

    #[test]
    fn create_destroy_pass_manager() {
        let context = Context::new(None, false);
        let pass_manager = PassManager::new(&context);
        pass_manager.destroy();
    }

    #[test]
    pub fn add_pass() {
        let context = Context::new(None, false);
        let pass_manager = PassManager::new(&context);
        let pass = transforms::create_dce_pass();
        pass_manager.add_pass(Pass::from_raw(pass));
        pass_manager.destroy();
    }

    #[test]
    pub fn run_pass() {
        let context = Context::new(None, false);
        let pass_manager = PassManager::new(&context);
        let pass = transforms::create_dce_pass();
        pass_manager.add_pass(Pass::from_raw(pass));
        let operation_source = "module {}";
        let operation = Operation::parse(&context, operation_source, "").unwrap();
        pass_manager.run(&operation);
        pass_manager.destroy();
    }
}
