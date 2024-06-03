use crate::ir::{LocationRef, Operation};
use crate::{Context, OperationRef, OwnedMlirValue, StringRef, UnownedMlirValue};
use mlir_sys::{
    mlirModuleCreateEmpty, mlirModuleCreateParse, mlirModuleFromOperation, mlirModuleGetOperation,
    MlirModule,
};
use std::ffi::CString;
use std::marker::PhantomData;

pub struct ModuleRef {
    raw: MlirModule,
    _context: PhantomData<Context>,
}

impl ModuleRef {
    pub fn new(location: LocationRef) {
        unsafe {
            Self::from_raw(mlirModuleCreateEmpty(location.to_raw()));
        }
    }

    pub fn parse(context: &Context, source: &str) -> Option<Self> {
        let source = CString::new(source).unwrap();
        let source = StringRef::from_cstring(&source);

        unsafe { Self::from_option_raw(mlirModuleCreateParse(context.to_raw(), source.to_raw())) }
    }

    pub fn from_operation(operation: Operation) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirModuleFromOperation(operation.to_raw())) }
    }

    pub fn as_operation(&self) -> &OperationRef {
        unsafe { OperationRef::from_raw(mlirModuleGetOperation(self.raw)) }
    }

    pub unsafe fn from_raw(raw: MlirModule) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    pub unsafe fn from_option_raw(raw: MlirModule) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }

    pub const fn to_raw(&self) -> MlirModule {
        self.raw
    }
}
