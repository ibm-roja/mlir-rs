use super::TypeRef;
use crate::support::{
    binding::{impl_unowned_mlir_value, UnownedMlirValue},
    string_reader::StringReader,
};

use std::{
    fmt::{Display, Formatter},
    marker::PhantomData,
};

use crate::ir::opoperand::OpOperandRef;
use crate::OwnedMlirValue;
use mlir_sys::{
    mlirValueEqual, mlirValueGetFirstUse, mlirValueGetType, mlirValueIsABlockArgument,
    mlirValueIsAOpResult, mlirValuePrint, mlirValueSetType, MlirValue,
};

/// [ValueRef] is a reference to an instance of the `mlir::Value` class, which represents a value in
/// the MLIR IR (such as the result of an operation or an argument to a block).
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirValueEqual`
/// - `mlirValueGetType`
/// - `mlirValueIsABlockArgument`
/// - `mlirValueIsAOpResult`
/// - `mlirValuePrint`
/// - `mlirValueSetType`
///
/// The following bindings are not used/supported:
/// - `mlirValueDump`
/// - `mlirValueGetFirstUse`
/// - `mlirValuePrintAsOperand`
/// - `mlirValueReplaceAllUsesOfWith`
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct ValueRef<'c> {
    _context: PhantomData<&'c ()>,
}

impl_unowned_mlir_value!(context_ref, ValueRef, MlirValue);

impl<'c> ValueRef<'c> {
    /// # Returns
    /// Returns the type of the value.
    pub fn r#type(&self) -> &'c TypeRef {
        unsafe { TypeRef::from_raw(mlirValueGetType(self.to_raw())) }
    }

    /// Sets the type of the value.
    ///
    /// # Arguments
    /// * `ty` - The type to set the value to.
    pub fn set_type(&self, ty: &'c TypeRef) {
        unsafe {
            mlirValueSetType(self.to_raw(), ty.to_raw());
        }
    }

    /// # Returns
    /// Returns whether the value is a block argument.
    pub fn is_block_argument(&self) -> bool {
        unsafe { mlirValueIsABlockArgument(self.to_raw()) }
    }

    /// # Returns
    /// Returns whether the value is an operation result.
    pub fn is_op_result(&self) -> bool {
        unsafe { mlirValueIsAOpResult(self.to_raw()) }
    }


    /// Finds the last use of a ValueRef as an OpOperandRef. This finds the last use because the uses
    /// are stored in a stack, so the last use is the first one to be popped off the stack.
    ///
    /// If the value is not used, this will return None.
    ///
    /// # Returns
    /// Returns the last use of the value as an Option<OpOperandRef>.
    pub fn get_first_use(&self) -> Option<OpOperandRef<'c>> {
        let raw = unsafe { OpOperandRef::from_raw(mlirValueGetFirstUse(self.to_raw())) };
        if raw.is_null() {
            None
        } else {
            Some(raw)
        }
    }
}

impl<'c> PartialEq for ValueRef<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirValueEqual(self.to_raw(), other.to_raw()) }
    }
}

impl<'c> Eq for ValueRef<'c> {}

impl<'c> Display for ValueRef<'c> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut reader = StringReader::new(f);
        unsafe { mlirValuePrint(self.to_raw(), reader.callback(), reader.as_raw_mut()) }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn no_owned_value_ref() {
        let _value_ref = ValueRef {
            _context: PhantomData,
        };
    }
}
