use std::{marker::PhantomData, mem::transmute};

use mlir_sys::MlirDialect;

/// [DialectRef] is a reference to an instance of the `mlir::Dialect` class, which holds MLIR
/// operations, types, and attributes, as well as the behaviour associated with the whole group.
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct DialectRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl DialectRef {
    /// Constructs a reference to a [DialectRef] from the provided raw [MlirDialect] value.
    ///
    /// # Safety
    /// The caller of this function is responsible for associating the reference to the [DialectRef]
    /// with a lifetime that is bound to its owner.
    ///
    /// # Arguments
    /// * `context` - The raw [MlirDialect] value.
    ///
    /// # Returns
    /// Returns a new [DialectRef] instance.
    pub unsafe fn from_raw<'a>(dialect: MlirDialect) -> &'a Self {
        transmute(dialect)
    }

    /// # Returns
    /// Returns the reference to the [DialectRef] as an [MlirDialect].
    pub fn to_raw(&self) -> MlirDialect {
        unsafe { transmute(self) }
    }
}
