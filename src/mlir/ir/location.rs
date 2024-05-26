use crate::{
    ir::AttributeRef,
    support::{
        binding::{impl_unowned_mlir_value, UnownedMlirValue},
        string_reader::StringReader,
    },
    ContextRef, StringRef,
};

use std::{
    fmt::{Display, Formatter},
    marker::PhantomData,
    ptr::null,
};

use mlir_sys::{
    mlirLocationCallSiteGet, mlirLocationEqual, mlirLocationFileLineColGet, mlirLocationFusedGet,
    mlirLocationGetContext, mlirLocationNameGet, mlirLocationPrint, mlirLocationUnknownGet,
    MlirLocation,
};

/// [LocationRef] is a reference to an instance of the `mlir::Location` class, which represents a
/// source location in the MLIR IR.
///
/// All relevant bindings into the MLIR C API are used/supported:
/// - `mlirLocationFileLineColGet`
/// - `mlirLocationCallSiteGet`
/// - `mlirLocationFusedGet`
/// - `mlirLocationNameGet`
/// - `mlirLocationUnknownGet`
/// - `mlirLocationGetContext`
/// - `mlirLocationEqual`
/// - `mlirLocationPrint`
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct LocationRef {
    _prevent_external_instantiation: PhantomData<()>,
}

impl_unowned_mlir_value!(LocationRef, MlirLocation);

impl LocationRef {
    /// Constructs a new [LocationRef] representing a file, line number, and column number.
    ///
    /// # Arguments
    /// * `context` - The context that should own the location.
    /// * `filename` - The name of the file the location refers to.
    /// * `line` - The line number in the file.
    /// * `col` - The column number in the file.
    ///
    /// # Returns
    /// Returns a new [LocationRef] reference.
    pub fn new_file_line_col<'a>(
        context: &'a ContextRef,
        filename: &str,
        line: u32,
        col: u32,
    ) -> &'a Self {
        unsafe {
            Self::from_raw(mlirLocationFileLineColGet(
                context.to_raw(),
                StringRef::from(&filename).to_raw(),
                line,
                col,
            ))
        }
    }

    /// Constructs a new [LocationRef] representing a function call site.
    ///
    /// # Arguments
    /// * `callee` - The location of the callee at the call site.
    /// * `caller` - The location of the caller at the call site.
    ///
    /// # Returns
    /// Returns a new [LocationRef] reference.
    pub fn new_call_site<'a>(callee: &'a LocationRef, caller: &'a LocationRef) -> &'a Self {
        unsafe { Self::from_raw(mlirLocationCallSiteGet(callee.to_raw(), caller.to_raw())) }
    }

    /// Constructs a new [LocationRef] representing a fused location with associated metadata.
    ///
    /// # Arguments
    /// * `context` - The context that should own the location.
    /// * `locations` - The locations to fuse.
    /// * `metadata` - The metadata to associate with the fused location.
    ///
    /// # Returns
    /// Returns a new [LocationRef] reference.
    pub fn new_fused<'a>(
        context: &'a ContextRef,
        locations: &[&'a LocationRef],
        metadata: &'a AttributeRef,
    ) -> &'a Self {
        let locations = locations.iter().map(|l| l.to_raw()).collect::<Vec<_>>();
        unsafe {
            Self::from_raw(mlirLocationFusedGet(
                context.to_raw(),
                locations.len() as isize,
                locations.as_ptr(),
                metadata.to_raw(),
            ))
        }
    }

    /// Constructs a new [LocationRef] representing a name with an optional child location.
    ///
    /// # Arguments
    /// * `context` - The context that should own the location.
    /// * `name` - The name of the location.
    /// * `child_location` - An optional child location.
    ///
    /// # Returns
    /// Returns a new [LocationRef] reference.
    pub fn new_name<'a>(
        context: &'a ContextRef,
        name: &str,
        child_location: Option<&'a LocationRef>,
    ) -> &'a Self {
        unsafe {
            Self::from_raw(mlirLocationNameGet(
                context.to_raw(),
                StringRef::from(&name).to_raw(),
                child_location.map_or(MlirLocation { ptr: null() }, |l| l.to_raw()),
            ))
        }
    }

    /// Constructs a new [LocationRef] representing an unknown location.
    ///
    /// # Arguments
    /// * `context` - The context that should own the location.
    ///
    /// # Returns
    /// Returns a new [LocationRef] reference.
    pub fn new_unknown(context: &ContextRef) -> &Self {
        unsafe { Self::from_raw(mlirLocationUnknownGet(context.to_raw())) }
    }

    /// # Returns
    /// Returns a reference to the context that owns the location.
    pub fn context(&self) -> &ContextRef {
        unsafe { ContextRef::from_raw(mlirLocationGetContext(self.to_raw())) }
    }
}

impl PartialEq for LocationRef {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirLocationEqual(self.to_raw(), other.to_raw()) }
    }
}

impl Eq for LocationRef {}

impl Display for LocationRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut reader = StringReader::new(f);
        unsafe { mlirLocationPrint(self.to_raw(), reader.callback(), reader.as_raw_mut()) }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn new_file_line_col() {
        let context = Context::new(None, false);
        let location = LocationRef::new_file_line_col(&context, "file", 1, 2);
        assert_eq!(location.to_string(), r#"loc("file":1:2)"#);
    }

    #[test]
    fn new_call_site() {
        let context = Context::new(None, false);
        let callee = LocationRef::new_file_line_col(&context, "callee", 1, 2);
        let caller = LocationRef::new_file_line_col(&context, "caller", 3, 4);
        let location = LocationRef::new_call_site(callee, caller);
        assert_eq!(
            location.to_string(),
            r#"loc(callsite("callee":1:2 at "caller":3:4))"#
        );
    }

    // TODO: new_fused test

    #[test]
    fn new_name() {
        let context = Context::new(None, false);
        let child = LocationRef::new_file_line_col(&context, "child", 1, 2);
        let location = LocationRef::new_name(&context, "name", Some(child));
        assert_eq!(location.to_string(), r#"loc("name"("child":1:2))"#);
    }

    #[test]
    fn new_unknown() {
        let context = Context::new(None, false);
        let location = LocationRef::new_unknown(&context);
        assert_eq!(location.to_string(), r#"loc(unknown)"#);
    }

    #[test]
    fn context() {
        let context = Context::new(None, false);
        let location = LocationRef::new_unknown(&context);
        assert_eq!(location.context(), &context);
    }

    #[test]
    fn compare_locations() {
        let context = Context::new(None, false);
        let location1a = LocationRef::new_file_line_col(&context, "file", 1, 2);
        let location1b = LocationRef::new_file_line_col(&context, "file", 1, 2);
        let location2 = LocationRef::new_unknown(&context);
        assert_eq!(location1a, location1a);
        assert_eq!(location1b, location1b);
        assert_eq!(location1a, location1b);
        assert_ne!(location1a, location2);
        assert_ne!(location1b, location2);
        assert_eq!(location2, location2);
    }

    #[test]
    #[should_panic]
    fn no_owned_location_ref() {
        let _location_ref = LocationRef {
            _prevent_external_instantiation: PhantomData,
        };
    }
}
