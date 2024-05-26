use crate::StringRef;

use std::{fmt::Write, marker::PhantomData, os::raw::c_void};

use mlir_sys::{MlirStringCallback, MlirStringRef};

pub(crate) struct StringReader<'a, T: Write> {
    destination: &'a mut T,
}

impl<'a, T> StringReader<'a, T>
where
    T: Write,
{
    pub(crate) fn new(destination: &'a mut T) -> Self {
        Self { destination }
    }

    pub(crate) fn as_raw_mut(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    pub(crate) fn callback(&self) -> MlirStringCallback {
        Some(Self::raw_callback)
    }

    fn push(&mut self, string: StringRef) {
        write!(self.destination, "{}", string.as_str())
            .expect("Could not write MLIR string to destination");
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn from_raw_mut<U: ?Sized>(raw: *mut c_void, _owner: &U) -> &mut Self {
        let reader = raw as *mut Self;
        &mut *reader
    }

    extern "C" fn raw_callback(raw_string: MlirStringRef, raw_reader: *mut c_void) {
        let reader_owner = PhantomData::<()>;
        let reader = unsafe { Self::from_raw_mut(raw_reader, &reader_owner) };
        let string = unsafe { StringRef::from_raw(raw_string) };
        reader.push(string);
    }
}
