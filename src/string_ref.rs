use std::{marker::PhantomData, os::raw::c_char, slice, str};

use mlir_sys::MlirStringRef;

/// [StringRef] wraps the `llvm::StringRef` class, an unowned fragment of a string.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct StringRef<'a> {
    raw: MlirStringRef,
    _string_owner: PhantomData<&'a ()>,
}

impl StringRef<'_> {
    /// # Returns
    /// Returns the raw [MlirStringRef] value.
    pub fn to_raw(&self) -> MlirStringRef {
        self.raw
    }
}

impl<'a, T> From<&'a T> for StringRef<'a>
where
    T: AsRef<str>,
{
    fn from(value: &'a T) -> StringRef<'a> {
        let string = value.as_ref();
        Self {
            // An [MLIRStringRef] does not need to be null-terminated, so using the Rust string's
            // data directly is safe.
            raw: MlirStringRef {
                data: string.as_ptr() as *const c_char,
                length: string.len(),
            },
            _string_owner: PhantomData,
        }
    }
}

impl<'a> From<StringRef<'a>> for &'a str {
    fn from(value: StringRef<'a>) -> &'a str {
        let bytes = unsafe { slice::from_raw_parts(value.raw.data as *const u8, value.raw.length) };
        str::from_utf8(bytes).expect("MLIR StringRef was not valid UTF-8")
    }
}
