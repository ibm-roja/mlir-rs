use std::{marker::PhantomData, os::raw::c_char, slice, str};

use mlir_sys::MlirStringRef;

/// [StringRef] wraps the `llvm::StringRef` class, an unowned fragment of a string.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct StringRef<'a> {
    raw: MlirStringRef,
    _string_owner: PhantomData<&'a ()>,
}

impl<'a> StringRef<'a> {
    /// Constructs a [StringRef] from the provided raw [MlirStringRef] value.
    ///
    /// # Safety
    /// The caller of this function is responsible for providing the correct lifetime for the
    /// resulting [StringRef] instance (bound to the owner of the string), and ensuring that the
    /// provided raw [MlirStringRef] value is valid.
    ///
    /// # Arguments
    /// * `raw` - The raw [MlirStringRef] value.
    ///
    /// # Returns
    /// Returns a new [StringRef] instance.
    pub unsafe fn from_raw(raw: MlirStringRef) -> StringRef<'a> {
        Self {
            raw,
            _string_owner: PhantomData,
        }
    }

    /// # Returns
    /// Returns the raw [MlirStringRef] value.
    pub fn to_raw(&self) -> MlirStringRef {
        self.raw
    }

    /// # Returns
    /// Returns the [StringRef] as a `&str`.
    pub fn as_str(&self) -> &'a str {
        if self.raw.length == 0 {
            return "";
        }

        let bytes = unsafe { slice::from_raw_parts(self.raw.data as *const u8, self.raw.length) };
        let string_bytes = if bytes[bytes.len() - 1] == 0 {
            &bytes[..bytes.len() - 1]
        } else {
            bytes
        };

        str::from_utf8(string_bytes).expect("MLIR StringRef was not valid UTF-8")
    }
}

impl<'a, T> From<&'a T> for StringRef<'a>
where
    T: AsRef<str>,
{
    fn from(value: &'a T) -> StringRef<'a> {
        let value: &str = value.as_ref();
        Self {
            // An [MLIRStringRef] does not need to be null-terminated, so using the Rust string's
            // data directly is safe.
            raw: MlirStringRef {
                data: value.as_ptr() as *const c_char,
                length: value.len(),
            },
            _string_owner: PhantomData,
        }
    }
}

impl<'a> From<StringRef<'a>> for &'a str {
    fn from(value: StringRef<'a>) -> &'a str {
        value.as_str()
    }
}

impl<'a> PartialEq for StringRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl<'a, T> PartialEq<T> for StringRef<'a>
where
    T: AsRef<str>,
{
    fn eq(&self, other: &T) -> bool {
        other.as_ref() == self.as_str()
    }
}

impl<'a> PartialEq<StringRef<'a>> for str {
    fn eq(&self, other: &StringRef<'a>) -> bool {
        self == other.as_str()
    }
}

impl<'a> PartialEq<StringRef<'a>> for String {
    fn eq(&self, other: &StringRef<'a>) -> bool {
        self.as_str() == other.as_str()
    }
}

impl<'a> Eq for StringRef<'a> {}
