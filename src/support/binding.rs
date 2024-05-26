/// [OwnedMlirValue] is a trait for types that represent MLIR values that are owned by their Rust
/// wrapper type.
pub trait OwnedMlirValue
where
    Self: Sized,
{
    /// The C API type that this Rust type owns.
    type Binding;

    /// Constructs a new instance of the Rust wrapper type from the provided raw value.
    ///
    /// # Safety
    /// The caller of this function is responsible for ensuring that the provided raw value is
    /// valid.
    ///
    /// # Arguments
    /// * `raw` - The raw MLIR value.
    ///
    /// # Returns
    /// Returns a new wrapper type instance.
    unsafe fn from_raw(raw: Self::Binding) -> Self;

    /// Optionally constructs a new instance of the Rust wrapper type from the provided raw value,
    /// if the raw value is not null.
    ///
    /// # Safety
    /// The caller of this function is responsible for ensuring that the provided raw value is
    /// valid if it is not null.
    ///
    /// # Arguments
    /// * `raw` - The raw MLIR value.
    ///
    /// # Returns
    /// Returns an optional wrapper type instance.
    unsafe fn try_from_raw(raw: Self::Binding) -> Option<Self>;

    /// # Returns
    /// Returns the raw MLIR value from the Rust wrapper type instance.
    fn to_raw(&self) -> Self::Binding;
}

/// [UnownedMlirValue] is a trait for types that represent MLIR values that are not owned by their
/// Rust wrapper type.
///
/// For example, this is useful for representing values that are be owned by an MLIR context. It is
/// also useful to represent the reference variant of types that also have an owned variant (e.g.
/// [Context] and [ContextRef], [DialectRegistry] and [DialectRegistryRef], etc.).
///
/// Any type implementing this trait should be assumed to ONLY be safe to use as a *reference*, not
/// as an owned value! This is because the underlying MLIR value is not owned by the Rust code,
/// the Rust reference is truly just pointing to an MLIR value, and Rust code should not be able to
/// drop it or otherwise invoke C++ code on memory managed by itself.
pub trait UnownedMlirValue
where
    Self: Sized,
{
    /// The C API type that this Rust type is a reference to.
    type Binding;

    /// Constructs a reference to the Rust wrapper type from the provided raw value.
    ///
    /// # Safety
    /// The caller of this function is responsible for associating the reference to the wrapper type
    /// with a lifetime that is bound to its owner, and ensuring that the provided raw value is
    /// valid.
    ///
    /// # Arguments
    /// * `raw` - The raw MLIR value.
    ///
    /// # Returns
    /// Returns a new wrapper type reference.
    unsafe fn from_raw<'a>(raw: Self::Binding) -> &'a Self;

    /// Optionally constructs a reference to the Rust wrapper type from the provided raw value, if
    /// the raw value is not null.
    ///
    /// # Safety
    /// The caller of this function is responsible for associating the reference to the wrapper type
    /// with a lifetime that is bound to its owner, and ensuring that the provided raw value is
    /// valid if it is not null.
    ///
    /// # Arguments
    /// * `raw` - The raw MLIR value.
    ///
    /// # Returns
    /// Returns an optional wrapper type reference.
    unsafe fn try_from_raw<'a>(raw: Self::Binding) -> Option<&'a Self>;

    /// # Returns
    /// Returns the raw MLIR value from the Rust wrapper type reference.
    fn to_raw(&self) -> Self::Binding;
}

macro_rules! impl_owned_mlir_value {
    ($owning_type:ty, $binding:ty) => {
        impl $crate::support::binding::OwnedMlirValue for $owning_type {
            type Binding = $binding;

            unsafe fn from_raw(raw: Self::Binding) -> Self {
                Self { raw }
            }

            unsafe fn try_from_raw(raw: Self::Binding) -> Option<Self> {
                if raw.ptr.is_null() {
                    None
                } else {
                    Some(Self::from_raw(raw))
                }
            }

            fn to_raw(&self) -> Self::Binding {
                self.raw
            }
        }
    };
}

macro_rules! impl_unowned_mlir_value {
    ($ref_type:ty, $binding:ty) => {
        impl $crate::support::binding::UnownedMlirValue for $ref_type {
            type Binding = $binding;

            unsafe fn from_raw<'a>(raw: Self::Binding) -> &'a Self {
                std::mem::transmute(raw)
            }

            unsafe fn try_from_raw<'a>(raw: Self::Binding) -> Option<&'a Self> {
                if raw.ptr.is_null() {
                    None
                } else {
                    Some(Self::from_raw(raw))
                }
            }

            fn to_raw(&self) -> Self::Binding {
                unsafe { std::mem::transmute(self) }
            }
        }

        impl Drop for $ref_type {
            fn drop(&mut self) {
                panic!(
                    "Owned instances of {} should never be created!",
                    stringify!($ref_type)
                )
            }
        }
    };
    ($owning_type:ty, $ref_type:ident, $binding:ty) => {
        impl Deref for $owning_type {
            type Target = $ref_type;

            fn deref(&self) -> &Self::Target {
                unsafe { $ref_type::from_raw(self.raw) }
            }
        }

        impl_unowned_mlir_value!($ref_type, $binding);
    };
}

pub(crate) use {impl_owned_mlir_value, impl_unowned_mlir_value};
