use mlir_sys::MlirLogicalResult;

/// [`LogicalResult`] wraps the `llvm::LogicalResult` class, an unowned fragment of a string.
pub struct LogicalResult {
    raw: MlirLogicalResult,
}

impl LogicalResult {
    /// Constructs a [`LogicalResult`] from a provided bool value representing whether the result is a success.
    ///
    /// # Arguments
    /// * `is_success` - The bool value.
    ///
    /// # Returns
    /// Returns a new [`LogicalResult`] instance.
    pub const fn success() -> Self {
        Self {
            raw: MlirLogicalResult { value: 1 },
        }
    }

    /// Constructs a [`LogicalResult`] from a provided bool value representing whether the result is a failure.
    ///
    /// # Arguments
    /// * `is_failure` - The bool value.
    ///
    /// # Returns
    /// Returns a new [`LogicalResult`] instance.
    pub const fn failure() -> Self {
        Self {
            raw: MlirLogicalResult { value: 0 },
        }
    }

    /// # Returns
    /// Returns whether the [`LogicalResult`] value represents a success.
    pub fn succeeded(&self) -> bool {
        self.raw.value != 0
    }

    /// # Returns
    /// Returns whether the [`LogicalResult`] value represents a failure.
    pub fn failed(&self) -> bool {
        self.raw.value == 0
    }

    /// Constructs a [`LogicalResult`] from the provided raw [`MlirLogicalResult`] value.
    ///
    /// # Arguments
    /// * `raw` - The raw [`MlirLogicalResult`] value.
    ///
    /// # Returns
    /// Returns a new [`LogicalResult`] instance.
    pub fn from_raw(raw: MlirLogicalResult) -> Self {
        Self { raw }
    }

    /// # Returns
    /// Returns the [`MlirLogicalResult`] contained within the [`LogicalResult`].
    pub fn to_raw(&self) -> MlirLogicalResult {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use crate::LogicalResult;

    #[test]
    fn success() {
        let result = LogicalResult::success();
        assert!(result.succeeded());
        assert!(!result.failed());
    }

    #[test]
    fn failure() {
        let result = LogicalResult::failure();
        assert!(!result.succeeded());
        assert!(result.failed());
    }
}
