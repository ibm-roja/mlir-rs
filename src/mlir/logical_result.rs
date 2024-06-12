use mlir_sys::MlirLogicalResult;

/// [`LogicalResult`] wraps the `llvm::LogicalResult` class, an unowned fragment of a string.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct LogicalResult {
    raw: MlirLogicalResult,
}

impl LogicalResult {
    /// Constructs a [`LogicalResult`] from the provided raw [`MlirLogicalResult`] value.
    ///
    /// # Arguments
    /// * `raw` - The raw [`MlirLogicalResult`] value.
    ///
    /// # Returns
    /// Returns a new [`LogicalResult`] instance.
    pub fn from_raw(raw: MlirLogicalResult) -> LogicalResult {
        Self { raw }
    }

    /// Constructs a [`LogicalResult`] from a provided bool value representing whether the result is a success.
    ///
    /// # Arguments
    /// * `is_success` - The bool value.
    ///
    /// # Returns
    /// Returns a new [`LogicalResult`] instance.
    pub fn success(is_success: bool) -> LogicalResult {
        return LogicalResult::from_raw(MlirLogicalResult {
            value: i8::from(is_success),
        });
    }

    /// Constructs a [`LogicalResult`] from a provided bool value representing whether the result is a failure.
    ///
    /// # Arguments
    /// * `is_failure` - The bool value.
    ///
    /// # Returns
    /// Returns a new [`LogicalResult`] instance.
    pub fn failure(is_failure: bool) -> LogicalResult {
        return LogicalResult::success(!is_failure);
    }

    /// # Returns
    /// Returns the [`MlirLogicalResult`] contained within the [`LogicalResult`].
    pub fn to_raw(&self) -> MlirLogicalResult {
        self.raw
    }

    /// # Returns
    /// Returns whether the [`LogicalResult`] value represents a success.
    pub fn succeeded(&self) -> bool {
        return self.to_raw().value != 0;
    }

    /// # Returns
    /// Returns whether the [`LogicalResult`] value represents a failure.
    pub fn failed(&self) -> bool {
        return !self.succeeded();
    }
}

#[cfg(test)]
mod tests {
    use crate::LogicalResult;

    #[test]
    fn test_success_true() {
        let result = LogicalResult::success(true);
        assert!(result.succeeded());
        assert!(!result.failed());
    }

    #[test]
    fn test_success_false() {
        let result = LogicalResult::success(false);
        assert!(!result.succeeded());
        assert!(result.failed());
    }

    #[test]
    fn test_failure_true() {
        let result = LogicalResult::failure(true);
        assert!(!result.succeeded());
        assert!(result.failed());
    }

    #[test]
    fn test_failure_false() {
        let result = LogicalResult::failure(false);
        assert!(result.succeeded());
        assert!(!result.failed());
    }
}
