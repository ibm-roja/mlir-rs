use mlir_sys::MlirLogicalResult;

/// [`LogicalResult`] represents the `llvm::LogicalResult` class.
#[derive(Debug, Clone, Copy)]
pub enum LogicalResult {
    Success,
    Failure,
}

impl LogicalResult {
    /// Constructs a [`LogicalResult`] from the provided raw [`MlirLogicalResult`] value.
    ///
    /// # Arguments
    /// * `raw` - The raw [`MlirLogicalResult`] value.
    ///
    /// # Returns
    /// Returns a new [`LogicalResult`] instance.
    pub const fn from_raw(raw: MlirLogicalResult) -> LogicalResult {
        if raw.value == 0 {
            Self::Failure
        } else {
            Self::Success
        }
    }

    /// # Returns
    /// Returns the [`MlirLogicalResult`] represented by the [`LogicalResult`].
    pub const fn to_raw(&self) -> MlirLogicalResult {
        match self {
            Self::Success => MlirLogicalResult { value: 1 },
            Self::Failure => MlirLogicalResult { value: 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::LogicalResult;

    #[test]
    fn test_to_raw() {
        let success = LogicalResult::Success;
        let failure = LogicalResult::Failure;
        assert_eq!(success.to_raw().value, 1);
        assert_eq!(failure.to_raw().value, 0);
    }
}
