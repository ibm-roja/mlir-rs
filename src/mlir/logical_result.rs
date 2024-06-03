use mlir_sys::MlirLogicalResult;

/// A LogicalResult is a wrapper around MlirLogicalResult.
/// MlirLogicalResult is a type used to represent the result of a logical operation.
pub struct LogicalResult {
    raw: MlirLogicalResult,
}

impl LogicalResult {

    /// Creates a success result.
    pub const fn success() -> Self {
        Self {
            raw: MlirLogicalResult { value: 1 },
        }
    }

    /// Creates a failure result.
    pub const fn failure() -> Self {
        Self {
            raw: MlirLogicalResult { value: 0 },
        }
    }

    /// Checks to see if the logical result is a success.
    pub fn succeeded(&self) -> bool {
        self.raw.value != 0
    }

    /// Checks to see if the logical result is a failure.
    pub fn failed(&self) -> bool {
        self.raw.value == 0
    }

    pub fn from_raw(raw: MlirLogicalResult) -> Self {
        Self { raw }
    }
}

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