mod builder;

pub use self::builder::OperationBuilder;
use crate::{
    ir::{AttributeRef, BlockRef, IdentifierRef, LocationRef, NamedAttribute, RegionRef, ValueRef},
    support::{
        binding::{
            impl_owned_mlir_value, impl_unowned_mlir_value, OwnedMlirValue, UnownedMlirValue,
        },
        string_reader::StringReader,
    },
    ContextRef, StringRef,
};

use std::ffi::c_void;
use std::{
    ffi::CString,
    fmt::{Display, Formatter},
    marker::PhantomData,
};

use mlir_sys::{
    mlirOperationClone, mlirOperationCreateParse, mlirOperationDestroy, mlirOperationEqual,
    mlirOperationGetAttribute, mlirOperationGetAttributeByName, mlirOperationGetBlock,
    mlirOperationGetContext, mlirOperationGetDiscardableAttribute,
    mlirOperationGetDiscardableAttributeByName, mlirOperationGetFirstRegion,
    mlirOperationGetInherentAttributeByName, mlirOperationGetLocation, mlirOperationGetName,
    mlirOperationGetNextInBlock, mlirOperationGetNumAttributes,
    mlirOperationGetNumDiscardableAttributes, mlirOperationGetNumOperands,
    mlirOperationGetNumRegions, mlirOperationGetNumResults, mlirOperationGetOperand,
    mlirOperationGetParentOperation, mlirOperationGetRegion, mlirOperationGetResult,
    mlirOperationHasInherentAttributeByName, mlirOperationMoveAfter, mlirOperationMoveBefore,
    mlirOperationPrint, mlirOperationRemoveAttributeByName,
    mlirOperationRemoveDiscardableAttributeByName, mlirOperationRemoveFromParent,
    mlirOperationSetAttributeByName, mlirOperationSetDiscardableAttributeByName,
    mlirOperationSetInherentAttributeByName, mlirOperationSetOperand, mlirOperationVerify,
    mlirOperationWalk, MlirOperation, MlirOperationWalkCallback,
};

/// [Operation] wraps the `mlir::Operation` class, which represents a single operation in the MLIR
/// IR.
///
/// The following bindings into the MLIR C API are used/supported:
/// - `mlirOperationClone`
/// - `mlirOperationCreateParse`
/// - `mlirOperationDestroy`
/// - `mlirOperationEqual`
/// - `mlirOperationGetAttribute`
/// - `mlirOperationGetAttributeByName`
/// - `mlirOperationGetBlock`
/// - `mlirOperationGetContext`
/// - `mlirOperationGetDiscardableAttribute`
/// - `mlirOperationGetDiscardableAttributeByName`
/// - `mlirOperationGetFirstRegion`
/// - `mlirOperationGetInherentAttributeByName`
/// - `mlirOperationGetLocation`
/// - `mlirOperationGetName`
/// - `mlirOperationGetNextInBlock`
/// - `mlirOperationGetNumAttributes`
/// - `mlirOperationGetNumDiscardableAttributes`
/// - `mlirOperationGetNumOperands`
/// - `mlirOperationGetNumRegions`
/// - `mlirOperationGetNumResults`
/// - `mlirOperationGetOperand`
/// - `mlirOperationGetParentOperation`
/// - `mlirOperationGetRegion`
/// - `mlirOperationGetResult`
/// - `mlirOperationHasInherentAttributeByName`
/// - `mlirOperationMoveAfter`
/// - `mlirOperationMoveBefore`
/// - `mlirOperationPrint`
/// - `mlirOperationRemoveAttributeByName`
/// - `mlirOperationRemoveDiscardableAttributeByName`
/// - `mlirOperationRemoveFromParent`
/// - `mlirOperationSetAttributeByName`
/// - `mlirOperationSetDiscardableAttributeByName`
/// - `mlirOperationSetInherentAttributeByName`
/// - `mlirOperationSetOperand`
/// - `mlirOperationVerify`
///
/// The following bindings are not used/supported:
/// - `mlirOperationDump`
/// - `mlirOperationGetNumSuccessors`
/// - `mlirOperationGetSuccessor`
/// - `mlirOperationGetTypeID`
/// - `mlirOperationImplementsInterfaceStatic`
/// - `mlirOperationImplementsInterface`
/// - `mlirOperationPrintWithFlags`
/// - `mlirOperationPrintWithState`
/// - `mlirOperationSetOperands`
/// - `mlirOperationSetSuccessor`
/// - `mlirOperationWalk`
/// - `mlirOperationWriteBytecodeWithConfig`
/// - `mlirOperationWriteBytecode`
#[repr(transparent)]
#[derive(Debug)]
pub struct Operation<'c> {
    raw: MlirOperation,
    _context: PhantomData<&'c ()>,
}

impl_owned_mlir_value!(context_ref, Operation, MlirOperation);

impl<'c> Operation<'c> {
    /// Attempts to parse an operation from the provided source string.
    ///
    /// # Arguments
    /// * `context` - The context to associate with the operation.
    /// * `source` - The source string to parse the operation from.
    /// * `source_filename` - The filename to associate with locations from the source string.
    ///
    /// # Returns
    /// Returns a new [Operation] if the operation could be parsed, otherwise `None`.
    pub fn parse(
        context: &'c ContextRef,
        source: &str,
        source_filename: &str,
    ) -> Option<Operation<'c>> {
        // MLIR Bug! The source string needs to be null-terminated despite the type being a
        // `StringRef`, which explicitly stores the length of the string & documents that it is
        // not necessarily null-terminated.
        let source = CString::new(source).expect("Failed to convert source string to CString");
        let source_ref = StringRef::from_cstring(&source).to_raw();
        let source_filename_ref = StringRef::from(&source_filename).to_raw();
        unsafe {
            Self::try_from_raw(mlirOperationCreateParse(
                context.to_raw(),
                source_ref,
                source_filename_ref,
            ))
        }
    }
}

impl<'c> Drop for Operation<'c> {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.raw) }
    }
}

impl<'c> PartialEq for Operation<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}

impl<'c> PartialEq<OperationRef<'c>> for Operation<'c> {
    fn eq(&self, other: &OperationRef<'c>) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.to_raw()) }
    }
}

impl<'c> Eq for Operation<'c> {}

/// [OperationRef] wraps the `mlir::Operation` class, which represents a single operation in the
/// MLIR IR.
///
/// # Safety
/// This type is ONLY ever safe to use if it is a **reference**! Owned instances will cause
/// undefined behaviour.
#[repr(transparent)]
#[derive(Debug)]
pub struct OperationRef<'c> {
    _context: PhantomData<&'c ()>,
}

impl_unowned_mlir_value!(context_ref, Operation, OperationRef, MlirOperation);

impl<'c> OperationRef<'c> {
    /// # Returns
    /// A new, owned [Operation] that is a clone of this operation reference.
    pub fn clone_op(&self) -> Operation<'c> {
        unsafe { Operation::from_raw(mlirOperationClone(self.to_raw())) }
    }

    /// # Returns
    /// Returns the context associated with the operation.
    pub fn context(&self) -> &'c ContextRef {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.to_raw())) }
    }

    /// Verifies the operation, emitting any diagnostics if the operation is invalid.
    ///
    /// # Returns
    /// Returns whether the operation is valid.
    pub fn verify(&self) -> bool {
        unsafe { mlirOperationVerify(self.to_raw()) }
    }

    /// # Returns
    /// Returns the name of the operation.
    pub fn name(&self) -> &IdentifierRef {
        unsafe { IdentifierRef::from_raw(mlirOperationGetName(self.to_raw())) }
    }

    /// # Returns
    /// Returns the location associated with the operation.
    pub fn location(&self) -> &LocationRef {
        unsafe { LocationRef::from_raw(mlirOperationGetLocation(self.to_raw())) }
    }

    /// # Returns
    /// If the operation is nested within another operation, returns the parent operation.
    pub fn parent_operation(&self) -> Option<&OperationRef<'c>> {
        unsafe { Self::try_from_raw(mlirOperationGetParentOperation(self.to_raw())) }
    }

    /// # Returns
    /// If the operation is nested within a block, returns the parent block.
    pub fn parent_block(&self) -> Option<&BlockRef<'c>> {
        unsafe { BlockRef::try_from_raw(mlirOperationGetBlock(self.to_raw())) }
    }

    /// Removes the operation from its parent block.
    pub fn remove_from_parent(&self) {
        unsafe { mlirOperationRemoveFromParent(self.to_raw()) }
    }

    /// Moves the operation after another operation, transferring ownership to the owner of the
    /// other operation.
    ///
    /// # Arguments
    /// * `other` - The operation to move this operation after.
    ///
    /// # Returns
    /// Returns a reference to this operation with the appropriate after being moved.
    pub fn move_after<'a, 'b>(&'a self, other: &'b OperationRef<'c>) -> &'b OperationRef<'c>
    where
        'a: 'b,
    {
        unsafe { mlirOperationMoveAfter(self.to_raw(), other.to_raw()) }
        self
    }

    /// Moves the operation before another operation, transferring ownership to the owner of the
    /// other operation.
    ///
    /// # Arguments
    /// * `other` - The operation to move this operation before.
    ///
    /// # Returns
    /// Returns a reference to this operation with the appropriate after being moved.
    pub fn move_before<'a, 'b>(&'a self, other: &'b OperationRef<'c>) -> &'b OperationRef<'c>
    where
        'a: 'b,
    {
        unsafe { mlirOperationMoveBefore(self.to_raw(), other.to_raw()) }
        self
    }

    /// # Returns
    /// If this operation is in a block and there is another operation after it in the block,
    /// returns that operation.
    pub fn next_in_parent_block(&self) -> Option<&OperationRef<'c>> {
        unsafe { Self::try_from_raw(mlirOperationGetNextInBlock(self.to_raw())) }
    }

    /// # Returns
    /// Returns the number of operands the operation has.
    pub fn num_operands(&self) -> isize {
        unsafe { mlirOperationGetNumOperands(self.to_raw()) }
    }

    /// Gets the operand at the provided index, verifying that the index is within bounds.
    ///
    /// # Arguments
    /// * `idx` - The index of the operand to get.
    ///
    /// # Returns
    /// Returns a reference to the operand value.
    pub fn operand(&self, idx: isize) -> &ValueRef<'c> {
        if idx >= self.num_operands() {
            panic!("Operand index {} out of bounds.", idx);
        }
        unsafe { ValueRef::from_raw(mlirOperationGetOperand(self.to_raw(), idx)) }
    }

    /// Sets the operand at the provided index to the specified value, verifying that the index is
    /// within bounds.
    ///
    /// # Arguments
    /// * `idx` - The index of the operand to set.
    /// * `new_value` - The new value to set the operand to.
    pub fn set_operand(&self, idx: isize, new_value: &ValueRef<'c>) {
        if idx >= self.num_operands() {
            panic!("Operand index {} out of bounds.", idx);
        }
        unsafe { mlirOperationSetOperand(self.to_raw(), idx, new_value.to_raw()) }
    }

    /// # Returns
    /// Returns the number of regions the operation has.
    pub fn num_regions(&self) -> isize {
        unsafe { mlirOperationGetNumRegions(self.to_raw()) }
    }

    /// Gets the region at the provided index, verifying that the index is within bounds.
    ///
    /// # Arguments
    /// * `idx` - The index of the region to get.
    ///
    /// # Returns
    /// Returns a reference to the region.
    pub fn region(&self, idx: isize) -> &RegionRef<'c> {
        if idx >= self.num_regions() {
            panic!("Region index {} out of bounds.", idx);
        }
        unsafe { RegionRef::from_raw(mlirOperationGetRegion(self.to_raw(), idx)) }
    }

    /// # Returns
    /// Returns the first region of the operation, if it has one.
    pub fn first_region(&self) -> Option<&RegionRef<'c>> {
        unsafe { RegionRef::try_from_raw(mlirOperationGetFirstRegion(self.to_raw())) }
    }

    /// # Returns
    /// Returns the number of results the operation has.
    pub fn num_results(&self) -> isize {
        unsafe { mlirOperationGetNumResults(self.to_raw()) }
    }

    /// Gets the result at the provided index, verifying that the index is within bounds.
    ///
    /// # Arguments
    /// * `idx` - The index of the result to get.
    ///
    /// # Returns
    /// Returns a reference to the result value.
    pub fn result(&self, idx: isize) -> &ValueRef<'c> {
        if idx >= self.num_results() {
            panic!("Result index {} out of bounds.", idx);
        }
        unsafe { ValueRef::from_raw(mlirOperationGetResult(self.to_raw(), idx)) }
    }

    /// Checks if the operation has an inherent attribute with the specified name.
    ///
    /// # Arguments
    /// * `name` - The name of the attribute to check for.
    ///
    /// # Returns
    /// Returns whether an inherent attribute with the specified name exists.
    pub fn has_inherent_attribute(&self, name: &str) -> bool {
        unsafe {
            mlirOperationHasInherentAttributeByName(self.to_raw(), StringRef::from(&name).to_raw())
        }
    }

    /// Gets the inherent attribute with the specified name, if it exists.
    ///
    /// # Arguments
    /// * `name` - The name of the attribute to get.
    ///
    /// # Returns
    /// Returns a reference to the inherent attribute, if it exists.
    pub fn inherent_attribute(&self, name: &str) -> Option<&AttributeRef> {
        unsafe {
            AttributeRef::try_from_raw(mlirOperationGetInherentAttributeByName(
                self.to_raw(),
                StringRef::from(&name).to_raw(),
            ))
        }
    }

    /// Sets the inherent attribute with the specified name to the provided value.
    ///
    /// # Arguments
    /// * `name` - The name of the attribute to set.
    /// * `value` - The value to set the attribute to.
    pub fn set_inherent_attribute(&self, name: &str, value: &'c AttributeRef) {
        unsafe {
            mlirOperationSetInherentAttributeByName(
                self.to_raw(),
                StringRef::from(&name).to_raw(),
                value.to_raw(),
            )
        }
    }

    /// # Returns
    /// Returns the number of discardable attributes the operation has.
    pub fn num_discardable_attributes(&self) -> isize {
        unsafe { mlirOperationGetNumDiscardableAttributes(self.to_raw()) }
    }

    /// Gets the discardable attribute at the provided index, verifying that the index is within
    /// bounds.
    ///
    /// # Arguments
    /// * `idx` - The index of the discardable attribute to get.
    ///
    /// # Returns
    /// Returns the discardable attribute as a name-value pair.
    pub fn discardable_attribute_at(&self, idx: isize) -> NamedAttribute {
        if idx >= self.num_discardable_attributes() {
            panic!("Discardable attribute index {} out of bounds.", idx);
        }
        unsafe {
            let discardable_attribute = mlirOperationGetDiscardableAttribute(self.to_raw(), idx);
            NamedAttribute::from_raw(discardable_attribute.name, discardable_attribute.attribute)
        }
    }

    /// Gets the discardable attribute with the specified name, if it exists.
    ///
    /// # Arguments
    /// * `name` - The name of the attribute to get.
    ///
    /// # Returns
    /// Returns a reference to the discardable attribute, if it exists.
    pub fn discardable_attribute(&self, name: &str) -> Option<&AttributeRef> {
        unsafe {
            AttributeRef::try_from_raw(mlirOperationGetDiscardableAttributeByName(
                self.to_raw(),
                StringRef::from(&name).to_raw(),
            ))
        }
    }

    /// Sets the discardable attribute with the specified name to the provided value.
    ///
    /// # Arguments
    /// * `name` - The name of the attribute to set.
    pub fn set_discardable_attribute(&self, name: &str, value: &'c AttributeRef) {
        unsafe {
            mlirOperationSetDiscardableAttributeByName(
                self.to_raw(),
                StringRef::from(&name).to_raw(),
                value.to_raw(),
            )
        }
    }

    /// Removes the discardable attribute with the specified name.
    ///
    /// # Arguments
    /// * `name` - The name of the attribute to remove.
    ///
    /// # Returns
    /// Returns whether the attribute was removed (i.e. if it existed).
    pub fn remove_discardable_attribute(&self, name: &str) -> bool {
        unsafe {
            mlirOperationRemoveDiscardableAttributeByName(
                self.to_raw(),
                StringRef::from(&name).to_raw(),
            )
        }
    }

    /// # Returns
    /// Returns the number of attributes the operation has.
    pub fn num_attributes(&self) -> isize {
        unsafe { mlirOperationGetNumAttributes(self.to_raw()) }
    }

    /// Gets the attribute at the provided index, verifying that the index is within bounds.
    ///
    /// # Arguments
    /// * `idx` - The index of the attribute to get.
    ///
    /// # Returns
    /// Returns the attribute as a name-value pair.
    pub fn attribute_at(&self, idx: isize) -> NamedAttribute<'c> {
        if idx >= self.num_attributes() {
            panic!("Attribute index {} out of bounds.", idx);
        }
        unsafe {
            let attribute = mlirOperationGetAttribute(self.to_raw(), idx);
            NamedAttribute::from_raw(attribute.name, attribute.attribute)
        }
    }

    /// Gets the attribute with the specified name, if it exists.
    ///
    /// # Arguments
    /// * `name` - The name of the attribute to get.
    ///
    /// # Returns
    /// Returns a reference to the attribute, if it exists.
    pub fn attribute(&self, name: &str) -> Option<&'c AttributeRef> {
        unsafe {
            AttributeRef::try_from_raw(mlirOperationGetAttributeByName(
                self.to_raw(),
                StringRef::from(&name).to_raw(),
            ))
        }
    }

    /// Sets the attribute with the specified name to the provided value.
    ///
    /// # Arguments
    /// * `name` - The name of the attribute to set.
    /// * `value` - The value to set the attribute to.
    pub fn set_attribute(&self, name: &str, value: &'c AttributeRef) {
        unsafe {
            mlirOperationSetAttributeByName(
                self.to_raw(),
                StringRef::from(&name).to_raw(),
                value.to_raw(),
            )
        }
    }

    /// Removes the attribute with the specified name.
    ///
    /// # Arguments
    /// * `name` - The name of the attribute to remove.
    ///
    /// # Returns
    /// Returns whether the attribute was removed (i.e. if it existed).
    pub fn remove_attribute(&self, name: &str) -> bool {
        unsafe {
            mlirOperationRemoveAttributeByName(self.to_raw(), StringRef::from(&name).to_raw())
        }
    }
}

pub fn walk(
    op: &OperationRef,
    callback: MlirOperationWalkCallback,
    user_data: *mut c_void,
) {
    unsafe { mlirOperationWalk(op.to_raw(), callback, user_data, 1) }
}

pub type OperationWalkCallback =
    Option<unsafe extern "C" fn(op: MlirOperation, user_data: *mut c_void) -> c_void>;

impl<'c> PartialEq for OperationRef<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.to_raw(), other.to_raw()) }
    }
}

impl<'c> PartialEq<Operation<'c>> for OperationRef<'c> {
    fn eq(&self, other: &Operation<'c>) -> bool {
        unsafe { mlirOperationEqual(self.to_raw(), other.to_raw()) }
    }
}

impl<'c> Eq for OperationRef<'c> {}

impl<'c> Display for OperationRef<'c> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut reader = StringReader::new(f);
        unsafe { mlirOperationPrint(self.to_raw(), reader.callback(), reader.as_raw_mut()) }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn parse() {
        #[rustfmt::skip]
        let operation_source = r#"
module {
}
"#.trim_start();
        let context = Context::new(None, false);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        assert_eq!(operation.to_string(), operation_source);
    }

    #[test]
    fn context() {
        let operation_source = "module {}";
        let context = Context::new(None, false);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        assert_eq!(operation.context(), &context);
    }

    #[test]
    fn name() {
        let operation_source = "module {}";
        let context = Context::new(None, false);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        // TODO: Why does as_str() need to be called here?
        // Without it, the left hand side of the comparison is the StringRef's debug representation.
        assert_eq!(operation.name().value().as_str(), "builtin.module");
    }

    #[test]
    fn location() {
        let operation_source = "module {}";
        let context = Context::new(None, false);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        assert_eq!(
            operation.location(),
            LocationRef::new_file_line_col(&context, "test.mlir", 1, 1),
        );
    }

    #[test]
    fn parent_operation() {
        #[rustfmt::skip]
        let operation_source = r#"
module {
    "dialect.op"() : () -> ()
}
"#;
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        assert!(operation.parent_operation().is_none());
        let first_op = operation
            .region(0)
            .first_block()
            .unwrap()
            .first_operation()
            .unwrap();
        assert_eq!(first_op.parent_operation().unwrap(), &operation);
    }

    #[test]
    fn parent_block() {
        #[rustfmt::skip]
        let operation_source = r#"
module {
    "dialect.op"() : () -> ()
}
"#;
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        assert!(operation.parent_block().is_none());
        let first_block = operation.region(0).first_block().unwrap();
        let first_nested_op = first_block.first_operation().unwrap();
        assert_eq!(first_nested_op.parent_block().unwrap(), first_block);
    }

    #[test]
    fn remove_from_parent() {
        #[rustfmt::skip]
        let operation_source = r#"
module {
    "dialect.op"() : () -> ()
}
"#;
        let resulting_operation = r#"
module {
}
"#
        .trim_start();
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        let first_operation = operation
            .region(0)
            .first_block()
            .unwrap()
            .first_operation()
            .unwrap();
        first_operation.remove_from_parent();
        assert_eq!(operation.to_string(), resulting_operation);
    }

    // TODO: test `move_after` and `move_before`

    #[test]
    fn next_in_parent_block() {
        #[rustfmt::skip]
        let operation_source = r#"
module {
    "dialect.op1"() : () -> ()
    "dialect.op2"() : () -> ()
}
"#;
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        assert!(operation.next_in_parent_block().is_none());
        let first_op = operation
            .region(0)
            .first_block()
            .unwrap()
            .first_operation()
            .unwrap();
        let second_op = first_op.next_in_parent_block().unwrap();
        // TODO: Why does as_str() need to be called here?
        assert_eq!(second_op.name().value().as_str(), "dialect.op2");
        assert!(second_op.next_in_parent_block().is_none());
    }

    #[test]
    fn num_operands() {
        #[rustfmt::skip]
        let operation_source = r#"
module {
    %0:2 = "dialect.op1"() {"attribute name" = 42 : i32} : () -> (i1, i16)
    "dialect.op2"(%0#0, %0#1) : (i1, i16) -> ()
}
"#;
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        let first_op = operation
            .region(0)
            .first_block()
            .unwrap()
            .first_operation()
            .unwrap();
        assert_eq!(first_op.num_operands(), 0);
        let second_op = first_op.next_in_parent_block().unwrap();
        assert_eq!(second_op.num_operands(), 2);
    }

    #[test]
    fn get_operands() {
        #[rustfmt::skip]
        let operation_source = r#"
module {
    %0:2 = "dialect.op1"() {"attribute name" = 42 : i32} : () -> (i1, i16)
    "dialect.op2"(%0#0, %0#1) : (i1, i16) -> ()
}
"#;
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        let _region = operation.region(0);
        let second_op = operation
            .region(0)
            .first_block()
            .unwrap()
            .first_operation()
            .unwrap()
            .next_in_parent_block()
            .unwrap();
        assert_eq!(second_op.operand(0).r#type().to_string(), "i1");
        assert_eq!(second_op.operand(1).r#type().to_string(), "i16");
    }

    #[test]
    #[should_panic]
    fn get_operand_out_of_bounds() {
        #[rustfmt::skip]
        let operation_source = r#"
module {
    %0:2 = "dialect.op1"() {"attribute name" = 42 : i32} : () -> (i1, i16)
    "dialect.op2"(%0#0, %0#1) : (i1, i16) -> ()
}
"#;
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let operation = Operation::parse(&context, operation_source, "test.mlir").unwrap();
        let second_op = operation
            .region(0)
            .first_block()
            .unwrap()
            .first_operation()
            .unwrap()
            .next_in_parent_block()
            .unwrap();
        let _third_operand = second_op.operand(2);
    }

    #[test]
    #[should_panic]
    fn no_owned_operation_ref() {
        let _operation_ref = OperationRef {
            _context: PhantomData,
        };
    }
}
