use crate::{
    ir::{LocationRef, NamedAttribute, Operation, Region, TypeRef, ValueRef},
    support::binding::OwnedMlirValue,
    StringRef, UnownedMlirValue,
};

use std::{marker::PhantomData, mem::forget};

use mlir_sys::{
    mlirOperationCreate, mlirOperationStateAddAttributes, mlirOperationStateAddOperands,
    mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateEnableResultTypeInference, mlirOperationStateGet, MlirNamedAttribute,
    MlirOperationState, MlirRegion, MlirType, MlirValue,
};

pub struct OperationBuilder<'a> {
    state: MlirOperationState,
    _context: PhantomData<&'a ()>,
}

impl<'a> OperationBuilder<'a> {
    pub fn new(name: &str, location: &'a LocationRef) -> OperationBuilder<'a> {
        Self {
            state: unsafe {
                mlirOperationStateGet(StringRef::from(&name).to_raw(), location.to_raw())
            },
            _context: PhantomData,
        }
    }

    pub fn add_results(mut self, types: &[&TypeRef]) -> Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.state as *mut MlirOperationState,
                types.len() as isize,
                types.as_ptr() as *const MlirType,
            );
        }
        self
    }

    pub fn add_operands(mut self, operands: &[&ValueRef]) -> Self {
        unsafe {
            mlirOperationStateAddOperands(
                &mut self.state as *mut MlirOperationState,
                operands.len() as isize,
                operands.as_ptr() as *const MlirValue,
            );
        }
        self
    }

    pub fn add_regions(mut self, regions: Vec<Region>) -> Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(
                &mut self.state as *mut MlirOperationState,
                regions.len() as isize,
                regions.as_ptr() as *const MlirRegion,
            )
        }
        forget(regions);
        self
    }

    // TODO: Add support for successors.

    pub fn add_attributes(mut self, attributes: &[NamedAttribute]) -> Self {
        unsafe {
            mlirOperationStateAddAttributes(
                &mut self.state as *mut MlirOperationState,
                attributes.len() as isize,
                attributes.as_ptr() as *const MlirNamedAttribute,
            );
        }
        self
    }

    pub fn enable_result_type_inference(mut self) -> Self {
        unsafe {
            mlirOperationStateEnableResultTypeInference(&mut self.state as *mut MlirOperationState)
        }
        self
    }

    pub fn build(mut self) -> Option<Operation<'a>> {
        unsafe {
            Operation::try_from_raw(mlirOperationCreate(
                &mut self.state as *mut MlirOperationState,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::AttributeRef;
    use crate::{ir::LocationRef, Context};

    #[test]
    fn build() {
        let context = Context::new(None, false);
        context.set_allow_unregistered_dialects(true);
        let loc = LocationRef::new_unknown(&context);
        let op = OperationBuilder::new("dialect.op1", loc)
            .add_attributes(&[AttributeRef::parse(&context, "42 : i32")
                .unwrap()
                .with_name("attribute name")])
            .add_results(&[
                TypeRef::parse(&context, "i1").unwrap(),
                TypeRef::parse(&context, "i16").unwrap(),
            ])
            .build()
            .unwrap();
        assert_eq!(
            op.to_string(),
            r#"%0:2 = "dialect.op1"() {"attribute name" = 42 : i32} : () -> (i1, i16)
"#
        );
    }
}
