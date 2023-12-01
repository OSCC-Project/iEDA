use std::default::Default;

/// liberty expression operation.
#[repr(C)]
#[derive(Clone, Copy)]
pub enum LibertyExprOp {
    Buffer,
    Not,
    Or,
    And,
    Xor,
    One,
    Zero,
    Plus,
    Mult,
}

/// liberty expr.
#[repr(C)]
pub struct LibertyExpr {
    op: LibertyExprOp,
    left: Option<Box<LibertyExpr>>,
    right: Option<Box<LibertyExpr>>,
    port_name: Option<String>,
}

impl LibertyExpr {
    pub fn new(op: LibertyExprOp) -> Self {
        Self { op, left: Option::None, right: Option::None, port_name: Option::None }
    }

    pub fn get_op(&self) -> LibertyExprOp {
        self.op
    }

    pub fn set_port_name(&mut self, port_name: String) {
        self.port_name = Some(port_name);
    }
    pub fn get_port_name(&self) -> &Option<String> {
        return &self.port_name;
    }

    pub fn set_left(&mut self, left: Box<LibertyExpr>) {
        self.left = Some(left);
    }
    pub fn get_left(&self) -> &Option<Box<LibertyExpr>> {
        return &self.left;
    }

    pub fn set_right(&mut self, right: Box<LibertyExpr>) {
        self.right = Some(right);
    }
    pub fn get_right(&self) -> &Option<Box<LibertyExpr>> {
        return &self.right;
    }
}
