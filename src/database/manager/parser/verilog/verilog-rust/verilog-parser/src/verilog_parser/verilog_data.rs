use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::rc::Rc;

pub trait VerilogVirtualBaseID: Debug + VerilogVirtualBaseIDClone {
    fn is_id(&self) -> bool {
        false
    }

    fn is_bus_index_id(&self) -> bool {
        false
    }

    fn is_bus_slice_id(&self) -> bool {
        false
    }

    fn is_constant_id(&self) -> bool {
        false
    }

    fn get_name(&self) -> &str {
        panic!("This is unknown value.");
    }

    fn get_base_name(&self) -> &str {
        panic!("This is unknown value.");
    }

    fn set_base_name(&mut self, new_base_name: &str);

    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait VerilogVirtualBaseIDClone {
    fn clone_box(&self) -> Box<dyn VerilogVirtualBaseID>;
}

impl<T> VerilogVirtualBaseIDClone for T
where
    T: 'static + VerilogVirtualBaseID + Clone,
{
    fn clone_box(&self) -> Box<dyn VerilogVirtualBaseID> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn VerilogVirtualBaseID> {
    fn clone(&self) -> Box<dyn VerilogVirtualBaseID> {
        self.clone_box()
    }
}

/// verilog id.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct VerilogID {
    id: String,
}

impl VerilogID {
    pub fn new(id: &str) -> VerilogID {
        VerilogID { id: id.to_string() }
    }

    pub fn get_base_name(&self) -> &str {
        &self.id
    }
}

impl VerilogVirtualBaseID for VerilogID {
    fn is_id(&self) -> bool {
        true
    }

    fn get_name(&self) -> &str {
        &self.id
    }

    fn get_base_name(&self) -> &str {
        &self.id
    }

    fn set_base_name(&mut self, id: &str) {
        self.id = id.to_string();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}


#[derive(Debug, Clone)]
pub struct VerilogIndexID {
    id: VerilogID,
    index: i32,
    formatted_index_id: String,
}

impl VerilogIndexID {
    pub fn new(id: &str, index: i32) -> VerilogIndexID {
        let formatted_index_id = format!("{}[{}]", id, index);
        VerilogIndexID { id: VerilogID::new(id), index, formatted_index_id }
    }

    #[allow(dead_code)]
    pub fn get_id(&self) -> &VerilogID {
        &self.id
    }

    pub fn get_index(&self) -> i32 {
        self.index
    }
}

impl VerilogVirtualBaseID for VerilogIndexID {
    fn is_bus_index_id(&self) -> bool {
        true
    }

    fn get_base_name(&self) -> &str {
        self.id.get_base_name()
    }

    fn get_name(&self) -> &str {
        &self.formatted_index_id
    }

    fn set_base_name(&mut self, id: &str) {
        self.id.set_base_name(id);
        self.formatted_index_id = format!("{}[{}]", self.id.get_base_name(), self.index);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct VerilogSliceID {
    id: VerilogID,
    range_from: i32,
    range_to: i32,
    formatted_slice_id: String,
}

impl VerilogSliceID {
    pub fn new(id: &str, range_from: i32, range_to: i32) -> VerilogSliceID {
        let formatted_slice_id = format!("{}[{}:{}]", id, range_from, range_to);
        VerilogSliceID {
            id: VerilogID::new(id),
            range_from,
            range_to,
            formatted_slice_id,
        }
    }

    #[allow(dead_code)]
    pub fn get_id(&self) -> &VerilogID {
        &self.id
    }

    pub fn get_range_from(&self) -> i32 {
        self.range_from
    }

    pub fn get_range_to(&self) -> i32 {
        self.range_to
    }

    #[allow(dead_code)]
    pub fn get_name_with_index(&self, index: i32) -> String {
        format!("{}[{}]", &self.id.get_base_name(), index)
    }

    pub fn set_range_from(&mut self, new_range_from: i32) {
        self.range_from = new_range_from;
        self.update_formatted_slice_id();
    }

    pub fn set_range_to(&mut self, new_range_to: i32) {
        self.range_to = new_range_to;
        self.update_formatted_slice_id();
    }

    fn update_formatted_slice_id(&mut self) {
        self.formatted_slice_id = format!("{}[{}:{}]", self.id.get_base_name(), self.range_from, self.range_to);
    }
}

impl VerilogVirtualBaseID for VerilogSliceID {
    fn is_bus_slice_id(&self) -> bool {
        true
    }

    fn get_base_name(&self) -> &str {
        self.id.get_base_name()
    }

    fn get_name(&self) -> &str {
        &self.formatted_slice_id
    }

    fn set_base_name(&mut self, id: &str) {
        self.id.set_base_name(id);
        self.formatted_slice_id = format!("{}[{}:{}]", self.id.get_base_name(), self.range_from, self.range_to);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct VerilogConstantID {
    bit_width: u32,
    value: VerilogID,
    formatted_constant_id: String,
}

impl VerilogConstantID {
    pub fn new(bit_width: u32, value: &str) -> VerilogConstantID {
        let formatted_constant_id = format!("{}'{}", bit_width, value);
        VerilogConstantID {
            bit_width,
            value: VerilogID::new(value),
            formatted_constant_id,
        }
    }

    pub fn get_bit_width(&self) -> u32 {
        self.bit_width
    }
    #[allow(dead_code)]
    pub fn get_value(&self) -> &VerilogID {
        &self.value
    }
}

impl VerilogVirtualBaseID for VerilogConstantID {
    fn is_constant_id(&self) -> bool {
        true
    }

    fn get_name(&self) -> &str {
        &self.formatted_constant_id
    }

    fn get_base_name(&self) -> &str {
        self.value.get_base_name()
    }

    fn set_base_name(&mut self, id: &str) {
        self.value.set_base_name(id);
        self.formatted_constant_id = format!("{}'{}", self.bit_width, self.value.get_base_name());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub trait VerilogVirtualBaseNetExpr: Debug + VerilogVirtualBaseNetExprClone {
    fn is_id_expr(&self) -> bool {
        false
    }
    fn is_concat_expr(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        false
    }
    fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        panic!("This is unknown value.");
    }

    fn get_line_no(&self) -> usize {
        panic!("This is unknown value.");
    }

    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait VerilogVirtualBaseNetExprClone {
    fn clone_box(&self) -> Box<dyn VerilogVirtualBaseNetExpr>;
}

impl<T> VerilogVirtualBaseNetExprClone for T
where
    T: 'static + VerilogVirtualBaseNetExpr + Clone,
{
    fn clone_box(&self) -> Box<dyn VerilogVirtualBaseNetExpr> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn VerilogVirtualBaseNetExpr> {
    fn clone(&self) -> Box<dyn VerilogVirtualBaseNetExpr> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct VerilogNetExpr {
    line_no: usize,
}

impl VerilogNetExpr {
    fn new(line_no: usize) -> VerilogNetExpr {
        VerilogNetExpr { line_no }
    }

    pub fn get_line_no(&self) -> usize {
        self.line_no
    }
}

#[derive(Debug, Clone)]
pub struct VerilogNetIDExpr {
    net_expr: VerilogNetExpr,
    verilog_id: Box<dyn VerilogVirtualBaseID>,
}

impl VerilogNetIDExpr {
    pub fn new(line_no: usize, verilog_id: Box<dyn VerilogVirtualBaseID>) -> VerilogNetIDExpr {
        VerilogNetIDExpr { net_expr: VerilogNetExpr::new(line_no), verilog_id }
    }

    pub fn get_net_expr(&self) -> &VerilogNetExpr {
        &self.net_expr
    }
    pub fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.verilog_id
    }

    pub fn set_verilog_id(&mut self, new_verilog_id: Box<dyn VerilogVirtualBaseID>) {
        self.verilog_id = new_verilog_id;
    }
}

impl VerilogVirtualBaseNetExpr for VerilogNetIDExpr {
    fn is_id_expr(&self) -> bool {
        true
    }
    fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.verilog_id
    }
    fn get_line_no(&self) -> usize {
        self.net_expr.get_line_no()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone)]
/// such as { 2'b00, _0_ } or  wire   [23:0] buf11_dout in dcl.
pub struct VerilogNetConcatExpr {
    net_expr: VerilogNetExpr,
    verilog_id_concat: Vec<Box<dyn VerilogVirtualBaseNetExpr>>,
}

impl VerilogNetConcatExpr {
    pub fn new(line_no: usize, verilog_id_concat: Vec<Box<dyn VerilogVirtualBaseNetExpr>>) -> VerilogNetConcatExpr {
        VerilogNetConcatExpr { net_expr: VerilogNetExpr::new(line_no), verilog_id_concat }
    }

    pub fn get_net_expr(&self) -> &VerilogNetExpr {
        &self.net_expr
    }

    pub fn get_verilog_id_concat(&self) -> &Vec<Box<dyn VerilogVirtualBaseNetExpr>> {
        &self.verilog_id_concat
    }
    #[allow(dead_code)]
    pub fn update_verilog_id_concat(&mut self, new_verilog_id_concat: Vec<Box<dyn VerilogVirtualBaseNetExpr>>) {
        self.verilog_id_concat = new_verilog_id_concat;
    }
}

impl VerilogVirtualBaseNetExpr for VerilogNetConcatExpr {
    fn is_concat_expr(&self) -> bool {
        true
    }

    fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        self.verilog_id_concat.first().unwrap().get_verilog_id()
    }

    fn get_line_no(&self) -> usize {
        self.net_expr.get_line_no()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone)]
/// 1'b0 or 1'b1.
pub struct VerilogConstantExpr {
    net_expr: VerilogNetExpr,
    verilog_id: Box<dyn VerilogVirtualBaseID>,
}

impl VerilogConstantExpr {
    pub fn new(line_no: usize, verilog_id: Box<dyn VerilogVirtualBaseID>) -> VerilogConstantExpr {
        VerilogConstantExpr { net_expr: VerilogNetExpr::new(line_no), verilog_id }
    }

    pub fn get_net_expr(&self) -> &VerilogNetExpr {
        &self.net_expr
    }
    pub fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.verilog_id
    }
}

impl VerilogVirtualBaseNetExpr for VerilogConstantExpr {
    fn is_constant(&self) -> bool {
        true
    }
    fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.verilog_id
    }
    fn get_line_no(&self) -> usize {
        self.net_expr.get_line_no()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The port connection such as .port_id(net_id).
#[derive(Debug)]
#[derive(Clone)]
pub struct VerilogPortRefPortConnect {
    port_id: Box<dyn VerilogVirtualBaseID>,
    net_expr: Option<Box<dyn VerilogVirtualBaseNetExpr>>,
}

impl VerilogPortRefPortConnect {
    pub fn new(
        port_id: Box<dyn VerilogVirtualBaseID>,
        net_expr: Option<Box<dyn VerilogVirtualBaseNetExpr>>,
    ) -> VerilogPortRefPortConnect {
        VerilogPortRefPortConnect { port_id, net_expr }
    }

    pub fn get_port_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.port_id
    }

    pub fn get_net_expr(&self) -> &Option<Box<dyn VerilogVirtualBaseNetExpr>> {
        &self.net_expr
    }

    pub fn set_net_expr(&mut self, new_net_expr: Option<Box<dyn VerilogVirtualBaseNetExpr>>) {
        self.net_expr = new_net_expr;
    }
}

pub trait VerilogVirtualBaseStmt: Debug + VerilogVirtualBaseStmtClone {
    fn is_module_inst_stmt(&self) -> bool {
        false
    }
    fn is_module_assign_stmt(&self) -> bool {
        false
    }
    fn is_verilog_dcl_stmt(&self) -> bool {
        false
    }
    fn is_verilog_dcls_stmt(&self) -> bool {
        false
    }
    fn is_module_stmt(&self) -> bool {
        false
    }
    fn get_line_no(&self) -> usize {
        panic!("This is unknown value.");
    }
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait VerilogVirtualBaseStmtClone {
    fn clone_box(&self) -> Box<dyn VerilogVirtualBaseStmt>;
}

impl<T> VerilogVirtualBaseStmtClone for T
where
    T: 'static + VerilogVirtualBaseStmt + Clone,
{
    fn clone_box(&self) -> Box<dyn VerilogVirtualBaseStmt> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn VerilogVirtualBaseStmt> {
    fn clone(&self) -> Box<dyn VerilogVirtualBaseStmt> {
        self.clone_box()
    }
}

/// The base class for verilog stmt,include module dcl,module dcls,module instance, module assign.
/// maybe dont need the base class***************************************
#[derive(Debug)]
#[derive(Clone)]
pub struct VerilogStmt {
    line_no: usize,
}

impl VerilogStmt {
    fn new(line_no: usize) -> VerilogStmt {
        VerilogStmt { line_no }
    }

    pub fn get_line_no(&self) -> usize {
        self.line_no
    }
}

#[derive(Debug)]
#[derive(Clone)]
pub struct VerilogInst {
    stmt: VerilogStmt, //stmt denote line_no
    inst_name: String,
    cell_name: String,
    port_connections: Vec<Box<VerilogPortRefPortConnect>>,
}

impl VerilogInst {
    pub fn new(
        line_no: usize,
        inst_name: &str,
        cell_name: &str,
        port_connections: Vec<Box<VerilogPortRefPortConnect>>,
    ) -> VerilogInst {
        VerilogInst {
            stmt: VerilogStmt::new(line_no),
            inst_name: inst_name.to_string(), // add for other to do
            cell_name: cell_name.to_string(),
            port_connections,
        }
    }

    pub fn get_line_no(&self) -> usize {
        self.stmt.get_line_no()
    }

    pub fn get_stmt(&self) -> &VerilogStmt {
        &self.stmt
    }

    pub fn get_inst_name(&self) -> &str {
        &self.inst_name
    }

    pub fn get_cell_name(&self) -> &str {
        &self.cell_name
    }

    pub fn get_port_connections(&self) -> &Vec<Box<VerilogPortRefPortConnect>> {
        &self.port_connections
    }

    pub fn set_inst_name(&mut self, new_inst_name: &str) {
        self.inst_name = new_inst_name.to_string();
    }

    fn get_concat_connect_net(
        &self,
        parent_module: &Rc<RefCell<VerilogModule>>,
        port_bus_wide_range: Option<(i32, i32)>,
        port_concat_connect_net: &VerilogNetConcatExpr,
        port_index: i32,
    ) -> Box<dyn VerilogVirtualBaseNetExpr> {
        let concat_expr_nets = port_concat_connect_net.get_verilog_id_concat();
        assert!(port_bus_wide_range.is_some(), "Fatal error: port_bus_wide_range is None");

        let bus_range_min = std::cmp::min(port_bus_wide_range.unwrap().0, port_bus_wide_range.unwrap().1);
        // bus_range_max is the bus max beyond range.
        let mut bus_range_max = std::cmp::max(port_bus_wide_range.unwrap().0, port_bus_wide_range.unwrap().1) + 1;

        let mut const_net_bit_index: Option<u32> = None;
        let mut net_index: Option<i32> = None;
        let mut connect_net_expr: Option<Box<dyn VerilogVirtualBaseNetExpr>> = None;

        for expr_net in concat_expr_nets {
            if expr_net.get_verilog_id().is_bus_index_id() {
                bus_range_max -= 1;
            } else if expr_net.get_verilog_id().is_constant_id() {
                let mut bit_width =
                    expr_net.get_verilog_id().as_any().downcast_ref::<VerilogConstantID>().unwrap().get_bit_width();
                while bit_width > 0 {
                    bus_range_max -= 1;
                    if bus_range_max == port_index {
                        const_net_bit_index = Some(bit_width - 1);
                        break;
                    }
                    bit_width -= 1;
                }
            } else if expr_net.get_verilog_id().is_bus_slice_id() {
                let slice_id = expr_net.get_verilog_id().as_any().downcast_ref::<VerilogSliceID>().unwrap();
                let from = slice_id.get_range_from();
                let to = slice_id.get_range_to();
                let mut j = from;
                while (from > to && j >= to) || (from <= to && j <= to) {
                    bus_range_max -= 1;
                    if bus_range_max == port_index {
                        net_index = Some(j);
                        break;
                    }
                    if from > to {
                        j -= 1;
                    } else {
                        j += 1;
                    }
                }
            } else {
                let borrowed_parent_module = parent_module.borrow();
                let stmt = borrowed_parent_module.find_dcls_stmt(expr_net.get_verilog_id().get_base_name());
                if stmt.is_none() {
                    println!("not found dcl stmt {}", expr_net.get_verilog_id().get_base_name());
                }
                if stmt.is_some() {
                    if stmt.unwrap().is_verilog_dcls_stmt() {
                        let verilog_dcls_stmt = stmt.unwrap().as_any().downcast_ref::<VerilogDcls>().unwrap();
                        for verilog_dcl in verilog_dcls_stmt.get_verilog_dcls() {
                            if verilog_dcl.get_dcl_name().eq(expr_net.get_verilog_id().get_base_name()) {
                                let range = verilog_dcl.get_range();
                                if let Some(range) = range {
                                    let mut j = range.0;
                                    while (range.0 > range.1 && j >= range.1) || (range.0 <= range.1 && j <= range.1) {
                                        bus_range_max -= 1;
                                        if bus_range_max == port_index {
                                            net_index = Some(j);
                                            break;
                                        }
                                        if range.0 > range.1 {
                                            j -= 1;
                                        } else {
                                            j += 1;
                                        }
                                    }
                                } else {
                                    bus_range_max -= 1;
                                }
                                break;
                            }
                        }
                    } else {
                        bus_range_max -= 1;
                    }
                } else {
                    bus_range_max -= 1;
                }
            }

            if bus_range_max == port_index {
                connect_net_expr = Some(expr_net.clone());
                break;
            } else if bus_range_max < bus_range_min {
                // should not beyond bus range min.
                break;
            }
        }

        if connect_net_expr.is_none() {
            panic!("not found connect net.");
        }

        if let Some(net_index) = net_index {
            let connect_net_id = connect_net_expr.as_ref().unwrap().get_verilog_id();
            let index_verilog_id = VerilogIndexID::new(connect_net_id.get_base_name(), net_index);
            let dyn_index_verilog_id: Box<dyn VerilogVirtualBaseID> = Box::new(index_verilog_id);
            let net_id_expr = VerilogNetIDExpr::new(0, dyn_index_verilog_id);
            let dyn_net_id_expr: Box<dyn VerilogVirtualBaseNetExpr> = Box::new(net_id_expr);
            dyn_net_id_expr
        } else if let Some(const_net_bit_index) = const_net_bit_index {
            let connect_net_id = connect_net_expr.as_ref().unwrap().get_verilog_id();
            let net_value = connect_net_id.get_base_name();
            let new_net_value = &net_value[1..];
            let bit_value = new_net_value.chars().nth(const_net_bit_index as usize).unwrap();
            let value = format!("{}{}", 'b', bit_value);
            let const_verilog_id = VerilogConstantID::new(1, &value);
            let dyn_const_verilog_id: Box<dyn VerilogVirtualBaseID> = Box::new(const_verilog_id);
            let net_id_expr = VerilogConstantExpr::new(0, dyn_const_verilog_id);
            let dyn_net_id_expr: Box<dyn VerilogVirtualBaseNetExpr> = Box::new(net_id_expr);
            dyn_net_id_expr
        } else {
            connect_net_expr.as_ref().unwrap().clone()
        }
    }

    fn get_dcl_range(&self, dcls_stmt: &Box<dyn VerilogVirtualBaseStmt>, dcl_name: &str) -> Option<(i32, i32)> {
        let mut range: Option<(i32, i32)> = None;

        if dcls_stmt.is_verilog_dcls_stmt() {
            if let Some(verilog_dcls_stmt) = dcls_stmt.as_any().downcast_ref::<VerilogDcls>() {
                for verilog_dcl in verilog_dcls_stmt.get_verilog_dcls() {
                    if verilog_dcl.get_dcl_name().eq(dcl_name) {
                        range = *verilog_dcl.get_range();
                        break;
                    }
                }
            }
        }
        range
    }

    pub fn get_port_connect_net(
        &self,
        cur_module: &Rc<RefCell<VerilogModule>>,
        parent_module: &Rc<RefCell<VerilogModule>>,
        port_id: Box<dyn VerilogVirtualBaseID>,
        port_bus_wide_range: Option<(i32, i32)>,
    ) -> Option<Box<dyn VerilogVirtualBaseNetExpr>> {
        // process the inst connection below.
        let mut port_connect_net_option: Option<Box<dyn VerilogVirtualBaseNetExpr>> = None;
        for port_connection in &self.port_connections {
            // for port may be splited, we need judge port full name and port base name. 
            if (port_connection.get_port_id().get_name() == port_id.get_base_name())
                || (port_connection.get_port_id().get_name() == port_id.get_name())
            {
                if let Some(net_expr) = port_connection.get_net_expr().clone() {
                    let mut port_connect_net: Box<dyn VerilogVirtualBaseNetExpr> = net_expr;
                    if port_connect_net.is_id_expr() {
                        // is not concat expr.Only port base name match need consider bus index and slice.
                        if (port_connection.get_port_id().get_name() == port_id.get_base_name())
                            && (port_id.is_bus_index_id() || port_id.is_bus_slice_id())
                        {
                            // find port range first
                            let borrowed_cur_module = cur_module.borrow();
                            let port_dcls_stmt = borrowed_cur_module.find_dcls_stmt(port_id.get_base_name()).unwrap();
                            let port_range = self.get_dcl_range(port_dcls_stmt, port_id.get_base_name());
                            port_range.expect("port_range is None");

                            if port_id.is_bus_index_id() {
                                if port_connect_net.get_verilog_id().is_bus_slice_id() {
                                    let port_connect_net_slice_id = port_connect_net
                                        .get_verilog_id()
                                        .as_any()
                                        .downcast_ref::<VerilogSliceID>()
                                        .unwrap();
                                    let mut index_gap =
                                        (port_id.as_any().downcast_ref::<VerilogIndexID>().unwrap().get_index()
                                            - port_range.unwrap().0)
                                            .abs();
                                    if port_connect_net_slice_id.get_range_from()
                                        > port_connect_net_slice_id.get_range_to()
                                    {
                                        index_gap = -index_gap;
                                    }
                                    let new_port_connect_port_id = VerilogIndexID::new(
                                        port_connect_net_slice_id.get_base_name(),
                                        port_connect_net_slice_id.get_range_from() + index_gap,
                                    );
                                    let dyn_new_port_connect_port_id: Box<dyn VerilogVirtualBaseID> =
                                        Box::new(new_port_connect_port_id);
                                    let port_connect_net_struct =
                                        port_connect_net.as_any().downcast_ref::<VerilogNetIDExpr>().unwrap();
                                    let mut port_connect_net_struct_clone = port_connect_net_struct.clone();
                                    port_connect_net_struct_clone.set_verilog_id(dyn_new_port_connect_port_id);
                                    port_connect_net = Box::new(port_connect_net_struct_clone);
                                    port_connect_net_option = Some(port_connect_net);
                                } else if !port_connect_net.get_verilog_id().is_bus_index_id() {
                                    let port_connect_name = port_connect_net.get_verilog_id().get_base_name();
                                    let borrowed_parent_module = parent_module.borrow();
                                    let port_connect_net_dcl_stmt =
                                        borrowed_parent_module.find_dcls_stmt(port_connect_name).unwrap();
                                    let port_connect_net_range_option =
                                        self.get_dcl_range(port_connect_net_dcl_stmt, port_connect_name);
                                    let port_index =
                                        port_id.as_any().downcast_ref::<VerilogIndexID>().unwrap().get_index();
                                    let mut index_gap = (port_index - port_range.unwrap().0).abs();
                                    if let Some(port_connect_net_range) = port_connect_net_range_option {
                                        if port_connect_net_range.0 > port_connect_net_range.1 {
                                            index_gap = -index_gap;
                                        }
                                    }

                                    let dyn_port_connect_port_id: Box<dyn VerilogVirtualBaseID> =
                                        if let Some(range) = port_connect_net_range_option {
                                            Box::new(VerilogIndexID::new(port_connect_name, range.0 + index_gap))
                                        } else {
                                            Box::new(VerilogID::new(port_connect_name))
                                        };
                                    let port_connect_net_struct =
                                        port_connect_net.as_any().downcast_ref::<VerilogNetIDExpr>().unwrap();
                                    let mut port_connect_net_struct_clone = port_connect_net_struct.clone();
                                    port_connect_net_struct_clone.set_verilog_id(dyn_port_connect_port_id);
                                    port_connect_net = Box::new(port_connect_net_struct_clone);
                                    port_connect_net_option = Some(port_connect_net);
                                } else {
                                    // the above used to be: else if port_connect_net.get_verilog_id().is_bus_index_id()
                                    port_connect_net_option = Some(port_connect_net);
                                }
                            } else if port_id.is_bus_slice_id() {
                                if port_connect_net.get_verilog_id().is_bus_slice_id() {
                                    let port_connect_net_slice_id = port_connect_net
                                        .get_verilog_id()
                                        .as_any()
                                        .downcast_ref::<VerilogSliceID>()
                                        .expect("port connect is not bus.");
                                    let port_slice_id = port_id.as_any().downcast_ref::<VerilogSliceID>().unwrap();
                                    let mut port_connect_net_slice_id_clone = port_connect_net_slice_id.clone();
                                    port_connect_net_slice_id_clone.set_range_from(port_slice_id.get_range_from());
                                    port_connect_net_slice_id_clone.set_range_to(port_slice_id.get_range_to());
                                    // the code below line is useful.                                    port_connect_net_slice_id = &port_connect_net_slice_id_clone;
                                    port_connect_net_option = Some(port_connect_net);
                                } else if !port_connect_net.get_verilog_id().is_bus_index_id() {
                                    let port_slice_id = port_id.as_any().downcast_ref::<VerilogSliceID>().unwrap();
                                    let new_port_connect_port_id = VerilogSliceID::new(
                                        port_connect_net.get_verilog_id().get_base_name(),
                                        port_slice_id.get_range_from(),
                                        port_slice_id.get_range_to(),
                                    );
                                    let new_dyn_port_connect_port_id = Box::new(new_port_connect_port_id);
                                    let port_connect_net_struct =
                                        port_connect_net.as_any().downcast_ref::<VerilogNetIDExpr>().unwrap();
                                    let mut port_connect_net_struct_clone = port_connect_net_struct.clone();
                                    port_connect_net_struct_clone.set_verilog_id(new_dyn_port_connect_port_id);
                                    port_connect_net = Box::new(port_connect_net_struct_clone);
                                    port_connect_net_option = Some(port_connect_net);
                                } else {
                                    port_connect_net_option = Some(port_connect_net);
                                }
                            }
                        } else {
                            port_connect_net_option = Some(port_connect_net);
                            if port_bus_wide_range.is_some() {}
                        }
                    } else if port_connect_net.is_concat_expr() {
                        // should be concat expr.
                        let port_concat_connect_net =
                            port_connect_net.as_any().downcast_ref::<VerilogNetConcatExpr>().unwrap();
                        if port_id.is_bus_index_id() {
                            let index = port_id.as_any().downcast_ref::<VerilogIndexID>().unwrap().get_index();
                            let index_port_connect = self.get_concat_connect_net(
                                parent_module,
                                port_bus_wide_range,
                                port_concat_connect_net,
                                index,
                            );
                            port_connect_net = index_port_connect;
                            port_connect_net_option = Some(port_connect_net);
                        } else if port_id.is_bus_slice_id() {
                            let port_slice_id = port_id.as_any().downcast_ref::<VerilogSliceID>().unwrap();
                            let from = port_slice_id.get_range_from();
                            let to = port_slice_id.get_range_to();
                            let mut slice_concat: Vec<Box<dyn VerilogVirtualBaseNetExpr>> = Vec::new();
                            let mut index = from;
                            while (from > to && index >= to) || (from <= to && index <= to) {
                                let index_port_connect = self.get_concat_connect_net(
                                    parent_module,
                                    port_bus_wide_range,
                                    port_concat_connect_net,
                                    index,
                                );
                                slice_concat.push(index_port_connect);
                                if from > to {
                                    index -= 1;
                                } else {
                                    index += 1;
                                }
                            }
                            let slice_concat_connect_net = VerilogNetConcatExpr::new(0, slice_concat);
                            port_connect_net = Box::new(slice_concat_connect_net);
                            port_connect_net_option = Some(port_connect_net);
                        } else {
                            port_connect_net_option = Some(port_connect_net);
                        }
                    } else if port_connect_net.is_constant() {
                        port_connect_net_option = Some(port_connect_net);
                        println!("port {} connect net is constant", port_connection.get_port_id().get_name());
                    } else {
                        panic!("not support.");
                    }
                }
                break;
            }
        }
        port_connect_net_option
    }
}

impl VerilogVirtualBaseStmt for VerilogInst {
    fn is_module_inst_stmt(&self) -> bool {
        true
    }
    fn get_line_no(&self) -> usize {
        self.stmt.get_line_no()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct VerilogAssign {
    stmt: VerilogStmt, //stmt denote line_no
    left_net_expr: Box<dyn VerilogVirtualBaseNetExpr>,
    right_net_expr: Box<dyn VerilogVirtualBaseNetExpr>,
}

#[allow(dead_code)]
impl VerilogAssign {
    pub fn new(
        line_no: usize,
        left_net_expr: Box<dyn VerilogVirtualBaseNetExpr>,
        right_net_expr: Box<dyn VerilogVirtualBaseNetExpr>,
    ) -> VerilogAssign {
        VerilogAssign { stmt: VerilogStmt::new(line_no), left_net_expr, right_net_expr }
    }

    pub fn get_stmt(&self) -> &VerilogStmt {
        &self.stmt
    }

    pub fn get_left_net_expr(&self) -> &Box<dyn VerilogVirtualBaseNetExpr> {
        &self.left_net_expr
    }

    pub fn get_right_net_expr(&self) -> &Box<dyn VerilogVirtualBaseNetExpr> {
        &self.right_net_expr
    }
}

impl VerilogVirtualBaseStmt for VerilogAssign {
    fn is_module_assign_stmt(&self) -> bool {
        true
    }
    fn get_line_no(&self) -> usize {
        self.stmt.get_line_no()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[allow(dead_code)]
#[repr(C)]
/// The wire or port declaration.
#[derive(Debug)]
#[derive(Clone, Copy)]
pub enum DclType {
    KInput = 0,
    KInout = 1,
    KOutput = 2,
    KSupply0 = 3,
    KSupply1 = 4,
    KTri = 5,
    KWand = 6,
    KWire = 7,
    KWor = 8,
}

#[repr(C)]
#[derive(Clone)]
#[derive(Debug)]
pub struct VerilogDcl {
    stmt: VerilogStmt, //stmt denote line_no
    dcl_type: DclType,
    dcl_name: String,
    range: Option<(i32, i32)>,
}

impl VerilogDcl {
    pub fn new(line_no: usize, dcl_type: DclType, dcl_name: &str, range: Option<(i32, i32)>) -> VerilogDcl {
        VerilogDcl { stmt: VerilogStmt::new(line_no), dcl_type, dcl_name: dcl_name.to_string(), range }
    }

    pub fn get_stmt(&self) -> &VerilogStmt {
        &self.stmt
    }
    pub fn get_dcl_type(&self) -> DclType {
        self.dcl_type
    }
    pub fn get_dcl_name(&self) -> &str {
        &self.dcl_name
    }
    pub fn get_range(&self) -> &Option<(i32, i32)> {
        &self.range
    }
    pub fn set_dcl_name(&mut self, new_dcl_name: &str) {
        self.dcl_name = new_dcl_name.to_string();
    }
}

impl VerilogVirtualBaseStmt for VerilogDcl {
    fn is_verilog_dcl_stmt(&self) -> bool {
        true
    }
    fn get_line_no(&self) -> usize {
        self.stmt.get_line_no()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

///The mutiple verilg dcl.
#[derive(Clone)]
#[derive(Debug)]
pub struct VerilogDcls {
    stmt: VerilogStmt, //stmt denote line_no
    verilog_dcls: Vec<Box<VerilogDcl>>,
}

impl VerilogDcls {
    pub fn new(line_no: usize, verilog_dcls: Vec<Box<VerilogDcl>>) -> VerilogDcls {
        VerilogDcls { stmt: VerilogStmt::new(line_no), verilog_dcls }
    }

    pub fn get_stmt(&self) -> &VerilogStmt {
        &self.stmt
    }
    pub fn get_verilog_dcls(&self) -> &Vec<Box<VerilogDcl>> {
        &self.verilog_dcls
    }
    #[allow(dead_code)]
    pub fn get_dcl_num(&self) -> usize {
        self.verilog_dcls.len()
    }
}

impl VerilogVirtualBaseStmt for VerilogDcls {
    fn is_verilog_dcls_stmt(&self) -> bool {
        true
    }
    fn get_line_no(&self) -> usize {
        self.stmt.get_line_no()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

///The verilog module class.
#[allow(dead_code)]
#[derive(Debug)]
pub enum PortDclType {
    KInput = 0,
    KInputWire = 1,
    KInout = 2,
    KInoutReg = 3,
    KInoutWire = 4,
    KOutput = 5,
    KOputputWire = 6,
    KOutputReg = 7,
}

#[derive(Debug, Clone)]
pub struct VerilogModule {
    stmt: VerilogStmt, //stmt denote line_no
    module_name: String,
    port_list: Vec<Box<dyn VerilogVirtualBaseID>>,
    module_stmts: Vec<Box<dyn VerilogVirtualBaseStmt>>,
}

///The verilog module class.
impl VerilogModule {
    pub fn new(
        line_no: usize,
        module_name: &str,
        port_list: Vec<Box<dyn VerilogVirtualBaseID>>,
        module_stmts: Vec<Box<dyn VerilogVirtualBaseStmt>>,
    ) -> VerilogModule {
        VerilogModule {
            stmt: VerilogStmt::new(line_no),
            module_name: module_name.to_string(),
            port_list,
            module_stmts,
        }
    }

    pub fn get_stmt(&self) -> &VerilogStmt {
        &self.stmt
    }
    pub fn get_module_name(&self) -> &str {
        &self.module_name
    }
    pub fn get_port_list(&self) -> &Vec<Box<dyn VerilogVirtualBaseID>> {
        &self.port_list
    }
    pub fn get_module_stmts(&self) -> &Vec<Box<dyn VerilogVirtualBaseStmt>> {
        &self.module_stmts
    }
    pub fn get_clone_module_stms(&self) -> Vec<Box<dyn VerilogVirtualBaseStmt>> {
        self.module_stmts.to_vec()
    }
    pub fn erase_stmt(&mut self, the_stmt: &Box<dyn VerilogVirtualBaseStmt>) {
        self.module_stmts.retain(|stmt| {
            let line_no_to_remove = the_stmt.get_line_no();
            let line_no = stmt.get_line_no();
            if line_no == line_no_to_remove {
                return false;
            }
            true
        });
    }
    pub fn is_port(&self, name: &str) -> bool {
        self.port_list.iter().any(|port| port.get_base_name() == name)
    }
    pub fn add_stmt(&mut self, module_stmt: Box<dyn VerilogVirtualBaseStmt>) {
        self.module_stmts.push(module_stmt);
    }

    pub fn find_dcls_stmt(&self, name: &str) -> Option<&Box<dyn VerilogVirtualBaseStmt>> {
        for module_stmt in &self.module_stmts {
            if module_stmt.is_verilog_dcls_stmt() {
                let dcls_stmt = module_stmt.as_any().downcast_ref::<VerilogDcls>().unwrap();
                for verilog_dcl in dcls_stmt.get_verilog_dcls() {
                    if verilog_dcl.get_dcl_name().eq(name) {
                        return Some(module_stmt);
                    }
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    pub fn find_inst_stmt(&self, inst_name: &str, cell_name: &str) -> Option<VerilogInst> {
        for module_stmt in &self.module_stmts {
            if module_stmt.is_module_inst_stmt() {
                let inst_stmt = module_stmt.as_any().downcast_ref::<VerilogInst>().unwrap();
                if inst_stmt.get_inst_name().contains(inst_name) && inst_stmt.get_cell_name().eq(cell_name) {
                    return Some(inst_stmt.clone());
                }
            }
        }
        None
    }
}

impl VerilogVirtualBaseStmt for VerilogModule {
    fn is_module_stmt(&self) -> bool {
        true
    }
    fn get_line_no(&self) -> usize {
        self.stmt.get_line_no()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct VerilogFile {
    verilog_modules: Vec<Rc<RefCell<VerilogModule>>>,
    module_map: HashMap<String, Rc<RefCell<VerilogModule>>>,
    top_module_name: String,
}

impl VerilogFile {
    pub fn new() -> VerilogFile {
        VerilogFile { verilog_modules: Vec::new(), module_map: HashMap::new(), top_module_name: String::new() }
    }

    pub fn add_module(&mut self, verilog_module: Rc<RefCell<VerilogModule>>) {
        let module_name = String::from(verilog_module.borrow().get_module_name());
        self.verilog_modules.push(verilog_module.clone());
        self.module_map.insert(module_name, verilog_module.clone());
    }

    pub fn get_verilog_modules(&self) -> &Vec<Rc<RefCell<VerilogModule>>> {
        &self.verilog_modules
    }

    pub fn get_module_map(&mut self) -> &mut HashMap<String, Rc<RefCell<VerilogModule>>> {
        &mut self.module_map
    }

    #[allow(dead_code)]
    pub fn get_module(&mut self, module_name: &str) -> Option<&mut Rc<RefCell<VerilogModule>>> {
        self.module_map.get_mut(module_name)
    }

    #[allow(dead_code)]
    pub fn get_top_module(&mut self) -> &Rc<RefCell<VerilogModule>> {
        self.module_map.get(&self.top_module_name).unwrap()
    }

    #[allow(dead_code)]
    pub fn get_top_module_name(&self) -> &str {
        &self.top_module_name
    }

    pub fn set_top_module_name(&mut self, name: &str) {
        self.top_module_name = name.to_string();
    }
}
