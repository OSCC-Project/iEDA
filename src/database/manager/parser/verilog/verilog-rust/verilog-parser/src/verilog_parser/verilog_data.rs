use std::fmt;
use std::fmt::Debug;

pub trait VerilogVirtualBaseID {
    fn is_id(&self) -> bool {
        false
    }

    fn is_bus_index_id(&self) -> bool {
        false
    }

    fn is_bus_slice_id(&self) -> bool {
        false
    }

    fn get_name(&self) -> String {
        panic!("This is unknown value.");
    }
}

impl fmt::Debug for dyn VerilogVirtualBaseID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VerilogVirtualBaseID {{ id: {:?} }}", self.get_name())
    }
}

/// verilog id.
#[derive(Debug)]
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

// impl fmt::Debug for VerilogID {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "VerilogID {{ id: {} }}", self.id)
//     }
// }

impl VerilogVirtualBaseID for VerilogID {
    fn is_id(&self) -> bool {
        true
    }

    fn get_name(&self) -> String {
        self.id.clone()
    }
}

impl Default for VerilogID {
    fn default() -> Self {
        VerilogID { id: String::new() }
    }
}

pub struct VerilogIndexID {
 id: VerilogID,
 index: i32,
}

impl VerilogIndexID {
    pub fn new(id: &str, index: i32) -> VerilogIndexID {
        VerilogIndexID {
            id: VerilogID::new(id),
            index:index
        }
    }

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
    // &str?
    fn get_name(&self) -> String {
       format!("{}[{}]", &self.id.get_base_name(), self.index)
    }
}


pub struct VerilogSliceID {
 id: VerilogID,
 range_from: i32, 
 range_to: i32,
}

impl VerilogSliceID {
    pub fn new(id: &str, range_from: i32, range_to: i32) -> VerilogSliceID {
        VerilogSliceID {
            id: VerilogID::new(id),
            range_from:range_from,
            range_to:range_to
        }
    }

    pub fn get_id(&self) -> &VerilogID {
        &self.id
    }

    pub fn get_range_from(&self) -> i32 {
        self.range_from
    }

    pub fn get_range_to(&self) -> i32 {
        self.range_to
    }

    pub  fn get_name_with_index(&self, index: i32) -> String {
        format!("{}[{}]", &self.id.get_base_name(), index)
    }
}

impl VerilogVirtualBaseID for VerilogSliceID {
    fn is_bus_slice_id(&self) -> bool {
        true
    }
    // &str?
    fn get_name(&self) -> String {
        format!("{}[{}:{}]", &self.id.get_base_name(), self.range_from, self.range_to)
    }
}


pub trait VerilogVirtualBaseNetExpr : Debug{
    fn is_id_expr(&self) -> bool {
        false
    }
    fn is_concat_expr(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        false
    }
    fn get_verilog_id(&self) ->&Box<dyn VerilogVirtualBaseID> {
        panic!("This is unknown value.");
    }
    fn get_verilog_id_concat(&self) ->&Vec<Box<dyn VerilogVirtualBaseID>> {
        panic!("This is unknown value.");
    }
    fn as_any(&self) -> &dyn std::any::Any;
}

// impl fmt::Debug for dyn VerilogVirtualBaseNetExpr {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "VerilogVirtualBaseNetExpr {{ verilog_id: {:?} }}", self.get_verilog_id())
//     }
// }

#[derive(Debug)]
pub struct VerilogNetExpr {
    line_no: usize,
}

impl VerilogNetExpr {
    fn new(line_no: usize) -> VerilogNetExpr {
        VerilogNetExpr{ line_no: line_no }
    }

    pub fn get_line_no(&self) -> usize {
        self.line_no
    }
}

#[derive(Debug)]
pub struct VerilogNetIDExpr {
    net_expr: VerilogNetExpr,
    verilog_id: Box<dyn VerilogVirtualBaseID>,
}

impl VerilogNetIDExpr {
    pub fn new(
        line_no: usize,
        verilog_id: Box<dyn VerilogVirtualBaseID>
    ) -> VerilogNetIDExpr {
        VerilogNetIDExpr {
            net_expr: VerilogNetExpr::new(line_no),
            verilog_id:verilog_id,
        }
    }

    pub fn get_net_expr(&self) -> &VerilogNetExpr {
        &self.net_expr
    }
    pub fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.verilog_id
    }
}

impl VerilogVirtualBaseNetExpr for VerilogNetIDExpr {
    fn is_id_expr(&self) -> bool {
        true
    }
    fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.verilog_id
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
/// such as { 2'b00, _0_ }
pub struct VerilogNetConcatExpr {
    net_expr: VerilogNetExpr,
    verilog_id_concat: Vec<Box<dyn VerilogVirtualBaseID>>,
}

impl VerilogNetConcatExpr {
    pub fn new(
        line_no: usize,
        verilog_id_concat: Vec<Box<dyn VerilogVirtualBaseID>>
    ) -> VerilogNetConcatExpr {
        VerilogNetConcatExpr {
            net_expr: VerilogNetExpr::new(line_no),
            verilog_id_concat: verilog_id_concat,
        }
    }

    pub fn get_net_expr(&self) -> &VerilogNetExpr {
        &self.net_expr
    }
    //get_verilog_id_concat
    pub fn get_verilog_id_concat(&self) -> &Vec<Box<dyn VerilogVirtualBaseID>> {
        &self.verilog_id_concat
    }
}

impl VerilogVirtualBaseNetExpr for VerilogNetConcatExpr {
    fn is_concat_expr(&self) -> bool {
        true
    }
    // get_verilog_id_concat
    fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.verilog_id_concat.first().unwrap()
    }

    fn get_verilog_id_concat(&self) ->&Vec<Box<dyn VerilogVirtualBaseID>> {
        &self.verilog_id_concat
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
/// 1'b0 or 1'b1.
pub struct VerilogConstantExpr {
    net_expr: VerilogNetExpr,
    verilog_id: Box<dyn VerilogVirtualBaseID>,
}

impl VerilogConstantExpr {
    pub fn new(
        line_no: usize,
        verilog_id: Box<dyn VerilogVirtualBaseID>
    ) -> VerilogConstantExpr {
        VerilogConstantExpr {
            net_expr: VerilogNetExpr::new(line_no),
            verilog_id:verilog_id,
        }
    }

    pub fn get_net_expr(&self) -> &VerilogNetExpr {
        &self.net_expr
    }
    pub fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.verilog_id
    }
    // get_verilog_id should realize in impl struct or impl trait? (according to simpleAttr, supposed to be struct)
}

impl VerilogVirtualBaseNetExpr for VerilogConstantExpr {
    fn is_constant(&self) -> bool {
        true
    }
    fn get_verilog_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.verilog_id
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The port connection such as .port_id(net_id).
#[derive(Debug)]
pub struct VerilogPortRefPortConnect {
    port_id: Box<dyn VerilogVirtualBaseID>,
    net_expr: Option<Box<dyn VerilogVirtualBaseNetExpr>>,
}

impl VerilogPortRefPortConnect {
    pub fn new(
        port_id: Box<dyn VerilogVirtualBaseID>,
        net_expr: Option<Box<dyn VerilogVirtualBaseNetExpr>>,
    ) -> VerilogPortRefPortConnect {
        VerilogPortRefPortConnect {
            port_id: port_id,
            net_expr: net_expr,
        }
    }

    pub fn get_port_id(&self) -> &Box<dyn VerilogVirtualBaseID> {
        &self.port_id
    }

    pub fn get_net_expr(&self) -> &Option<Box<dyn VerilogVirtualBaseNetExpr>> {
        &self.net_expr
    }
}

pub trait VerilogVirtualBaseStmt: Debug {
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

// impl fmt::Debug for dyn VerilogVirtualBaseStmt {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "VerilogVirtualBaseStmt {{ line_no: {} }}", self.get_line_no())
//     }
// }

/// The base class for verilog stmt,include module dcl, module instance, module assign.
/// maybe dont need the base class***************************************
#[derive(Debug)]
#[derive(Clone)]
pub struct VerilogStmt {
    line_no: usize,
}

impl VerilogStmt {
    fn new(line_no: usize) -> VerilogStmt {
        VerilogStmt{ line_no: line_no }
    }

    pub fn get_line_no(&self) -> usize {
        self.line_no
    }
}

#[derive(Debug)]
pub struct VerilogInst {
    stmt: VerilogStmt,  //stmt denote line_no 
    inst_name: String,
    cell_name: String,
    port_connections: Vec<Box<VerilogPortRefPortConnect>>,  
}

impl VerilogInst {
    pub fn new(
        line_no: usize,  
        inst_name: &str,
        cell_name: &str,
        port_connections: Vec<Box<VerilogPortRefPortConnect>>  
    ) -> VerilogInst {
        VerilogInst {
            stmt: VerilogStmt::new(line_no),  
            inst_name: inst_name.to_string(),          // add for other to do 
            cell_name: cell_name.to_string(),
            port_connections: port_connections,  
        }
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
/// #define FOREACH_VERILOG_PORT_CONNECT(inst, port_connect) for (auto& port_connect : inst->get_port_connections())
#[derive(Debug)]
pub struct VerilogAssign {
    stmt: VerilogStmt,  //stmt denote line_no 
    left_net_expr: Box<dyn VerilogVirtualBaseNetExpr>,
    right_net_expr: Box<dyn VerilogVirtualBaseNetExpr>, 
}

impl VerilogAssign {
    pub fn new(
        line_no: usize,  
        left_net_expr: Box<dyn VerilogVirtualBaseNetExpr>,
        right_net_expr: Box<dyn VerilogVirtualBaseNetExpr>,
    ) -> VerilogAssign {
        VerilogAssign {
            stmt: VerilogStmt::new(line_no),  
            left_net_expr: left_net_expr,
            right_net_expr: right_net_expr,
        }
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

#[repr(C)]
/// The wire or port declaration.
#[derive(Debug)]
#[derive(Clone, Copy)]
pub enum DclType {
    KInput = 0,
    KInout = 1,
    KOutput = 2,
    KSupply0 =3,
    KSupply1 =4,
    KTri = 5,
    KWand = 6,
    KWire = 7,
    KWor = 8,
}

#[repr(C)]
#[derive(Clone)]
#[derive(Debug)]
pub struct VerilogDcl {
    stmt: VerilogStmt,  //stmt denote line_no 
    dcl_type: DclType,
    dcl_name: String,
    range: Option<(i32, i32)>,
}

impl VerilogDcl {
    pub fn new(
        line_no: usize,  
        dcl_type: DclType,
        dcl_name: &str,
        range: Option<(i32, i32)>
    ) -> VerilogDcl {
        VerilogDcl {
            stmt: VerilogStmt::new(line_no),  
            dcl_type: dcl_type,
            dcl_name: dcl_name.to_string(),
            range: range, 
        }
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
    stmt: VerilogStmt,  //stmt denote line_no 
    verilog_dcls: Vec<Box<VerilogDcl>>,
}

impl VerilogDcls {
    pub fn new(
        line_no: usize,  
        verilog_dcls: Vec<Box<VerilogDcl>>
    ) -> VerilogDcls {
        VerilogDcls {
            stmt: VerilogStmt::new(line_no),  
            verilog_dcls: verilog_dcls,
        }
    }

    pub fn get_stmt(&self) -> &VerilogStmt {
        &self.stmt
    }
    pub fn get_verilog_dcls(&self) -> &Vec<Box<VerilogDcl>> {
        &self.verilog_dcls
    }
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

#[derive(Debug)]
pub struct VerilogModule {
    stmt: VerilogStmt,  //stmt denote line_no 
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
            port_list: port_list,
            module_stmts: module_stmts,
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





