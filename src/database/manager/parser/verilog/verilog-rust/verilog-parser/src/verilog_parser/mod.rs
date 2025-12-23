pub mod verilog_c_api;
pub mod verilog_data;

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;
use verilog_data::{
    VerilogID, VerilogNetIDExpr, VerilogVirtualBaseID, VerilogVirtualBaseNetExpr, VerilogVirtualBaseStmt,
};

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::rc::Rc;

#[derive(Parser)]
#[grammar = "verilog_parser/grammar/verilog.pest"]
pub struct VerilogParser;

fn process_module_id(pair: Pair<Rule>) -> Result<&str, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair_clone.as_rule() {
        Rule::module_id => {
            let module_name = pair_clone.as_str();
            Ok(module_name)
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Failed to parse module id".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_port_or_wire_id(
    pair: Pair<Rule>,
) -> Result<Box<dyn verilog_data::VerilogVirtualBaseID>, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair_clone.as_rule() {
        Rule::port_or_wire_id => {
            let id = pair_clone.as_str();
            let verilog_id = verilog_data::VerilogID::new(id);
            let verilog_virtual_base_id: Box<dyn verilog_data::VerilogVirtualBaseID> = Box::new(verilog_id);

            Ok(verilog_virtual_base_id)
        }
        _ => todo!(),
    }
}

fn process_port_or_wire_declaration(
    pair: Pair<Rule>,
) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair.as_rule() {
        Rule::input_declaration => {
            let dcl_type = verilog_data::DclType::KInput;

            process_inner_port_declaration(pair, dcl_type)
        }
        Rule::output_declaration => {
            let dcl_type = verilog_data::DclType::KOutput;

            process_inner_port_declaration(pair, dcl_type)
        }
        Rule::inout_declaration => {
            let dcl_type = verilog_data::DclType::KInout;

            process_inner_port_declaration(pair, dcl_type)
        }
        Rule::wire_declaration => {
            let dcl_type = verilog_data::DclType::KWire;

            process_inner_wire_declaration(pair, dcl_type)
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_inner_port_declaration(
    pair: Pair<Rule>,
    dcl_type: verilog_data::DclType,
) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>> {
    let line_no = pair.line_col().0;
    let pair_clone = pair.clone();
    let mut inner_pair = pair.into_inner();
    let port_list_or_bus_slice_pair = inner_pair.next();
    match port_list_or_bus_slice_pair.clone().unwrap().as_rule() {
        Rule::bus_slice => {
            let mut decimal_digits_pair = port_list_or_bus_slice_pair.unwrap().into_inner();

            let range_from = decimal_digits_pair.next().unwrap().as_str().parse::<i32>().unwrap();
            let range_to = decimal_digits_pair.next().unwrap().as_str().parse::<i32>().unwrap();
            let range = Some((range_from, range_to));
            let mut verilog_dcl_vec: Vec<Box<verilog_data::VerilogDcl>> = Vec::new();
            let port_list_pair = inner_pair.next();
            for port_pair in port_list_pair.unwrap().into_inner() {
                let port_line_no = port_pair.line_col().0;
                let dcl_name = port_pair.as_str().to_string();
                let verilog_dcl = verilog_data::VerilogDcl::new(port_line_no, dcl_type, &dcl_name, range);
                verilog_dcl_vec.push(Box::new(verilog_dcl.clone()));
            }
            let verilog_dcls = verilog_data::VerilogDcls::new(line_no, verilog_dcl_vec);
            Ok(Box::new(verilog_dcls) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
        }
        Rule::port_list => {
            let range = None;
            let mut verilog_dcl_vec: Vec<Box<verilog_data::VerilogDcl>> = Vec::new();
            let port_list_pair = port_list_or_bus_slice_pair;
            for port_pair in port_list_pair.unwrap().into_inner() {
                let port_line_no = port_pair.line_col().0;
                let dcl_name = port_pair.as_str().to_string();
                let verilog_dcl = verilog_data::VerilogDcl::new(port_line_no, dcl_type, &dcl_name, range);
                verilog_dcl_vec.push(Box::new(verilog_dcl.clone()));
            }
            let verilog_dcls = verilog_data::VerilogDcls::new(line_no, verilog_dcl_vec);
            Ok(Box::new(verilog_dcls) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_inner_wire_declaration(
    pair: Pair<Rule>,
    dcl_type: verilog_data::DclType,
) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    let line_no = pair.line_col().0;
    let mut inner_pair = pair.into_inner();
    let wire_list_or_bus_slice_pair = inner_pair.next();
    match wire_list_or_bus_slice_pair.clone().unwrap().as_rule() {
        Rule::bus_slice => {
            let mut decimal_digits_pair = wire_list_or_bus_slice_pair.unwrap().into_inner();
            let range_from = decimal_digits_pair.next().unwrap().as_str().parse::<i32>().unwrap();
            let range_to = decimal_digits_pair.next().unwrap().as_str().parse::<i32>().unwrap();
            let range = Some((range_from, range_to));
            let mut verilog_dcl_vec: Vec<Box<verilog_data::VerilogDcl>> = Vec::new();
            let wire_list_pair = inner_pair.next();
            for wire_pair in wire_list_pair.unwrap().into_inner() {
                let wire_line_no = wire_pair.line_col().0;
                let dcl_name = wire_pair.as_str().to_string();
                let verilog_dcl = verilog_data::VerilogDcl::new(wire_line_no, dcl_type, &dcl_name, range);
                verilog_dcl_vec.push(Box::new(verilog_dcl.clone()));
            }
            let verilog_dcls = verilog_data::VerilogDcls::new(line_no, verilog_dcl_vec);
            Ok(Box::new(verilog_dcls) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
        }
        Rule::wire_list => {
            let range = None;
            let mut verilog_dcl_vec: Vec<Box<verilog_data::VerilogDcl>> = Vec::new();
            let wire_list_pair = wire_list_or_bus_slice_pair;
            for wire_pair in wire_list_pair.unwrap().into_inner() {
                let wire_line_no = wire_pair.line_col().0;
                let dcl_name = wire_pair.as_str().to_string();
                let verilog_dcl = verilog_data::VerilogDcl::new(wire_line_no, dcl_type, &dcl_name, range);
                verilog_dcl_vec.push(Box::new(verilog_dcl.clone()));
            }
            let verilog_dcls = verilog_data::VerilogDcls::new(line_no, verilog_dcl_vec);
            Ok(Box::new(verilog_dcls) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_assign_or_inst_declaration(
    pair: Pair<Rule>,
) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair.as_rule() {
        Rule::assign_declaration => process_assign_declaration(pair),
        Rule::inst_declaration => process_inst_declaration(pair),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_assign_declaration(
    pair: Pair<Rule>,
) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>> {
    let line_no = pair.line_col().0;
    let pair_clone = pair.clone();
    match pair.as_rule() {
        Rule::assign_declaration => {
            let mut inner_pair = pair.into_inner();
            let left_net_str = inner_pair.next().unwrap().as_str();
            let left_net_base_id: Box<dyn verilog_data::VerilogVirtualBaseID> =
                build_verilog_virtual_base_id(left_net_str);
            // assign lhs=rhs;(only consider lhs/rhs is VerilogNetIDExpr)
            let left_net_id_expr = verilog_data::VerilogNetIDExpr::new(line_no, left_net_base_id);
            let left_net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> = Box::new(left_net_id_expr);

            if inner_pair.len() == 1 {
                let right_net_str = inner_pair.next().unwrap().as_str();
                let right_net_base_id: Box<dyn verilog_data::VerilogVirtualBaseID> =
                    build_verilog_virtual_base_id(right_net_str);
                // assign lhs=rhs;(only consider lhs/rhs is VerilogNetIDExpr)
                let right_net_id_expr = verilog_data::VerilogNetIDExpr::new(line_no, right_net_base_id);
                let right_net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> = Box::new(right_net_id_expr);

                let verilog_assign = verilog_data::VerilogAssign::new(line_no, left_net_expr, right_net_expr);
                Ok(Box::new(verilog_assign) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
            } else {
                let mut verilog_id_concat: Vec<Box<dyn verilog_data::VerilogVirtualBaseNetExpr>> = Vec::new();

                for inner_pair in inner_pair {
                    let right_net_str = inner_pair.as_str();
                    let right_net_base_id: Box<dyn verilog_data::VerilogVirtualBaseID> =
                        build_verilog_virtual_base_id(right_net_str);
                    // assign lhs=rhs;(only consider lhs/rhs is VerilogNetIDExpr)
                    let right_net_id_expr = verilog_data::VerilogNetIDExpr::new(line_no, right_net_base_id);
                    let right_net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> = Box::new(right_net_id_expr);
                    verilog_id_concat.push(right_net_expr);
                }

                let right_net_id_expr = verilog_data::VerilogNetConcatExpr::new(line_no, verilog_id_concat);

                let right_net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> = Box::new(right_net_id_expr);

                let verilog_assign = verilog_data::VerilogAssign::new(line_no, left_net_expr, right_net_expr);
                Ok(Box::new(verilog_assign) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
            }
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn extract_range(input: &str) -> Option<(&str, i32, i32)> {
    if let Some(open_bracket) = input.rfind('[') {
        if let Some(close_bracket) = input.rfind(']') {
            if let Some(colon) = input.rfind(':') {
                let name = &input[..open_bracket];
                let start = input[open_bracket + 1..colon].parse().ok()?;
                let end = input[colon + 1..close_bracket].parse().ok()?;
                return Some((name.trim_end(), start, end));
            }
        }
    }
    None
}

fn extract_single(input: &str) -> Option<(&str, i32)> {
    if let Some(open_bracket) = input.find('[') {
        if let Some(close_bracket) = input.find(']') {
            // return None when input likes "cpuregs[19][1]" or "\core_top_inst/ifu_inst/ICache_top_inst/icache_inst/cache_core_inst/cache_way_inst [0].cache_way_inst/_015_".
            if input[close_bracket + 1..].is_empty() {
                let name = &input[..open_bracket];
                let index = input[open_bracket + 1..close_bracket].parse().ok()?;
                return Some((name.trim_end(), index));
            }
        }
    }
    None
}

fn extract_name(input: &str) -> Option<&str> {
    Some(input)
}

fn build_verilog_virtual_base_id(input: &str) -> Box<dyn verilog_data::VerilogVirtualBaseID> {
    let verilog_virtual_base_id: Box<dyn verilog_data::VerilogVirtualBaseID>;
    if let Some((name, range_from, range_to)) = extract_range(input) {
        let verilog_slice_id = verilog_data::VerilogSliceID::new(name, range_from, range_to);
        verilog_virtual_base_id = Box::new(verilog_slice_id);
    } else if let Some((name, index)) = extract_single(input) {
        let verilog_index_id = verilog_data::VerilogIndexID::new(name, index);
        verilog_virtual_base_id = Box::new(verilog_index_id);
    } else if let Some(name) = extract_name(input) {
        let verilog_id = verilog_data::VerilogID::new(name);
        verilog_virtual_base_id = Box::new(verilog_id);
    } else {
        let verilog_id = verilog_data::VerilogID::default();
        verilog_virtual_base_id = Box::new(verilog_id);
    }

    verilog_virtual_base_id
}

fn process_first_port_connection_single_connect(
    pair: Pair<Rule>,
) -> Result<Box<verilog_data::VerilogPortRefPortConnect>, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    let mut inner_pairs = pair.into_inner();
    let length = inner_pairs.clone().count();
    match length {
        2 => {
            let port = inner_pairs.next().unwrap().as_str();
            let port_id = build_verilog_virtual_base_id(port);
            let net_connect_pair = inner_pairs.next().unwrap();
            match net_connect_pair.as_rule() {
                Rule::scalar_constant => {
                    let net_connect_line_no = net_connect_pair.line_col().0;
                    let net_connect = net_connect_pair.as_str();
                    let verilog_id = verilog_data::VerilogID::new(net_connect);
                    let verilog_virtual_base_id: Box<dyn verilog_data::VerilogVirtualBaseID> = Box::new(verilog_id);
                    let verilog_const_net_expr =
                        verilog_data::VerilogConstantExpr::new(net_connect_line_no, verilog_virtual_base_id);
                    let net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> = Box::new(verilog_const_net_expr);
                    let port_ref = Box::new(verilog_data::VerilogPortRefPortConnect::new(port_id, Some(net_expr)));
                    Ok(port_ref)
                }
                Rule::port_or_wire_id => {
                    let net_connect_line_no = net_connect_pair.line_col().0;
                    let net_connect = net_connect_pair.as_str();
                    let verilog_virtual_base_id: Box<dyn verilog_data::VerilogVirtualBaseID> =
                        build_verilog_virtual_base_id(net_connect);
                    let verilog_net_id_expr =
                        verilog_data::VerilogNetIDExpr::new(net_connect_line_no, verilog_virtual_base_id);
                    let net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> = Box::new(verilog_net_id_expr);
                    let port_ref = Box::new(verilog_data::VerilogPortRefPortConnect::new(port_id, Some(net_expr)));
                    Ok(port_ref)
                }
                _ => Err(pest::error::Error::new_from_span(
                    pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
                    pair_clone.as_span(),
                )),
            }
        }
        1 => {
            let port = inner_pairs.next().unwrap().as_str();
            let port_id = build_verilog_virtual_base_id(port);
            let net_expr: Option<Box<dyn verilog_data::VerilogVirtualBaseNetExpr>> = None;
            let port_ref = Box::new(verilog_data::VerilogPortRefPortConnect::new(port_id, net_expr));
            Ok(port_ref)
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn extract_bitwidth_value(input: &str) -> Option<(u32, String)> {
    // Split the input string by the character `'` and collect into a tuple(2'b00->(2,b'00))
    input.split_once('\'').and_then(|(bw, val)| bw.parse::<u32>().ok().map(|bit_width| (bit_width, val.to_string())))
}

fn process_first_port_connection_multiple_connect(
    pair: Pair<Rule>,
) -> Result<Box<verilog_data::VerilogPortRefPortConnect>, pest::error::Error<Rule>> {
    let line_no = pair.line_col().0;
    let mut inner_pairs = pair.into_inner();
    let port = inner_pairs.next().unwrap().as_str();
    let port_id = build_verilog_virtual_base_id(port);
    let mut verilog_id_concat: Vec<Box<dyn verilog_data::VerilogVirtualBaseNetExpr>> = Vec::new();
    for inner_pair in inner_pairs {
        match inner_pair.as_rule() {
            Rule::scalar_constant => {
                let net_connect_line_no = inner_pair.line_col().0;
                let net_connect = inner_pair.as_str();
                let verilog_constant_id: verilog_data::VerilogConstantID;
                if let Some((bit_width, value)) = extract_bitwidth_value(net_connect) {
                    verilog_constant_id = verilog_data::VerilogConstantID::new(bit_width, &value);
                } else {
                    panic!("Error: invalid format");
                }
                let verilog_virtual_base_id: Box<dyn verilog_data::VerilogVirtualBaseID> =
                    Box::new(verilog_constant_id);
                let verilog_net_constant_expr =
                    verilog_data::VerilogConstantExpr::new(net_connect_line_no, verilog_virtual_base_id);
                let verilog_virtual_base_net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> =
                    Box::new(verilog_net_constant_expr);
                verilog_id_concat.push(verilog_virtual_base_net_expr);
            }
            Rule::port_or_wire_id => {
                let net_connect_line_no = inner_pair.line_col().0;
                let net_connect = inner_pair.as_str();
                let verilog_virtual_base_id: Box<dyn verilog_data::VerilogVirtualBaseID> =
                    build_verilog_virtual_base_id(net_connect);
                let verilog_net_id_expr =
                    verilog_data::VerilogNetIDExpr::new(net_connect_line_no, verilog_virtual_base_id);
                let verilog_virtual_base_net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> =
                    Box::new(verilog_net_id_expr);
                verilog_id_concat.push(verilog_virtual_base_net_expr);
            }
            _ => unreachable!(),
        }
    }
    let verilog_net_concat_expr = verilog_data::VerilogNetConcatExpr::new(line_no, verilog_id_concat);
    let net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> = Box::new(verilog_net_concat_expr);
    let port_ref = Box::new(verilog_data::VerilogPortRefPortConnect::new(port_id, Some(net_expr)));
    Ok(port_ref)
}

fn process_port_block_connection(
    pair: Pair<Rule>,
) -> Result<Vec<Box<verilog_data::VerilogPortRefPortConnect>>, pest::error::Error<Rule>> {
    let mut port_connections: Vec<Box<verilog_data::VerilogPortRefPortConnect>> = Vec::new();
    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::first_port_connection_single_connect => {
                let port_connection = process_first_port_connection_single_connect(inner_pair);
                port_connections.push(port_connection.unwrap());
            }
            Rule::first_port_connection_multiple_connect => {
                let port_connection = process_first_port_connection_multiple_connect(inner_pair);
                port_connections.push(port_connection.unwrap());
            }
            // refactor
            _ => unreachable!(),
        }
    }
    Ok(port_connections)
}

fn process_inner_inst_declaration(
    pair: Pair<Rule>,
) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>> {
    let line_no = pair.line_col().0;
    let pair_clone = pair.clone();
    let mut inner_pair = pair.into_inner();
    let cell_id_pair = inner_pair.next();
    // println!("{:#?}", cell_id_pair);
    match cell_id_pair.clone().unwrap().as_rule() {
        Rule::cell_id => {
            let cell_name = cell_id_pair.unwrap().as_str();
            let inst_name = inner_pair.next().unwrap().as_str();
            let port_connections = process_port_block_connection(inner_pair.next().unwrap());
            let port_connections_vec = port_connections.unwrap();

            let verilog_inst = verilog_data::VerilogInst::new(line_no, inst_name, cell_name, port_connections_vec);
            Ok(Box::new(verilog_inst) as Box<dyn verilog_data::VerilogVirtualBaseStmt>)
        }
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_inst_declaration(
    pair: Pair<Rule>,
) -> Result<Box<dyn verilog_data::VerilogVirtualBaseStmt>, pest::error::Error<Rule>> {
    let pair_clone = pair.clone();
    match pair.as_rule() {
        Rule::inst_declaration => process_inner_inst_declaration(pair),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair_clone.as_span(),
        )),
    }
}

fn process_dcl(
    dcl_stmt: &Box<verilog_data::VerilogDcl>,
    cur_module: &Rc<RefCell<verilog_data::VerilogModule>>,
    parent_module: &Rc<RefCell<verilog_data::VerilogModule>>,
    inst_stmt: &verilog_data::VerilogInst,
) {
    let dcl_name = dcl_stmt.get_dcl_name();
    let dcl_type = dcl_stmt.get_dcl_type();
    if let verilog_data::DclType::KWire = dcl_type {
        if !verilog_data::VerilogModule::is_port(&cur_module.borrow(), dcl_name) {
            let new_dcl_name = format!("{}/{}", inst_stmt.get_inst_name(), dcl_name);
            let mut new_dcl_stmt: verilog_data::VerilogDcl = (**dcl_stmt).clone();
            new_dcl_stmt.set_dcl_name(&new_dcl_name);
            let mut verilog_dcl_vec: Vec<Box<verilog_data::VerilogDcl>> = Vec::new();
            verilog_dcl_vec.push(Box::new(new_dcl_stmt));
            let verilog_dcls = verilog_data::VerilogDcls::new((**dcl_stmt).get_stmt().get_line_no(), verilog_dcl_vec);
            let new_dcls_stmt: Box<dyn verilog_data::VerilogVirtualBaseStmt> = Box::new(verilog_dcls);
            let mut parent_module_mut = parent_module.borrow_mut();
            parent_module_mut.add_stmt(new_dcls_stmt);
        }
    }
}

fn find_dcl_stmt_range(
    cur_module: &Rc<RefCell<verilog_data::VerilogModule>>,
    net_base_name: &str,
) -> Option<(i32, i32)> {
    let borrowed_module = cur_module.borrow();
    let mut range: Option<(i32, i32)> = None;
    let dcls_stmt_opt = borrowed_module.find_dcls_stmt(net_base_name);
    if dcls_stmt_opt.is_some() {
        let dcls_stmt = dcls_stmt_opt.unwrap();
        if dcls_stmt.is_verilog_dcls_stmt() {
            let verilog_dcls_stmt = dcls_stmt.as_any().downcast_ref::<verilog_data::VerilogDcls>().unwrap();
            for verilog_dcl in verilog_dcls_stmt.get_verilog_dcls() {
                if verilog_dcl.get_dcl_name().eq(net_base_name) {
                    range = *verilog_dcl.get_range();
                    break;
                }
            }
        }
    }

    range
}

fn process_port_connect(
    net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr>,
    cur_module: &Rc<RefCell<verilog_data::VerilogModule>>,
    parent_module: &Rc<RefCell<verilog_data::VerilogModule>>,
    inst_stmt: &verilog_data::VerilogInst,
) -> Option<Box<dyn verilog_data::VerilogVirtualBaseNetExpr>> {
    let net_expr_id = net_expr.get_verilog_id();
    let mut net_expr_id_clone = net_expr_id.clone();
    let net_base_name = net_expr_id.get_base_name();
    let mut range: Option<(i32, i32)> = None;

    // for may be port_name is split, we need judge port base name and full name
    if !verilog_data::VerilogModule::is_port(&cur_module.borrow(), net_base_name)
        && !verilog_data::VerilogModule::is_port(&cur_module.borrow(), net_expr_id.get_name())
    {
        // for common name, should check whether bus, get range first.
        if !net_base_name.contains("/") && !net_expr_id.is_bus_index_id() && !net_expr_id.is_bus_slice_id() {
            range = find_dcl_stmt_range(cur_module, net_base_name);
        }

        // not port, change net name to inst name / net_name.
        if range.is_none() {
            let new_net_base_name = format!("{}/{}", inst_stmt.get_inst_name(), net_base_name);

            net_expr_id_clone.set_base_name(&new_net_base_name);
            let new_expr = Box::new(verilog_data::VerilogNetIDExpr::new(net_expr.get_line_no(), net_expr_id_clone));
            let dyn_new_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> = Box::new(*new_expr);
            Some(dyn_new_expr)
        } else {
            let mut verilog_id_concat: Vec<Box<dyn verilog_data::VerilogVirtualBaseNetExpr>> = Vec::new();
            let is_first_greater = range.unwrap().0 > range.unwrap().1;
            let mut index = range.unwrap().0;
            while is_first_greater && index >= range.unwrap().1 || !is_first_greater && index <= range.unwrap().1 {
                let new_net_name = format!("{}/{}", inst_stmt.get_inst_name(), net_base_name);
                let index_id = verilog_data::VerilogIndexID::new(&new_net_name, index);
                let dyn_index_id = Box::new(index_id);
                let new_index_net_id =
                    Box::new(verilog_data::VerilogNetIDExpr::new(net_expr.get_line_no(), dyn_index_id));
                let verilog_virtual_base_net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> =
                    Box::new(*new_index_net_id);
                verilog_id_concat.push(verilog_virtual_base_net_expr);
                if is_first_greater {
                    index -= 1;
                } else {
                    index += 1;
                }
            }
            let new_concat_net_id = verilog_data::VerilogNetConcatExpr::new(net_expr.get_line_no(), verilog_id_concat);
            let dyn_new_concat_net_id = Box::new(new_concat_net_id);
            Some(dyn_new_concat_net_id)
        }
    } else {
        // is port, check the port whether port or port bus, then get
        // the port or port bus connect parent net.
        let port_name = net_expr_id.get_name();
        // println!("port name {}", port_name);
        range = find_dcl_stmt_range(cur_module, net_base_name);
        if range.is_none() {
            // for may be port_name is split.
            range = find_dcl_stmt_range(cur_module, port_name);
        }

        // get port connected parent module net.***************

        // println!("port connect net {}", port_connect_net.clone().unwrap().get_verilog_id().get_name());
        verilog_data::VerilogInst::get_port_connect_net(inst_stmt, cur_module, parent_module, net_expr_id_clone, range)
    }
}

fn process_concat_net_expr(
    one_net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr>,
    cur_module: &Rc<RefCell<verilog_data::VerilogModule>>,
    parent_module: &Rc<RefCell<verilog_data::VerilogModule>>,
    inst_stmt: &verilog_data::VerilogInst,
) -> Box<dyn verilog_data::VerilogVirtualBaseNetExpr> {
    let mut new_one_net_expr = one_net_expr.clone();
    if one_net_expr.is_id_expr() {
        let port_connect_net = process_port_connect(one_net_expr, cur_module, parent_module, inst_stmt);
        if port_connect_net.is_some() {
            new_one_net_expr = port_connect_net.unwrap();
        }
    } else if one_net_expr.is_concat_expr() {
        let one_net_expr_concat = one_net_expr.as_any().downcast_ref::<verilog_data::VerilogNetConcatExpr>().unwrap();
        let mut new_net_expr_concat: Vec<Box<dyn verilog_data::VerilogVirtualBaseNetExpr>> = Vec::new();
        for net_expr in one_net_expr_concat.get_verilog_id_concat() {
            new_net_expr_concat.push(process_concat_net_expr(net_expr.clone(), cur_module, parent_module, inst_stmt));
        }
        new_one_net_expr = Box::new(verilog_data::VerilogNetConcatExpr::new(
            one_net_expr_concat.get_net_expr().get_line_no(),
            new_net_expr_concat,
        ));
    }

    // let new_one_net_expr_id = new_one_net_expr.get_verilog_id();
    // let new_net_base_name = new_one_net_expr_id.get_base_name();
    new_one_net_expr
}

fn flatten_the_module(
    cur_module: &Rc<RefCell<verilog_data::VerilogModule>>,
    parent_module: &Rc<RefCell<verilog_data::VerilogModule>>,
    inst_stmt: &verilog_data::VerilogInst,
    module_map: &HashMap<String, Rc<RefCell<verilog_data::VerilogModule>>>,
) {
    let mut have_sub_module;
    // flatten all sub module.
    loop {
        have_sub_module = false;
        let module_stmts: Vec<Box<dyn verilog_data::VerilogVirtualBaseStmt>>;
        {
            module_stmts = (*cur_module).clone().borrow().get_clone_module_stms();
        }
        for stmt in module_stmts {
            if stmt.is_module_inst_stmt() {
                let module_inst_stmt = (*stmt).as_any().downcast_ref::<verilog_data::VerilogInst>().unwrap();
                let sub_module = module_map.get(module_inst_stmt.get_cell_name());
                if let Some(sub_module) = sub_module {
                    have_sub_module = true;
                    println!(
                        "flatten module {} inst {}",
                        module_inst_stmt.get_cell_name(),
                        module_inst_stmt.get_inst_name()
                    );
                    // for debugging purpose
                    // if module_inst_stmt.get_cell_name() == "xxx" && module_inst_stmt.get_inst_name() == "xxx" {
                    //     println!("Debug");
                    // }
                    // if !(module_inst_stmt.get_cell_name() == "xxx" && module_inst_stmt.get_inst_name() == "xxx") {
                    //     flatten_the_module(sub_module, cur_module, module_inst_stmt, module_map);
                    // }

                    flatten_the_module(sub_module, cur_module, module_inst_stmt, module_map);

                    let mut cur_module_mut = cur_module.borrow_mut();

                    cur_module_mut.erase_stmt(&stmt);
                    break;
                }
            }
        }

        if !have_sub_module {
            break;
        }
    }

    let module_stmts: Vec<Box<dyn verilog_data::VerilogVirtualBaseStmt>>;
    {
        module_stmts = (*cur_module).clone().borrow().get_clone_module_stms();
    }
    for stmt in module_stmts {
        // for verilog dcl stmt, change the dcl name to inst name / dcl_name, then
        // add stmt to parent.
        if stmt.is_verilog_dcls_stmt() {
            let dcls_stmt = (*stmt).as_any().downcast_ref::<verilog_data::VerilogDcls>().unwrap();
            for dcl_stmt in dcls_stmt.get_verilog_dcls() {
                process_dcl(dcl_stmt, cur_module, parent_module, inst_stmt);
            }
        } else if stmt.is_module_inst_stmt() {
            // for verilog module instant stmt, first copy the module inst stmt,
            // then change the inst stmt connect net to net name / parent net
            // name(for port), next change the inst name to parent inst name /
            // current inst name.
            let module_inst_stmt = (*stmt).as_any().downcast_ref::<verilog_data::VerilogInst>().unwrap();
            // for debugging purpose
            // if module_inst_stmt.get_cell_name() == "sky130_fd_sc_hs__buf_1"
            //     && module_inst_stmt.get_inst_name().contains("_113_")
            // {
            //     println!("Debug");
            // }

            let mut new_module_inst_connection: Vec<Box<verilog_data::VerilogPortRefPortConnect>> = Vec::new();
            for port_connect in module_inst_stmt.get_port_connections() {
                let net_expr_option = port_connect.get_net_expr();
                // for debugging purpose
                // let port_id = port_connect.get_port_id().get_name();
                // if port_id == "A" {
                //     println!("Debug");
                // }

                if let Some(net_expr) = net_expr_option {
                    if net_expr.is_id_expr() {
                        let port_connect_net =
                            process_port_connect(net_expr.clone(), cur_module, parent_module, inst_stmt);
                        if port_connect_net.is_some() {
                            // is port connect net, set new net.
                            let mut port_connect_clone = port_connect.clone();
                            port_connect_clone.set_net_expr(port_connect_net);
                            new_module_inst_connection.push(port_connect_clone);
                        }
                    } else if net_expr.is_concat_expr() {
                        let concat_connect_net =
                            net_expr.as_any().downcast_ref::<verilog_data::VerilogNetConcatExpr>().unwrap();
                        let mut new_verilog_id_concat: Vec<Box<dyn verilog_data::VerilogVirtualBaseNetExpr>> =
                            Vec::new();
                        for one_net_expr in concat_connect_net.get_verilog_id_concat() {
                            let new_one_net_expr =
                                process_concat_net_expr(one_net_expr.clone(), cur_module, parent_module, inst_stmt);
                            new_verilog_id_concat.push(new_one_net_expr);
                        }
                        let new_concat_connect_net: verilog_data::VerilogNetConcatExpr =
                            verilog_data::VerilogNetConcatExpr::new(
                                concat_connect_net.get_net_expr().get_line_no(),
                                new_verilog_id_concat,
                            );

                        let mut port_connect_clone = port_connect.clone();
                        let dyn_new_concat_connect_net = Box::new(new_concat_connect_net);
                        port_connect_clone.set_net_expr(Some(dyn_new_concat_connect_net));
                        new_module_inst_connection.push(port_connect_clone);
                    }
                } else {
                    let port_connect_clone = port_connect.clone();
                    new_module_inst_connection.push(port_connect_clone);
                }
            }
            let the_stmt_inst_name = module_inst_stmt.get_inst_name();
            let new_inst_name = format!("{}/{}", inst_stmt.get_inst_name(), the_stmt_inst_name);
            let mut new_module_inst_stmt: verilog_data::VerilogInst = verilog_data::VerilogInst::new(
                module_inst_stmt.get_line_no(),
                module_inst_stmt.get_inst_name(),
                module_inst_stmt.get_cell_name(),
                new_module_inst_connection,
            );

            new_module_inst_stmt.set_inst_name(&new_inst_name);
            let new_dyn_module_inst_stmt: Box<dyn verilog_data::VerilogVirtualBaseStmt> =
                Box::new(new_module_inst_stmt);
            verilog_data::VerilogModule::add_stmt(&mut parent_module.borrow_mut(), new_dyn_module_inst_stmt);
        } else if stmt.is_module_assign_stmt() {
            // for assign stmt, we need change the assignment both side net expr to net name / parent net,
            // or the port to connected net name / parent net.
            let module_assign_stmt = (*stmt).as_any().downcast_ref::<verilog_data::VerilogAssign>().unwrap();
            let left_net_expr = module_assign_stmt.get_left_net_expr();
            let right_net_expr = module_assign_stmt.get_right_net_expr();

            let get_connect_net = |cur_module, parent_module, net_expr_id_clone, net_base_name, port_name| {
                // println!("port name {}", port_name);
                let mut range = find_dcl_stmt_range(cur_module, net_base_name);
                if range.is_none() {
                    // for may be port_name is split.
                    range = find_dcl_stmt_range(cur_module, port_name);
                }

                // get port connected parent module net.***************
                verilog_data::VerilogInst::get_port_connect_net(
                    inst_stmt,
                    cur_module,
                    parent_module,
                    net_expr_id_clone,
                    range,
                )
            };

            // for debug
            // println!("curr module {}", cur_module.clone().borrow().get_module_name());
            // println!("left base name {} port name {}", left_net_base_name, left_port_name);
            // println!("right base name {} port name {}", right_net_base_name, right_port_name);
            // println!("assign {} = {}", left_port_name, right_port_name);

            let get_or_create_port_connect_net =
                |port_connect_net_opt: Option<Box<dyn VerilogVirtualBaseNetExpr>>,
                 net_expr: &Box<dyn VerilogVirtualBaseNetExpr>| {
                    if port_connect_net_opt.is_none() {
                        // modify net name to inst name/net name
                        let port_connect_net = net_expr.clone();

                        let port_name = port_connect_net.get_verilog_id().get_name();
                        let new_port_name = format!("{}/{}", inst_stmt.get_inst_name(), port_name);

                        let new_port_connect_port_id = VerilogID::new(&new_port_name);
                        let dyn_new_port_connect_port_id: Box<dyn VerilogVirtualBaseID> =
                            Box::new(new_port_connect_port_id);

                        let new_left_port_connect_net =
                            VerilogNetIDExpr::new(port_connect_net.get_line_no(), dyn_new_port_connect_port_id);
                        Box::new(new_left_port_connect_net)
                    } else {
                        port_connect_net_opt.unwrap()
                    }
                };

            // get assign left parent connect net.
            let left_net_expr_id = left_net_expr.get_verilog_id();
            let left_port_name = left_net_expr_id.get_name();
            let left_net_base_name = left_net_expr_id.get_base_name();

            let left_port_connect_net_opt = get_connect_net(
                cur_module,
                parent_module,
                left_net_expr_id.clone(),
                left_net_base_name,
                left_port_name,
            );

            let left_port_connect_net = get_or_create_port_connect_net(left_port_connect_net_opt, left_net_expr);

            // get assign right parent connect net.

            if right_net_expr.is_id_expr() {
                let right_net_expr_id = right_net_expr.get_verilog_id();
                let right_port_name = right_net_expr_id.get_name();
                let right_net_base_name = right_net_expr_id.get_base_name();

                let right_port_connect_net_opt = get_connect_net(
                    cur_module,
                    parent_module,
                    right_net_expr_id.clone(),
                    right_net_base_name,
                    right_port_name,
                );

                let right_port_connect_net = get_or_create_port_connect_net(right_port_connect_net_opt, right_net_expr);

                // for debug
                // let new_left_port_name = left_port_connect_net.get_verilog_id().get_name();
                // let new_right_port_name = right_port_connect_net.get_verilog_id().get_name();
                // println!("new assign {} = {}", new_left_port_name, new_right_port_name);

                let new_assignment_stmt: verilog_data::VerilogAssign = verilog_data::VerilogAssign::new(
                    module_assign_stmt.get_line_no(),
                    left_port_connect_net,
                    right_port_connect_net,
                );

                let new_dyn_assign_stmt: Box<dyn verilog_data::VerilogVirtualBaseStmt> = Box::new(new_assignment_stmt);

                verilog_data::VerilogModule::add_stmt(&mut parent_module.borrow_mut(), new_dyn_assign_stmt);
            } else {
                if !right_net_expr.is_concat_expr() {
                    panic!("Error: right net expr is not id expr or concat expr, please check the verilog file.");
                }

                // should be concat expr, so we need to process each net expr in the concat expr.
                let right_net_expr_concat =
                    right_net_expr.as_any().downcast_ref::<verilog_data::VerilogNetConcatExpr>().unwrap();

                let mut new_right_net_expr_concat: Vec<Box<dyn verilog_data::VerilogVirtualBaseNetExpr>> = Vec::new();
                right_net_expr_concat.get_verilog_id_concat().iter().for_each(|net_expr| {
                    let right_net_expr_id = net_expr.get_verilog_id();
                    let right_port_name = right_net_expr_id.get_name();
                    let right_net_base_name = right_net_expr_id.get_base_name();
                    let right_port_connect_net_opt = get_connect_net(
                        cur_module,
                        parent_module,
                        right_net_expr_id.clone(),
                        right_net_base_name,
                        right_port_name,
                    );
                    let right_port_connect_net = get_or_create_port_connect_net(right_port_connect_net_opt, net_expr);
                    new_right_net_expr_concat.push(right_port_connect_net);
                });

                let new_right_net_expr = verilog_data::VerilogNetConcatExpr::new(
                    module_assign_stmt.get_line_no(),
                    new_right_net_expr_concat,
                );
                let new_dyn_right_net_expr: Box<dyn verilog_data::VerilogVirtualBaseNetExpr> =
                    Box::new(new_right_net_expr);

                let new_assignment_stmt: verilog_data::VerilogAssign = verilog_data::VerilogAssign::new(
                    module_assign_stmt.get_line_no(),
                    left_port_connect_net,
                    new_dyn_right_net_expr,
                );
                let new_dyn_assign_stmt: Box<dyn verilog_data::VerilogVirtualBaseStmt> = Box::new(new_assignment_stmt);
                verilog_data::VerilogModule::add_stmt(&mut parent_module.borrow_mut(), new_dyn_assign_stmt);
            }
        }
    }
}

pub fn parse_verilog_file(verilog_file_path: &str) -> verilog_data::VerilogFile {
    // Generate verilog.pest parser
    let input_str =
        std::fs::read_to_string(verilog_file_path).unwrap_or_else(|_| panic!("Can't read file: {}", verilog_file_path));
    let parse_result = VerilogParser::parse(Rule::verilog_file, input_str.as_str());

    let mut verilog_file = verilog_data::VerilogFile::new();

    match parse_result {
        Ok(pairs) => {
            // pairs:module_declaration+
            for pair in pairs {
                let line_no = pair.line_col().0;
                if pair.as_rule() == Rule::EOI {
                    continue;
                }
                let inner_pairs = pair.into_inner();
                let mut module_name = " ";
                let mut port_list: Vec<Box<dyn verilog_data::VerilogVirtualBaseID>> = Vec::new();
                let mut module_stmts: Vec<Box<dyn verilog_data::VerilogVirtualBaseStmt>> = Vec::new();
                for inner_pair in inner_pairs {
                    match inner_pair.as_rule() {
                        // Just read not deal with Rule::yosys_hierarchy_declaration.
                        Rule::yosys_hierarchy_declaration => {}
                        Rule::module_id => {
                            module_name = process_module_id(inner_pair).unwrap();
                        }
                        Rule::port_list => {
                            for inner_inner_pair in inner_pair.into_inner() {
                                let port_id = process_port_or_wire_id(inner_inner_pair).unwrap();
                                port_list.push(port_id);
                            }
                        }
                        Rule::port_or_wire_block_declaration => {
                            for inner_inner_pair in inner_pair.into_inner() {
                                let verilog_dcls = process_port_or_wire_declaration(inner_inner_pair).unwrap();
                                module_stmts.push(verilog_dcls);
                            }
                        }
                        Rule::assign_or_inst_block_declaration => {
                            for inner_inner_pair in inner_pair.into_inner() {
                                let verilog_assign_or_inst =
                                    process_assign_or_inst_declaration(inner_inner_pair).unwrap();
                                module_stmts.push(verilog_assign_or_inst);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                let verilog_module = Rc::new(RefCell::new(verilog_data::VerilogModule::new(
                    line_no,
                    module_name,
                    port_list,
                    module_stmts,
                )));
                verilog_file.add_module(verilog_module);
            }
        }
        Err(err) => {
            // Handle parsing error
            panic!("Fatal: {}", err);
        }
    }

    verilog_file
}

#[no_mangle]
pub extern "C" fn rust_parse_verilog(verilog_path: *const c_char) -> *mut c_void {
    let c_str_verilog_path = unsafe { std::ffi::CStr::from_ptr(verilog_path) };
    let r_str_verilog_path = c_str_verilog_path.to_string_lossy().into_owned();
    println!("r str {}", r_str_verilog_path);

    let verilog_file = parse_verilog_file(&r_str_verilog_path);

    let verilog_file_pointer = Box::new(verilog_file);

    let raw_pointer = Box::into_raw(verilog_file_pointer);
    raw_pointer as *mut c_void
}

pub fn flatten_module(verilog_file: &mut verilog_data::VerilogFile, top_module_name: &str) {
    verilog_file.set_top_module_name(top_module_name);
    let module_map = verilog_file.get_module_map();
    if module_map.len() > 1 {
        println!("flatten module {} start", top_module_name);
        let the_module =
            module_map.get(top_module_name).unwrap_or_else(|| panic!("can't find top module {}", top_module_name));
        let mut have_sub_module;
        loop {
            have_sub_module = false;

            let module_stmts: Vec<Box<dyn verilog_data::VerilogVirtualBaseStmt>>;
            {
                module_stmts = (*the_module).clone().borrow().get_clone_module_stms();
            }

            // let the_module_ref = the_module.borrow();
            for stmt in module_stmts {
                if stmt.is_module_inst_stmt() {
                    let module_inst_stmt = (*stmt).as_any().downcast_ref::<verilog_data::VerilogInst>().unwrap();
                    let sub_module = module_map.get(module_inst_stmt.get_cell_name());
                    if let Some(sub_module) = sub_module {
                        have_sub_module = true;
                        println!(
                            "flatten module {} inst {}",
                            module_inst_stmt.get_cell_name(),
                            module_inst_stmt.get_inst_name()
                        );

                        flatten_the_module(sub_module, the_module, module_inst_stmt, module_map);
                        let mut the_module_mut = the_module.borrow_mut();
                        the_module_mut.erase_stmt(&stmt);
                        break;
                    }
                }
            }
            if !have_sub_module {
                break;
            }
        }

        println!("flatten module {} end", top_module_name);
    }
}

#[no_mangle]
pub extern "C" fn rust_flatten_module(c_verilog_file: *mut verilog_data::VerilogFile, top_module_name: *const c_char) {
    let c_str_top_module_name = unsafe { std::ffi::CStr::from_ptr(top_module_name) };
    let r_str_top_module_name = c_str_top_module_name.to_string_lossy().into_owned();

    let verilog_file = unsafe { &mut (*c_verilog_file) };

    flatten_module(verilog_file, &r_str_top_module_name);
}

// To do
#[no_mangle]
pub extern "C" fn rust_free_verilog_file(c_verilog_file: *mut verilog_data::VerilogFile) {
    let _: Box<verilog_data::VerilogFile> = unsafe { Box::from_raw(c_verilog_file) };
}

#[cfg(test)]
mod tests {

    use pest::iterators::Pair;
    use pest::iterators::Pairs;

    use super::*;

    fn process_pair(pair: Pair<Rule>) {
        // A pair is a combination of the rule which matched and a span of input
        println!("Rule:    {:?}", pair.as_rule());
        println!("Span:    {:?}", pair.as_span());
        println!("Text:    {}", pair.as_str());

        for inner_pair in pair.into_inner() {
            process_pair(inner_pair);
        }
    }

    fn print_parse_result(parse_result: Result<Pairs<Rule>, pest::error::Error<Rule>>) {
        let parse_result_clone = parse_result.clone();
        match parse_result {
            Ok(pairs) => {
                for pair in pairs {
                    // A pair is a combination of the rule which matched and a span of input
                    process_pair(pair);
                }
            }
            Err(err) => {
                // Handle parsing error
                println!("Error: {}", err);
            }
        }

        assert!(parse_result_clone.is_ok());
    }

    fn extract_range(input: &str) -> Option<(&str, i32, i32)> {
        if let Some(open_bracket) = input.find('[') {
            if let Some(close_bracket) = input.find(']') {
                if let Some(colon) = input.find(':') {
                    let name = &input[..open_bracket];
                    let start = input[open_bracket + 1..colon].parse().ok()?;
                    let end = input[colon + 1..close_bracket].parse().ok()?;
                    return Some((name.trim_end(), start, end));
                }
            }
        }
        None
    }

    fn extract_single(input: &str) -> Option<(&str, i32)> {
        if let Some(open_bracket) = input.find('[') {
            if let Some(close_bracket) = input.find(']') {
                // return None when input likes "cpuregs[19][1]" or "\core_top_inst/ifu_inst/ICache_top_inst/icache_inst/cache_core_inst/cache_way_inst [0].cache_way_inst/_015_".
                if input[close_bracket + 1..].is_empty() {
                    let name = &input[..open_bracket];
                    let index = input[open_bracket + 1..close_bracket].parse().ok()?;
                    return Some((name.trim_end(), index));
                }
            }
        }
        None
    }

    fn extract_name(input: &str) -> Option<&str> {
        Some(input)
    }

    #[test]
    fn test_parse_comment() {
        let input_str = "/*test
        test
        */";
        let parse_result = VerilogParser::parse(Rule::COMMENT, input_str);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_yosys_hierarchy_declaration() {
        let input_str = "(* top =  1  *)";
        let parse_result = VerilogParser::parse(Rule::yosys_hierarchy_declaration, input_str);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_port_or_wire_id() {
        let _input_str = "q\n        ";
        let input_str = "q        ";
        let parse_result = VerilogParser::parse(Rule::port_or_wire_id, input_str);
        println!("{:#?}", parse_result);
        // print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_port_list() {
        let input_str = "in1, in2, clk1, clk2, clk3, out
        \n";
        let parse_result = VerilogParser::parse(Rule::port_list, input_str);
        println!("{:#?}", parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_port_or_wire_id1() {
        let _input_str = "clk ";
        let input_str = "\\in_$002 [0]"; //(wire)
        let _input_str = "_fd_sc_hs__nor2_1 _17_"; //(cell inst)
        let parse_result = VerilogParser::parse(Rule::port_or_wire_id, input_str);
        println!("{:#?}", parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_inst_or_cell_id() {
        let input_str = "i_cache_subsystem/i_nbdcache/sram_block[7].tag_sram/macro_mem[2].i_ram";
        let parse_result = VerilogParser::parse(Rule::inst_id, input_str);
        println!("{:#?}", parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_input_declaration() {
        let input_str = r#"input chiplink_rx_clk_pad;"#;
        let parse_result = VerilogParser::parse(Rule::input_declaration, input_str);

        println!("{:#?}", parse_result);
    }

    #[test]
    fn test_parse_input_declaration1() {
        let input_str = "input [1:0] din;";
        let parse_result = VerilogParser::parse(Rule::input_declaration, input_str);
        println!("{:#?}", parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_port_block_declaration() {
        let input_str = r#"output [1:0] a_mux_sel;
        output a_reg_en;
        output b_mux_sel;
        output b_reg_en;
        input clk;
      "#;
        let parse_result = VerilogParser::parse(Rule::port_or_wire_block_declaration, input_str);
        println!("{:#?}", parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_wire_declaration() {
        let input_str = "wire \\vga_b[0] ;";
        let _input_str = "wire ps2_dat;";
        let parse_result = VerilogParser::parse(Rule::wire_declaration, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_assign_port_or_wire_id() {
        let _input_str = "DATA_9_31";
        let input_str = "WX1010";
        let parse_result = VerilogParser::parse(Rule::assign_port_or_wire_id, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_assign_declaration() {
        let _input_str = "assign _01_ = b_pad;";
        let _input_str = "assign DATA_9_31 = WX1010;";
        let input_str = "assign n10 = 1'b0;";
        let parse_result = VerilogParser::parse(Rule::assign_declaration, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_wire_block_declaration() {
        let input_str = r#"wire \core_dip[2] ;
        wire \core_dip[1] ;
        wire \core_dip[0] ;
        wire \u0_rcg/u0_pll_bp ;
        wire \u0_rcg/u0_pll_fbdiv_5_ ;
        wire \u0_rcg/u0_pll_postdiv2_1_ ;
        wire \u0_rcg/u0_pll_clk ;"#;
        let parse_result = VerilogParser::parse(Rule::port_or_wire_block_declaration, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_first_port_connection_single_connect() {
        let input_str = r#".I(\u0_soc_top/u0_ysyx_210539/writeback_io_excep_en )"#;
        let parse_result = VerilogParser::parse(Rule::first_port_connection_single_connect, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_first_port_connection_multiple_connect() {
        let _input_str = r#".rid_nic400_axi4_ps2({ 1'b0,1'b0,rid_nic400_axi4_ps2_1_,1'b0 })"#;
        let input_str = r#".araddr_cpu_axi4_nic400({
                                n571, n563, n560, n564, n565, araddr_cpu_axi4_nic400[26:24], n561,
                                n566, n567, n562, n570, araddr_cpu_axi4_nic400[18], n572, n568,
                                araddr_cpu_axi4_nic400[15], n569, araddr_cpu_axi4_nic400[13], n573,
                                araddr_cpu_axi4_nic400[11:0]})"#;
        let parse_result = VerilogParser::parse(Rule::first_port_connection_multiple_connect, input_str);
        println!("{:#?}", parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_port_connection() {
        let input_str = r#",
        .Z(hold_net_52144)"#;
        let parse_result = VerilogParser::parse(Rule::port_connection, input_str);

        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_port_block_connection() {
        let input_str = r#"(.BYPASS(\u0_rcg/u0_pll_bp ),
        .REFDIV({ DRV_net_6,
                DRV_net_6,
                DRV_net_6,
                DRV_net_6,
                DRV_net_7,
                DRV_net_6 }),
        .POSTDIV2({ DRV_net_6,
                FE_PDN5026_u0_rcg_u0_pll_postdiv2_1,
                \u0_rcg/n34  }),
        .DSMPD(DRV_net_7),
        .FOUTPOSTDIVPD(DRV_net_6),
        .POSTDIV1({ \u0_rcg/n37 ,
                FE_PDN3515_pll_cfg_2,
                FE_PDN4015_u0_rcg_n35 }),
        .PD(DRV_net_6),
        .FOUTVCOPD(DRV_net_6),
        .FBDIV({ FE_PDN11668_DRV_net_6,
                FE_PDN11668_DRV_net_6,
                FE_PDN11668_DRV_net_6,
                FE_PDN1270_DRV_net_6,
                FE_PDN1270_DRV_net_6,
                DRV_net_7,
                FE_PDN4133_u0_rcg_u0_pll_fbdiv_5,
                \u0_rcg/n36 ,
                FE_PDN1270_DRV_net_6,
                FE_PDN1270_DRV_net_6,
                FE_PDN1270_DRV_net_6,
                FE_PDN1270_DRV_net_6 }),
        .FREF(FE_ECON20449_sys_clk_25m_buf),
        .FOUTVCO(),
        .CLKSSCG(),
        .LOCK(),
        .FOUTPOSTDIV(\u0_rcg/u0_pll_clk ));"#;

        let parse_result = VerilogParser::parse(Rule::port_block_connection, input_str);
        println!("{:#?}", parse_result);
        // print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_inst_declaration() {
        let _input_str = r#"PLLHPMLAINT \u0_rcg/u0_pll  (.BYPASS(\u0_rcg/u0_pll_bp ),
        .REFDIV({ DRV_net_6,
                DRV_net_6,
                DRV_net_6,
                DRV_net_6,
                DRV_net_7,
                DRV_net_6 }),
        .POSTDIV2({ DRV_net_6,
                FE_PDN5026_u0_rcg_u0_pll_postdiv2_1,
                \u0_rcg/n34  }),
        .DSMPD(DRV_net_7),
        .FOUTPOSTDIVPD(DRV_net_6),
        .POSTDIV1({ \u0_rcg/n37 ,
                FE_PDN3515_pll_cfg_2,
                FE_PDN4015_u0_rcg_n35 }),
        .PD(DRV_net_6),
        .FOUTVCOPD(DRV_net_6),
        .FBDIV({ FE_PDN11668_DRV_net_6,
                FE_PDN11668_DRV_net_6,
                FE_PDN11668_DRV_net_6,
                FE_PDN1270_DRV_net_6,
                FE_PDN1270_DRV_net_6,
                DRV_net_7,
                FE_PDN4133_u0_rcg_u0_pll_fbdiv_5,
                \u0_rcg/n36 ,
                FE_PDN1270_DRV_net_6,
                FE_PDN1270_DRV_net_6,
                FE_PDN1270_DRV_net_6,
                FE_PDN1270_DRV_net_6 }),
        .FREF(FE_ECON20449_sys_clk_25m_buf),
        .FOUTVCO(),
        .CLKSSCG(),
        .LOCK(),
        .FOUTPOSTDIV(\u0_rcg/u0_pll_clk ));"#;
        //inst_id =  @{ (char+ ~ " " ~ char+) | char+  }  adapt to:inst_id contain " "
        let input_str = r#"INV_X1 \core_top_inst/ifu_inst/ICache_inst/cache_core_inst/cache_way_inst [0].cache_way_inst/_118_ 
        ( .A(\core_top_inst/ifu_inst/ICache_inst/cache_core_inst/cache_way_inst [0].cache_way_inst/_015_ ), 
        .ZN(\core_top_inst/ifu_inst/ICache_inst/cache_core_inst/cache_way_inst [0].cache_way_inst/_046_ ) );"#;
        let parse_result = VerilogParser::parse(Rule::inst_declaration, input_str);
        println!("{:#?}", parse_result);
        // print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_module_id() {
        let input_str = "soc_top_0";
        let parse_result = VerilogParser::parse(Rule::module_id, input_str);
        println!("{:#?}", parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_inst_block_declaration() {
        let input_str = r#"DEL150MD1BWP40P140HVT hold_buf_52163 (.I(\u0_soc_top/u0_ysyx_210539/csrs/n3692 ),
        .Z(hold_net_52163));
        DEL150MD1BWP40P140HVT hold_buf_52164 (.I(\u0_soc_top/u0_ysyx_210539/icache/Ram_bw_3_io_wdata[123] ),
        .Z(hold_net_52164));"#;
        let parse_result = VerilogParser::parse(Rule::assign_or_inst_block_declaration, input_str);
        println!("{:#?}", parse_result);
        // print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_module_declaration() {
        let _input_str = r#"module nic400_cdc_capt_sync_bus1_87 ( clk, resetn, d_async, sync_en, q
        );
         input [0:0] d_async;
         output [0:0] q;
         input clk, resetn, sync_en;
         wire   d_sync1_0_;
       
         DFCNQDP40P140LVT d_sync1_reg_0_ ( .D(d_async[0]), .CP(clk), .CDN(resetn),
               .Q(d_sync1_0_) );
         DFCNQDP40P140LVT q_reg_0_ ( .D(d_sync1_0_), .CP(clk), .CDN(resetn), .Q(
               q[0]) );
       endmodule
       
    "#;
        let input_str = r#"(* top =  1  *)
module b1_comb.aig (b_pad, d_pad, e_pad, f_pad, g_pad, a_pad);
  wire _00_;
  output g_pad;
  wire g_pad;
  sky130_fd_sc_hs__clkinv_1 _26_ (
    .A(_02_),
    .Y(_05_)
  );
  assign _01_ = b_pad;
  assign f_pad = _04_;
endmodule
"#;
        let parse_result = VerilogParser::parse(Rule::module_declaration, input_str);
        println!("{:#?}", parse_result);
        // print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_verilog_file1() {
        let verilog_file_path  = "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/asic_top_flatten.v";
        let input_str = std::fs::read_to_string(verilog_file_path)
            .unwrap_or_else(|_| panic!("Can't read file: {}", verilog_file_path));

        let parse_result = VerilogParser::parse(Rule::verilog_file, input_str.as_str());
        // println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_verilog_file2() {
        let verilog_file_path  = "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/example1.v";
        let input_str = std::fs::read_to_string(verilog_file_path)
            .unwrap_or_else(|_| panic!("Can't read file: {}", verilog_file_path));

        let parse_result = VerilogParser::parse(Rule::verilog_file, input_str.as_str());
        // println!("{:#?}",parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_wire_list_with_scalar_constant() {
        let _input_str = r#"1'b0, 1'b0, rid_nic400_axi4_ps2_1_, 1'b0"#;
        let input_str1 = r#"\u0_rcg/n33 ,  \u0_rcg/n33 ,  \u0_rcg/n33"#;

        let parse_result = VerilogParser::parse(Rule::wire_list_with_scalar_constant, input_str1);
        println!("{:#?}", parse_result);
        print_parse_result(parse_result);
    }

    #[test]
    fn test_parse_verilog_file3() {
        let verilog_file_path  = "/home/longshuaiying/iEDA/src/database/manager/parser/verilog/verilog-rust/verilog-parser/example/asic_top_DC_downsize.v";
        let input_str = std::fs::read_to_string(verilog_file_path)
            .unwrap_or_else(|_| panic!("Can't read file: {}", verilog_file_path));

        let parse_result = VerilogParser::parse(Rule::verilog_file, input_str.as_str());
        println!("{:#?}", parse_result);
        // print_parse_result(parse_result);
    }

    #[test]
    fn test_extract_funs() {
        let _input1 = "gpio[3:0]";
        let _input2 = "gpio [0]";
        let input2 = "cpuregs[0][0]";
        let _input3 = "gpio";
        if let Some((name, range_from, range_to)) = extract_range(input2) {
            // extract gpio，3，0
            println!("extract_range:name={}, range_from={}, range_to={}", name, range_from, range_to);
        } else if let Some((name, index)) = extract_single(input2) {
            // extract:gpio，0
            println!("extract_single:name={}, index={}", name, index);
        } else if let Some(name) = extract_name(input2) {
            // extract:gpio
            println!("extract_name:name={}", name);
        } else {
            panic!("error format!");
        }
    }
}
