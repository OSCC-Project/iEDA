use super::liberty_expr_data;

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use std::collections::VecDeque;

use std::ffi::c_void;
use std::os::raw::c_char;

use crate::liberty_parser::liberty_c_api::string_to_c_char;

#[derive(Parser)]
#[grammar = "liberty_parser/grammar/lib_expr.pest"]
pub struct LibertyExprParser;

/// process expr port.
fn process_port(pair: &mut Pair<Rule>) -> Result<Box<liberty_expr_data::LibertyExpr>, pest::error::Error<Rule>> {
    let mut buffer = liberty_expr_data::LibertyExpr::new(liberty_expr_data::LibertyExprOp::Buffer);
    buffer.set_port_name(String::from(pair.as_str()));
    Ok(Box::new(buffer))
}

/// process constant zero
fn process_zero(pair: &mut Pair<Rule>) -> Result<Box<liberty_expr_data::LibertyExpr>, pest::error::Error<Rule>> {
    let zero = liberty_expr_data::LibertyExpr::new(liberty_expr_data::LibertyExprOp::Zero);
    Ok(Box::new(zero))
}

/// process constant one
fn process_one(pair: &mut Pair<Rule>) -> Result<Box<liberty_expr_data::LibertyExpr>, pest::error::Error<Rule>> {
    let one = liberty_expr_data::LibertyExpr::new(liberty_expr_data::LibertyExprOp::One);
    Ok(Box::new(one))
}

/// process not expr
fn process_not(
    pair: &mut Pair<Rule>,
    substitute_queue: &mut VecDeque<Box<liberty_expr_data::LibertyExpr>>,
) -> Result<Box<liberty_expr_data::LibertyExpr>, pest::error::Error<Rule>> {
    let mut not = liberty_expr_data::LibertyExpr::new(liberty_expr_data::LibertyExprOp::Not);
    let left = substitute_queue.pop_front().unwrap();
    not.set_left(left);
    Ok(Box::new(not))
}

/// process default and expr.
fn process_default_and_expr(
    pair: &mut Pair<Rule>,
    substitute_queue: &mut VecDeque<Box<liberty_expr_data::LibertyExpr>>,
) -> Result<Box<liberty_expr_data::LibertyExpr>, pest::error::Error<Rule>> {
    let mut and = Box::new(liberty_expr_data::LibertyExpr::new(liberty_expr_data::LibertyExprOp::And));
    let mut left = substitute_queue.pop_front().unwrap();
    and.set_left(left);

    for index in 0..substitute_queue.len() {
        let right = substitute_queue.pop_front().unwrap();
        and.set_right(right);
        left = and;
        and = Box::new(liberty_expr_data::LibertyExpr::new(liberty_expr_data::LibertyExprOp::And));
    }

    Ok(and)
}

/// process expr op.
fn process_expr_op(pair: &mut Pair<Rule>) -> Result<Box<liberty_expr_data::LibertyExpr>, pest::error::Error<Rule>> {
    let op_type = match pair.as_str() {
        "+" => liberty_expr_data::LibertyExprOp::Plus,
        "|" => liberty_expr_data::LibertyExprOp::Or,
        "*" => liberty_expr_data::LibertyExprOp::Mult,
        "&" => liberty_expr_data::LibertyExprOp::And,
        "^" => liberty_expr_data::LibertyExprOp::Xor,
        _ => panic!("not supoort operator."),
    };

    let op = liberty_expr_data::LibertyExpr::new(op_type);
    Ok(Box::new(op))
}

/// process expr.
fn process_expr(
    pair: &mut Pair<Rule>,
    substitute_queue: &mut VecDeque<Box<liberty_expr_data::LibertyExpr>>,
) -> Result<Box<liberty_expr_data::LibertyExpr>, pest::error::Error<Rule>> {
    let mut left = substitute_queue.pop_front().unwrap();

    for index in (0..substitute_queue.len()).step_by(2) {
        let mut op = substitute_queue.pop_front().unwrap();
        op.set_left(left);
        let right = substitute_queue.pop_front().unwrap();
        op.set_right(right);
        left = op;
    }

    Ok(left)
}

/// process pest pair data.
fn process_pair(
    pair: &mut Pair<Rule>,
    parser_queue: &mut VecDeque<Box<liberty_expr_data::LibertyExpr>>,
) -> Result<Box<liberty_expr_data::LibertyExpr>, pest::error::Error<Rule>> {
    let current_queue_len = parser_queue.len();
    for mut inner_pair in pair.clone().into_inner() {
        let pair_result = process_pair(&mut inner_pair, parser_queue);
        parser_queue.push_back(pair_result.unwrap());
    }

    // println!("Rule:    {:?}", pair.as_rule());
    // println!("Span:    {:?}", pair.as_span());
    // println!("Text:    {}", pair.as_str());

    let mut substitute_queue: VecDeque<Box<liberty_expr_data::LibertyExpr>> = VecDeque::new();
    while parser_queue.len() > current_queue_len {
        substitute_queue.push_front(parser_queue.pop_back().unwrap());
    }

    match pair.as_rule() {
        Rule::port => process_port(pair),
        Rule::zero => process_zero(pair),
        Rule::one => process_one(pair),
        Rule::not_expr => process_not(pair, &mut substitute_queue),
        Rule::default_and_expr => process_default_and_expr(pair, &mut substitute_queue),
        Rule::expr_op => process_expr_op(pair),
        Rule::expr => process_expr(pair, &mut substitute_queue),
        _ => Err(pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError { message: "Unknown rule".into() },
            pair.as_span(),
        )),
    }
}

/// process vcd file data.
pub fn parse_expr(expr_str: &str) -> Result<Box<liberty_expr_data::LibertyExpr>, pest::error::Error<Rule>> {
    let parse_result = LibertyExprParser::parse(Rule::expr_result, expr_str);
    let mut parser_queue: VecDeque<Box<liberty_expr_data::LibertyExpr>> = VecDeque::new();

    match parse_result {
        Ok(pairs) => {
            let mut pair = pairs.into_iter().next().unwrap();
            let lib_expr = process_pair(&mut pair, &mut parser_queue);
            lib_expr
        }
        Err(err) => {
            // Handle parsing error
            println!("Error: {}", err);
            Err(err.clone())
        }
    }
}

#[no_mangle]
pub extern "C" fn rust_parse_expr(expr_str: *const c_char) -> *mut c_void {
    let c_expr_str = unsafe { std::ffi::CStr::from_ptr(expr_str) };
    let mut r_expr_str = c_expr_str.to_string_lossy().into_owned();
    // println!("r str {}", r_expr_str);

    if (!r_expr_str.ends_with(";")) {
        r_expr_str = r_expr_str + ";";
    }

    let lib_expr = parse_expr(&r_expr_str);
    assert!(lib_expr.is_ok());

    let raw_pointer = Box::into_raw(lib_expr.unwrap());
    raw_pointer as *mut c_void
}

#[repr(C)]
pub struct RustLibertyExpr {
    op: liberty_expr_data::LibertyExprOp,
    left: *mut c_void,
    right: *mut c_void,
    port_name: *mut c_char,
}

/// convert expr to c expr.
#[no_mangle]
pub extern "C" fn rust_convert_expr(c_expr: *mut liberty_expr_data::LibertyExpr) -> *mut RustLibertyExpr {
    unsafe {
        let op = (*c_expr).get_op();
        let left = (*c_expr).get_left();
        let right = (*c_expr).get_right();
        let port_name = (*c_expr).get_port_name();

        let c_left =
            if left.is_some() { left.as_deref().unwrap() as *const _ as *mut c_void } else { std::ptr::null_mut() };
        let c_right =
            if right.is_some() { right.as_deref().unwrap() as *const _ as *mut c_void } else { std::ptr::null_mut() };
        let c_port_name =
            if port_name.is_some() { string_to_c_char(&port_name.as_deref().unwrap()) } else { std::ptr::null_mut() };

        let expr =
            RustLibertyExpr { op, left: c_left as *mut c_void, right: c_right as *mut c_void, port_name: c_port_name };
        let expr_pointer = Box::new(expr);
        Box::into_raw(expr_pointer)
    }
}

#[cfg(test)]
mod tests {

    use crate::liberty_parser::liberty_expr::parse_expr;

    #[test]
    fn test_parse_expr_path() {
        let expr_str = "(A1 A2);";
        let parse_result = parse_expr(expr_str);
        assert!(parse_result.is_ok());
    }
}
