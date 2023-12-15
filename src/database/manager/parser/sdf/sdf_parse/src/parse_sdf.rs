pub mod parse_header;
pub mod parse_del_spec;
pub mod parse_tc_spec;
pub mod parse_te_spec;
pub mod data_structure;
pub mod common_fun;

use pest::Parser;
use pest_derive::Parser;
#[derive(Parser)]
#[grammar="sdf.pest"]
pub struct SDFParser;
use pest::iterators::Pair;
use std::fs;

fn parse_celltype(pairs:Pair<Rule>)->String{
    let mut celltype=String::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::cell_type=>celltype=i.as_str().to_string(),
            _=>(),
        }
    }
    celltype
}

fn parse_cell_instance(pairs:Pair<Rule>)->String{
    let mut instance=String::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::cellinstance_path=>{
                for j in i.into_inner(){
                    match j.as_rule(){
                        Rule::path=>instance=j.as_str().to_string(),
                        _=>(),
                    }
                }
            },
            Rule::cellinstance_wildcard=>instance="*".to_string(),
            _=>(),
        }
    }
    instance
}

fn parse_timing_spec(pairs:Pair<Rule>)->data_structure::timing_spec{
    let mut ts=data_structure::timing_spec::none;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::del_spec=>{
                let tmp=parse_del_spec::extract_del_spec(i.clone());
                ts=data_structure::timing_spec::del_spec(tmp);
            },
            Rule::tc_spec=>{
                let tmp=parse_tc_spec::extract_tc_spec(i.clone());
                ts=data_structure::timing_spec::tc_spec(tmp);
            },
            Rule::te_spec=>{
                let tmp=parse_te_spec::extract_te_spec(i.clone());
                ts=data_structure::timing_spec::te_spec(tmp);
            },
            _=>(),
        }
    }
    ts
}

fn parse_cell(pairs:Pair<Rule>)->data_structure::cell{
    let mut keyword=String::new();
    let mut cell_type=String::new();
    let mut cell_instance=String::new();
    let mut timing_spec=std::collections::LinkedList::<data_structure::timing_spec>::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::keyword=>keyword=i.as_str().to_string(),
            Rule::celltype=>cell_type=parse_celltype(i.clone()),
            Rule::cell_instance=>cell_instance=parse_cell_instance(i.clone()),
            Rule::timing_spec=>{
                let tmp=parse_timing_spec(i.clone());
                timing_spec.push_back(tmp);
            },
            _=>(),
        }
    }
    data_structure::cell{keyword:keyword,cell_type:cell_type,cell_instance:cell_instance,timing_spec:timing_spec}
}

fn get_sdf(file_name:String)->data_structure::sdf_data{
    let mut keyword=String::new();
    let mut sdf_header=std::collections::HashMap::<String,data_structure::sdf_header_val>::new();
    let mut cells=std::collections::LinkedList::<data_structure::cell>::new();

    let unparsed_text=std::fs::read_to_string(file_name).expect("read file unsuccessfully");
    let parsed_text=SDFParser::parse(Rule::delay_file,&unparsed_text);
    for i in parsed_text.unwrap().next().unwrap().into_inner(){
        match i.as_rule(){
            Rule::keyword=>keyword=i.as_str().to_string(),
            Rule::sdf_header=>sdf_header=parse_header::extract_sdf_header_info(i.clone()),
            Rule::cell=>{
                let tmp=parse_cell(i.clone());
                cells.push_back(tmp);
            },
            _=>(),
        }
    }
    data_structure::sdf_data{keyword:keyword,sdf_header:sdf_header,cell:cells}
}

pub fn parse_sdf(filename:String)->data_structure::sdf_data{
    get_sdf(filename)
}




