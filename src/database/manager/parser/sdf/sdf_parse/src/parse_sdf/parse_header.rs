use crate::parse_sdf::data_structure;
use crate::parse_sdf::common_fun;
use pest::iterators::Pair;
use crate::parse_sdf::Rule;
fn get_sdf_header_val_key(pair:Pair<Rule>)->(String,data_structure::sdf_header_val){
    let mut tuple:(String,data_structure::sdf_header_val)=(String::new(),data_structure::sdf_header_val::str(String::new()));
    let mut time_scale=data_structure::time_scale_structure{number:0.0,unit:String::new()};
    let mut time_scale_number=1.0;
    let mut time_scale_unit="ns".to_string();
    for i in pair.into_inner(){
        match i.as_rule(){
            Rule::sdf_header_keyword=>{
                tuple.0=i.as_str().to_string();
            },
            Rule::val_string|Rule::divider=>{
                let temp=i.as_str().to_string();
                tuple.1=data_structure::sdf_header_val::str(temp.clone());
            },
            Rule::rtriple|Rule::rnumber=>{
                tuple.1=data_structure::sdf_header_val::val(common_fun::parse_r_triple_number(i.clone()));  
            },
            Rule::time_scale_number=>time_scale_number=i.as_str().parse::<f64>().unwrap(),
            Rule::time_scale_unit=>{
                time_scale_unit=i.as_str().to_string();
                tuple.1=data_structure::sdf_header_val::time_scale(
                    data_structure::time_scale_structure{number:time_scale_number,unit:time_scale_unit}
                );
            },
            _=>(),
        }
    }
    tuple
}

pub fn extract_sdf_header_info(pair:Pair<Rule>)->std::collections::HashMap::<String,data_structure::sdf_header_val>{
    let mut header_info=std::collections::HashMap::<String,data_structure::sdf_header_val>::new();
    let mut temp:(String,data_structure::sdf_header_val);
    for i in pair.into_inner(){
        temp=get_sdf_header_val_key(i.clone());
        header_info.insert(temp.0,temp.1);
    }
    header_info
}