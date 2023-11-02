use crate::parse_sdf::data_structure;
use pest::iterators::Pair;
use crate::parse_sdf::Rule;
pub fn get_r_triple(pair:Pair<Rule>,pos:[u32;3])->data_structure::r_triple_number{
    let mut arr:[f64;3]=[0.0,0.0,0.0];
    let mut res=pair.into_inner();
    for i in 0..pos.len(){
        if pos[i]!=3{
            arr[pos[i] as usize]=res.next().unwrap().as_str().parse::<f64>().unwrap();
        }
    }
    data_structure::r_triple_number::r_triple(arr)
}

pub fn get_r_triple_by_classify(pair:Pair<Rule>)->data_structure::r_triple_number{
    let mut pos=[0,0,0];
    for i in pair.clone().into_inner(){
        match i.as_rule(){
            Rule::rtriple_type1|Rule::triple_type1=>pos=[0,3,3],
            Rule::rtriple_type2|Rule::triple_type2=>pos=[3,1,3],
            Rule::rtriple_type3|Rule::triple_type3=>pos=[3,3,2],
            Rule::rtriple_type4|Rule::triple_type4=>pos=[0,1,3],
            Rule::rtriple_type5|Rule::triple_type5=>{pos=[0,3,2];},
            Rule::rtriple_type6|Rule::triple_type6=>{pos=[0,1,2];},
            Rule::rtriple_type7|Rule::triple_type7=>{pos=[0,1,2];},
            _=>(),
        }
    }
    get_r_triple(pair.into_inner().next().unwrap().clone(),pos)
}

pub fn parse_r_triple_number(pair:Pair<Rule>)->data_structure::r_triple_number{
    let mut r=data_structure::r_triple_number::r_number(0.0);
    match pair.as_rule(){
        Rule::rnumber|Rule::number=>{
            r=data_structure::r_triple_number::r_number(pair.as_str().parse::<f64>().unwrap())
        },
        Rule::rtriple|Rule::triple=>{
            r=get_r_triple_by_classify(pair.clone());
        },
        _=>(),
    }
    r
}

pub fn parse_r_value(pairs:Pair<Rule>)->data_structure::r_value{
    let mut val=data_structure::r_value::none;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::triple|Rule::rtriple|Rule::number|Rule::rnumber=>{
                let temp=parse_r_triple_number(i.clone());
                val=data_structure::r_value::val(temp);
            },
            _=>(),
        }
    }
    val
}

pub fn parse_port_edge(pairs:Pair<Rule>)->(String,String){
    let mut edge_identifier:String=String::new();
    let mut port_instance:String=String::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::edge_identifier=>edge_identifier=i.as_str().to_string(),
            Rule::port_instance=>port_instance=i.as_str().to_string(),
            _=>(),
        }
    }
    (edge_identifier,port_instance)
}

pub fn parse_port_spec(pairs:Pair<Rule>)->data_structure::port_spec{
    let mut port_spec:data_structure::port_spec=data_structure::port_spec::port_instance(String::new());
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::port_edge=>{
                let tmp=parse_port_edge(i.clone());
                port_spec=data_structure::port_spec::port_edge(tmp.0,tmp.1);
            },
            Rule::port_instance=>{
                port_spec=data_structure::port_spec::port_instance(i.as_str().to_string());
            },
            _=>(),
        }
    }
    port_spec
}

pub fn parse_cell_instance(pairs:Pair<Rule>)->data_structure::cell_instance{
    let mut keyword=String::new();
    let mut val=String::new();
    let mut flag=false;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::cellinstance_path=>{
                for j in i.into_inner(){
                    match j.as_rule(){
                        Rule::cell_type_instance_keyword=>keyword=j.as_str().to_string(),
                        Rule::cell_type=>{
                            val=j.as_str().to_string();
                            flag=true;
                        },
                        _=>(),
                    }   
                }
            },
            Rule::cellinstance_wildcard=>{
                for j in i.into_inner(){
                    match j.as_rule(){
                        Rule::cell_type_instance_keyword=>keyword=j.as_str().to_string(),
                        Rule::wildcard=>val=j.as_str().to_string(),
                        _=>(),
                    }
                }
            },
            _=>(),
        }
    }
    if flag==false{
        data_structure::cell_instance::no_path(keyword)
    }
    else{
        data_structure::cell_instance::with_path(keyword,val)
    }
}
