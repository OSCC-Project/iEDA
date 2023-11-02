use crate::parse_sdf::data_structure;
use crate::parse_sdf::common_fun;
use pest::iterators::Pair;
use crate::parse_sdf::Rule;

fn parse_name(pairs:Pair<Rule>)->data_structure::name{
    let mut str1=String::new();
    let mut str2=String::new();
    let mut flag=false;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>str1=i.as_str().to_string(),
            Rule::sccond_qstring=>{
                str2=i.as_str().to_string();
                flag=true;
            },
            _=>(),
        }
    }
    let mut item=data_structure::name::no_qstring(str1.clone());
    if flag{
        item=data_structure::name::with_qstring(str1,str2);
    }
    item
}

fn parse_path_constraint_item(pairs:Pair<Rule>)->data_structure::path_constraint_item{
    let mut name=data_structure::name::no_qstring(String::new());
    let mut flag=false;
    let mut keyword=String::new();
    let mut ports:Vec<String>=vec![];
    let mut rvalues=[data_structure::r_value::none,data_structure::r_value::none];
    let mut n=0;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>keyword=i.as_str().to_string(),
            Rule::name=>{
                name=parse_name(i.clone());
                flag=true;
            },
            Rule::port_instance=>{
                ports.push(i.as_str().to_string());
            },
            Rule::rvalue=>{
                let tmp=common_fun::parse_r_value(i.clone());
                rvalues[n]=tmp;
                n+=1;
            },
            _=>(),
        }
    }
    let mut item=data_structure::path_constraint_item::no_name(keyword.clone(),ports.clone(),rvalues.clone());
    if flag{
        item=data_structure::path_constraint_item::with_name(keyword,name,ports,rvalues);
    }
    item
}

fn parse_exception(pairs:Pair<Rule>)->data_structure::exception{
    let mut keyword=String::new();
    let mut vec:Vec<data_structure::cell_instance>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>keyword=i.as_str().to_string(),
            Rule::cell_instance=>{
                let tmp=common_fun::parse_cell_instance(i.clone());
                vec.push(tmp);
            },
            _=>(),
        }
    }
    data_structure::exception{keyword:keyword,cell_instances:vec}
}

fn parse_period_constraint_item(pairs:Pair<Rule>)->data_structure::period_constraint_item{
    let mut keyword=String::new();
    let mut port=String::new();
    let mut val=data_structure::r_value::none;
    let mut exception=data_structure::exception{keyword:String::new(),cell_instances:vec![]};
    let mut flag=false;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>keyword=i.as_str().to_string(),
            Rule::port_instance=>port=i.as_str().to_string(),
            Rule::value=>val=common_fun::parse_r_value(i.clone()),
            Rule::exception=>{
                exception=parse_exception(i.clone());
                flag=true;
            },
            _=>(),
        }
    }
    let mut item=data_structure::period_constraint_item::no_exception(keyword.clone(),port.clone(),val.clone());
    if flag{
        item=data_structure::period_constraint_item::with_exception(keyword,port,val,exception);
    }
    item    
}

fn parse_constraint_path(pairs:Pair<Rule>)->[String;2]{
    let mut arr=[String::new(),String::new()];
    let mut n=0;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::port_instance=>{
                arr[n]=i.as_str().to_string();
                n+=1;
            },
            _=>(),
        }
    }
    arr
}

fn parse_sum_item(pairs:Pair<Rule>)->data_structure::sum_item{
    let mut keyword=String::new();
    let mut constraint_path:Vec<[String;2]>=vec![];
    let mut values:Vec<data_structure::r_value>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>keyword=i.as_str().to_string(),
            Rule::constraint_path=>{
                let tmp=parse_constraint_path(i.clone());
                constraint_path.push(tmp);
            },
            Rule::rvalue=>{
                let tmp=common_fun::parse_r_value(i.clone());
                values.push(tmp);
            },
            _=>(),
        }
    }
    data_structure::sum_item{keyword:keyword,constraint_paths:constraint_path,vals:values}
}

fn parse_diff_item(pairs:Pair<Rule>)->data_structure::diff_item{
    let mut keyword=String::new();
    let mut constraint_paths=[[String::new(),String::new()],[String::new(),String::new()]];
    let mut n=0;
    let mut vals:Vec<data_structure::r_value>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>keyword=i.as_str().to_string(),
            Rule::constraint_path=>{
                let tmp=parse_constraint_path(i.clone());
                constraint_paths[n]=tmp;
                n+=1;
            }
            Rule::value=>{
                let tmp=common_fun::parse_r_value(i.clone());
                vals.push(tmp);
            },
            _=>(),
        }
    }
    data_structure::diff_item{keyword:keyword,constraint_paths:constraint_paths,vals:vals}
}

fn parse_skew_constraint_item(pairs:Pair<Rule>)->data_structure::skew_constraint_item{
    let mut keyword=String::new();
    let mut port=data_structure::port_spec::port_instance(String::new());
    let mut value:data_structure::r_value=data_structure::r_value::none;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>keyword=i.as_str().to_string(),
            Rule::port=>port=common_fun::parse_port_spec(i.clone()),
            Rule::value=>value=common_fun::parse_r_value(i.clone()),
            _=>(),
        }
    }
    data_structure::skew_constraint_item{keyword:keyword,port:port,val:value}
}

fn parse_cns_def_item(pairs:Pair<Rule>)->data_structure::cns_def_item{
    let mut item:data_structure::cns_def_item=data_structure::cns_def_item::none;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::path_constraint_item=>{
                let tmp=parse_path_constraint_item(i.clone());
                item=data_structure::cns_def_item::path_constraint(tmp);
            },
            Rule::period_constraint_item=>{
                let tmp=parse_period_constraint_item(i.clone());
                item=data_structure::cns_def_item::period_constraint(tmp);
            },
            Rule::sum_item=>{
                let tmp=parse_sum_item(i.clone());
                item=data_structure::cns_def_item::sum(tmp);
            },
            Rule::diff_item=>{
                let tmp=parse_diff_item(i.clone());
                item=data_structure::cns_def_item::diff(tmp);
            },
            Rule::skew_constraint_item=>{
                let tmp=parse_skew_constraint_item(i.clone());
                item=data_structure::cns_def_item::skew_constraint(tmp)
            },
            _=>(),
        }
    }
    item
}

fn parse_arrival_departure_item(pairs:Pair<Rule>)->data_structure::arrival_departure_item{
    let mut keyword=String::new();
    let mut port_edge=(String::new(),String::new());
    let mut flag=false;
    let mut port_instance=String::new();
    let mut vals:[data_structure::r_value;4]=[data_structure::r_value::none,data_structure::r_value::none,
                                            data_structure::r_value::none,data_structure::r_value::none];
    let mut n=0;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>keyword=i.as_str().to_string(),
            Rule::port_edge=>{
                port_edge=common_fun::parse_port_edge(i.clone());
                flag=true;
            }
            Rule::port_instance=>port_instance=i.as_str().to_string(),
            Rule::rvalue=>{
                let tmp=common_fun::parse_r_value(i.clone());
                vals[n]=tmp;
                n+=1;
            }
            _=>(),
        }
    }
    let mut item=data_structure::arrival_departure_item::no_port_edge(keyword.clone(),port_instance.clone(),vals.clone());
    if flag{
        item=data_structure::arrival_departure_item::with_port_edge(keyword,port_edge,port_instance,vals);
    }
    item
}

fn parse_slack_item(pairs:Pair<Rule>)->data_structure::slack_item{
    let mut keyword=String::new();
    let mut port=String::new();
    let mut vals=[data_structure::r_value::none,data_structure::r_value::none,
                data_structure::r_value::none,data_structure::r_value::none];
    let mut n=0;
    let mut num:f64=0.0;
    let mut flag=false;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>keyword=i.as_str().to_string(),
            Rule::port_instance=>port=i.as_str().to_string(),
            Rule::rvalue=>{
                let tmp=common_fun::parse_r_value(i.clone());
                vals[n]=tmp;
                n+=1;
            }
            Rule::number=>{
                num=i.as_str().parse::<f64>().unwrap();
                flag=true;
            },
            _=>(),
        }
    }
    let mut item=data_structure::slack_item::no_number(keyword.clone(),port.clone(),vals.clone());
    if flag{
        item=data_structure::slack_item::with_number(keyword,port,vals,num);
    }
    item
}

fn parse_pos_neg_pair_posedge_negedge(pairs:Pair<Rule>)->data_structure::pos_neg{
    let mut keyword=String::new();
    let mut vec:Vec<f64>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::posedge|Rule::negedge=>keyword=i.as_str().to_string(),
            Rule::rnumber=>{
                let tmp=i.as_str().parse::<f64>().unwrap();
                vec.push(tmp);
            },
            _=>(),
        }
    }
    data_structure::pos_neg{keyword:keyword,nums:vec}
}

fn parse_pos_neg_pair(pairs:Pair<Rule>)->data_structure::pos_neg_pair{
    let mut res=pairs.into_inner();
    let part1=parse_pos_neg_pair_posedge_negedge(res.next().unwrap().clone());
    let part2=parse_pos_neg_pair_posedge_negedge(res.next().unwrap().clone());
    data_structure::pos_neg_pair{part1:part1,part2:part2}
}


fn parse_edge_list_type(pairs:Pair<Rule>)->Vec<data_structure::pos_neg_pair>{
    let mut edge_list:Vec<data_structure::pos_neg_pair>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::pos_pair|Rule::neg_pair=>{
                let tmp=parse_pos_neg_pair(i.clone());
                edge_list.push(tmp);
            },
            _=>{},
        }
    }
    
    edge_list
}

fn parse_edge_list(pairs:Pair<Rule>)->Vec<data_structure::pos_neg_pair>{
    let mut edge_list:Vec<data_structure::pos_neg_pair>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::edge_list_type1|Rule::edge_list_type2=>{
                edge_list=parse_edge_list_type(i.clone());
            },
            _=>(),
        }
    }
    edge_list
}

fn parse_waveform_item(pairs:Pair<Rule>)->data_structure::waveform_item{
    let mut keyword=String::new();
    let mut port=String::new();
    let mut num:f64=0.0;
    let mut edge_list:Vec<data_structure::pos_neg_pair>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>keyword=i.as_str().to_string(),
            Rule::port_instance=>port=i.as_str().to_string(),
            Rule::number=>{
                num=i.as_str().parse::<f64>().unwrap();
            },
            Rule::edge_list=>{
                edge_list=parse_edge_list(i.clone());
            },
            _=>(),
        }
    }
    data_structure::waveform_item{keyword:keyword,port:port,num:num,edge_list:edge_list}
}

fn parse_tenv_def_item(pairs:Pair<Rule>)->data_structure::tenv_def_item{
    let mut item=data_structure::tenv_def_item::none;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::arrival_departure_item=>{
                let tmp=parse_arrival_departure_item(i.clone());
                item=data_structure::tenv_def_item::arrival_departure(tmp);
            },
            Rule::slack_item=>{
                let tmp=parse_slack_item(i.clone());
                item=data_structure::tenv_def_item::slack(tmp);
            },
            Rule::waveform_item=>{
                let tmp=parse_waveform_item(i.clone());
                item=data_structure::tenv_def_item::waveform(tmp);
            },
            _=>(),
        }
    }
    item
}

fn parse_te_def(pairs:Pair<Rule>)->data_structure::te_def{
    let mut te_def_item=data_structure::te_def::none;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::cns_def=>{
                let tmp=parse_cns_def_item(i.clone());
                te_def_item=data_structure::te_def::cns_def(tmp);
            },
            Rule::tenv_def=>{
                let tmp=parse_tenv_def_item(i.clone());
                te_def_item=data_structure::te_def::tenv_def(tmp);
            },
            _=>(),
        }
    }
    te_def_item
}

pub fn extract_te_spec(pairs:Pair<Rule>)->data_structure::te_spec{
    let mut keyword:String=String::new();
    let mut linkedlist=std::collections::LinkedList::<data_structure::te_def>::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::te_keyword=>{
                keyword=i.as_str().to_string();
            },
            Rule::te_def=>{
                let tmp=parse_te_def(i.clone());
                linkedlist.push_back(tmp);
            },
            _=>(),
        }
    }
    data_structure::te_spec{keyword:keyword,te_defs:linkedlist}
}