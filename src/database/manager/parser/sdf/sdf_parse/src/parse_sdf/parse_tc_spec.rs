/*
    parse tc_spec、tchk_def
*/
use crate::parse_sdf::data_structure;
use crate::parse_sdf::common_fun;
use pest::iterators::Pair;
use crate::parse_sdf::Rule;

fn parse_scalar_node(pairs:Pair<Rule>)->data_structure::scalar_node_item{
    let mut port:String=String::new();
    let mut net:String=String::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::scalar_port=>port=i.as_str().to_string(),
            Rule::scalar_net=>net=i.as_str().to_string(),
            _=>(),
        }
    }
    data_structure::scalar_node_item{scalar_port:port,scalar_net:net}
}

fn parse_timing_check_condtion_type2(pairs:Pair<Rule>)->(String,data_structure::scalar_node_item){
    let mut inv_id:String=String::new();
    let mut scalar_node:data_structure::scalar_node_item=data_structure::scalar_node_item{scalar_port:String::new(),scalar_net:String::new()};
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::INVERSION_OPERATOR=>inv_id=i.as_str().to_string(),
            Rule::scalar_node=>scalar_node=parse_scalar_node(i.clone()),
            _=>(),
        }
    }
    (inv_id,scalar_node)
}

fn parse_timing_check_condtion_type3(pairs:Pair<Rule>)->(data_structure::scalar_node_item,String,String){
    let mut scalar_node:data_structure::scalar_node_item=data_structure::scalar_node_item{scalar_port:String::new(),scalar_net:String::new()};
    let mut equal:String=String::new();
    let mut scalar_const:String=String::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::scalar_node=>scalar_node=parse_scalar_node(i.clone()),
            Rule::EQUALITY_OPERATOR=>equal=i.as_str().to_string(),
            Rule::SCALAR_CONSTANT=>scalar_const=i.as_str().to_string(),
            _=>(),
        }
    }
    (scalar_node,equal,scalar_const)
}

fn parse_timing_check_condition(pairs:Pair<Rule>)->data_structure::timing_check_condition_item{
    let mut item:data_structure::timing_check_condition_item=data_structure::timing_check_condition_item::scalar_node(
        data_structure::scalar_node_item{scalar_port:String::new(),scalar_net:String::new()}
    );
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::timing_check_condition_type3=>{
                let tmp=parse_timing_check_condtion_type3(i.clone());
                item=data_structure::timing_check_condition_item::node_equal_const(tmp.0,tmp.1,tmp.2);
            },
            Rule::timing_check_condition_type2=>{
                let tmp=parse_timing_check_condtion_type2(i.clone());
                item=data_structure::timing_check_condition_item::inv_scalar_node(tmp.0,tmp.1);
            },
            Rule::timing_check_condition_type1=>{
                let tmp=parse_scalar_node(i.clone().into_inner().next().unwrap());
                item=data_structure::timing_check_condition_item::scalar_node(tmp);
            },
            _=>(),
        }
    }
    item
}

fn parse_cond(pairs:Pair<Rule>)->data_structure::cond{
    let mut keyword:String=String::new();
    let mut qstr:String=String::new();
    let mut timing_check_condition:data_structure::timing_check_condition_item=data_structure::timing_check_condition_item::scalar_node(
        data_structure::scalar_node_item{scalar_port:String::new(),scalar_net:String::new()}
    );
    let mut port_spec:data_structure::port_spec=data_structure::port_spec::port_instance(String::new());
    let mut flag=false;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::port_tchk_type2_keyword=>keyword=i.as_str().to_string(),
            Rule::sccond_qstring=>{
                qstr=i.as_str().to_string();
                flag=true;
            },
            Rule::timing_check_condition=>timing_check_condition=parse_timing_check_condition(i.clone()),
            Rule::port_spec=>port_spec=common_fun::parse_port_spec(i.clone()),
            _=>(),
        }
    }
    let mut item=data_structure::cond::no_qstring(keyword.clone(),timing_check_condition.clone(),port_spec.clone());
    if flag{
        item=data_structure::cond::with_qstring(keyword,qstr,timing_check_condition,port_spec);
    }
    item
}

fn parse_port_tchk(pairs:Pair<Rule>)->data_structure::port_tchk{
    let mut port:data_structure::port_tchk=data_structure::port_tchk::port_spec(
        data_structure::port_spec::port_instance(String::new()));
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::port_tchk_type1=>{
                let tmp=common_fun::parse_port_spec(i.clone().into_inner().next().unwrap());
                port=data_structure::port_tchk::port_spec(tmp);
            },
            Rule::port_tchk_type2=>{
                let tmp=parse_cond(i.clone());
                port=data_structure::port_tchk::cond(tmp);
            },
            _=>(),
        }
    }
    port
}

fn parse_setup_hold_recovery_removal_item(pairs:Pair<Rule>)->data_structure::setup_hold_recovery_removal_item{
    let mut keyword:String=String::new();
    let mut val=data_structure::r_value::none;
    let mut arr_port:[data_structure::port_tchk;2]=[data_structure::port_tchk::port_spec(data_structure::port_spec::port_instance(String::new())),
        data_structure::port_tchk::port_spec(data_structure::port_spec::port_instance(String::new()))];
    let mut n=0;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::tchk_keyword=>keyword=i.as_str().to_string(),
            Rule::port_tchk=>{
                let tmp=parse_port_tchk(i.clone());
                arr_port[n]=tmp;
                n=n+1;
            },
            Rule::value=>val=common_fun::parse_r_value(i.clone()),
            _=>(),
        }
    }
    data_structure::setup_hold_recovery_removal_item{keyword:keyword,in_out:arr_port,val:val}
}



fn parse_setuphold_item1_recrem_item1_nochange_item(pairs:Pair<Rule>)->data_structure::setuphold_item1_recrem_item1_nochange_item{
    let mut arr_port=[data_structure::port_tchk::port_spec(data_structure::port_spec::port_instance(String::new())),
        data_structure::port_tchk::port_spec(data_structure::port_spec::port_instance(String::new()))];
    let mut arr_val=[data_structure::r_value::none,data_structure::r_value::none];
    let mut n1=0;
    let mut n2=0;
    let mut keyword:String=String::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::tchk_keyword=>keyword=i.as_str().to_string(),
            Rule::port_tchk=>{
                let tmp=parse_port_tchk(i.clone());
                arr_port[n1]=tmp;
                n1+=1;
            },
            Rule::rvalue=>{
                let tmp=common_fun::parse_r_value(i.clone());
                arr_val[n2]=tmp;
                n2+=1;
            },
            _=>(),
        }
    }
    data_structure::setuphold_item1_recrem_item1_nochange_item{keyword:keyword,in_out:arr_port,value:arr_val}
}

fn parse_sccond(pairs:Pair<Rule>)->data_structure::sccond_item{
    let mut keyword:String=String::new();
    let mut qstr:String=String::new();
    let mut timing_check_condition:data_structure::timing_check_condition_item=data_structure::timing_check_condition_item::scalar_node(
        data_structure::scalar_node_item{scalar_port:String::new(),scalar_net:String::new()});
    let mut flag=true;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::sccond_keyword=>keyword=i.as_str().to_string(),
            Rule::sccond_qstring=>{
                qstr=i.as_str().to_string();
                flag=true;
            },
            Rule::timing_check_condition=>timing_check_condition=parse_timing_check_condition(i.clone()),
            _=>(),
        }
    }
    let mut item=data_structure::sccond_item::no_qstring(keyword.clone(),timing_check_condition.clone());
    if flag{
        item=data_structure::sccond_item::with_qstring(keyword,qstr,timing_check_condition);
    }
    item
}

fn parse_setuphold_item2_recrem_item2_item(pairs:Pair<Rule>)->data_structure::setuphold_item2_recrem_item2_item{
    let mut keyword:String=String::new();
    let mut port_specs=[data_structure::port_spec::port_instance(String::new()),data_structure::port_spec::port_instance(String::new())];
    let mut rvalues=[data_structure::r_value::none,data_structure::r_value::none];
    let mut n1=0;
    let mut n2=0;
    let mut scconds:Vec<data_structure::sccond_item>=vec![];
    let mut flag=false;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::tchk_keyword=>keyword=i.as_str().to_string(),
            Rule::port_spec=>{
                let tmp=common_fun::parse_port_spec(i.clone());
                port_specs[n1]=tmp;
                n1+=1;
            },
            Rule::rvalue=>{
                let tmp=common_fun::parse_r_value(i.clone());
                rvalues[n2]=tmp;
                n2+=1;
            },
            Rule::sccond=>{
                let tmp=parse_sccond(i.clone());
                scconds.push(tmp);
                flag=true;
            },
            _=>(),
        }
    }
    let mut item=data_structure::setuphold_item2_recrem_item2_item::no_sccond(keyword.clone(),port_specs.clone(),rvalues.clone());
    if flag{
        item=data_structure::setuphold_item2_recrem_item2_item::with_sccond(keyword,port_specs,rvalues,scconds);
    }
    item
}

fn parse_skew_item(pairs:Pair<Rule>)->data_structure::skew_item{
    let mut keyword:String=String::new();
    let mut port_tchks=[data_structure::port_tchk::port_spec(data_structure::port_spec::port_instance(String::new())),
        data_structure::port_tchk::port_spec(data_structure::port_spec::port_instance(String::new()))];
    let mut value:data_structure::r_value=data_structure::r_value::none;
    let mut n=0;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::tchk_keyword=>keyword=i.as_str().to_string(),
            Rule::port_tchk=>{
                let tmp=parse_port_tchk(i.clone());
                port_tchks[n]=tmp;
                n+=1;
            },
            Rule::rvalue=>value=common_fun::parse_r_value(i.clone()),
            _=>(),
        }
    }
    data_structure::skew_item{keyword:keyword,in_out:port_tchks,val:value}
}   

fn parse_width_period_item(pairs:Pair<Rule>)->data_structure::width_period_item{
    let mut keyword:String=String::new();
    let mut port:data_structure::port_tchk=data_structure::port_tchk::port_spec(data_structure::port_spec::port_instance(String::new()));
    let mut val:data_structure::r_value=data_structure::r_value::none;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::tchk_keyword=>keyword=i.as_str().to_string(),
            Rule::port_tchk=>port=parse_port_tchk(i.clone()),
            Rule::value=>val=common_fun::parse_r_value(i.clone()),
            _=>(),
        }
    }
    data_structure::width_period_item{keyword:keyword,port:port,val:val}
}

fn parse_tchk_def(pairs:Pair<Rule>)->data_structure::tchk_def{
    let mut res:data_structure::tchk_def=data_structure::tchk_def::none;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::setup_hold_recovery_removal_item=>{
                let tmp=parse_setup_hold_recovery_removal_item(i.clone());
                res=data_structure::tchk_def::setup_hold_recovery_removal(tmp);
            },
            Rule::setuphold_item1_recrem_item1_nochange_item=>{
                let tmp=parse_setuphold_item1_recrem_item1_nochange_item(i.clone());
                res=data_structure::tchk_def::setuphold_item1_recrem_item1_nochange(tmp);
            },
            Rule::setuphold_item2_recrem_item2_item=>{
                let tmp=parse_setuphold_item2_recrem_item2_item(i.clone());
                res=data_structure::tchk_def::setuphold_item2_recrem_item2(tmp);
            },
            Rule::skew_item=>{
                let tmp=parse_skew_item(i.clone());
                res=data_structure::tchk_def::skew(tmp);
            },
            Rule::width_period_item=>{
                let tmp=parse_width_period_item(i.clone());
                res=data_structure::tchk_def::width_period(tmp);
            },
            _=>(),
        }
    }
    res
}

pub fn extract_tc_spec(pairs:Pair<Rule>)->data_structure::tc_spec{
    let mut keyword=String::new();
    let mut linkedlist=std::collections::LinkedList::<data_structure::tchk_def>::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::tchk_keyword=>keyword=i.as_str().to_string(),
            Rule::tchk_def=>{
                let tmp=parse_tchk_def(i.clone());
                linkedlist.push_back(tmp);
            },
            _=>(),
        }
    }
    data_structure::tc_spec{keyword:keyword,tchk_defs:linkedlist}
}