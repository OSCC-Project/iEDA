use crate::parse_sdf::data_structure;
use crate::parse_sdf::common_fun;
use pest::iterators::Pair;
use crate::parse_sdf::Rule;

fn parse_input_output_path(pairs:Pair<Rule>)->[String;2]{
    let mut in_out_path:[String;2]=[String::new(),String::new()];
    let mut n=0;
    for i in pairs.into_inner(){
        in_out_path[n]=i.as_str().to_string();
        n+=1;
    }
    in_out_path
}

fn get_path_pulse_percent_item(pairs:Pair<Rule>)->data_structure::path_pulse_percent_item{
    let mut keyword=String::new();
    let mut arr_str:[String;2]=[String::new(),String::new()];
    let mut vec:Vec<data_structure::r_value>=vec![];
    let mut flag=false;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::deltype_keyword=>{
                keyword=i.as_str().to_string();
            },
            Rule::input_output_path=>{
                arr_str=parse_input_output_path(i.clone());
                flag=true;
            },
            Rule::value=>{
                let temp=common_fun::parse_r_value(i.clone());
                vec.push(temp);
            },
            _=>(),
        }
    }
    let mut item=data_structure::path_pulse_percent_item::no_in_out_path(keyword.clone(),vec.clone());
    if flag{
        item=data_structure::path_pulse_percent_item::with_in_out_path(keyword,arr_str,vec);
    }
    item
}



fn parse_delval(pairs:Pair<Rule>)->Vec<data_structure::r_value>{
    let mut vec:Vec<data_structure::r_value>=vec![];
    for i in pairs.into_inner(){
        for j in i.into_inner(){
            match j.as_rule(){
                Rule::rvalue=>{
                    let tmp=common_fun::parse_r_value(j.clone());
                    vec.push(tmp);
                },
                _=>(),
            }
        }
    }
    vec
}

fn parse_delval_list(pairs:Pair<Rule>)->Vec<Vec<data_structure::r_value>>{
    let mut vec:Vec<Vec<data_structure::r_value>>=vec![];
    for i in pairs.into_inner(){
        for j in i.into_inner(){
            match j.as_rule(){
                Rule::delval=>{
                    let tmp=parse_delval(j.clone());
                    vec.push(tmp);
                },
                _=>(),
            }
        }
    }
    vec
}

fn parse_retain_item(pairs:Pair<Rule>)->data_structure::retain_item{
    let mut keyword:String=String::new();
    let mut vec:Vec<Vec<data_structure::r_value>>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::deltype_keyword=>{
                keyword=i.as_str().to_string();
            },
            Rule::delval_list=>{
                vec=parse_delval_list(i.clone());
            },
            _=>(),
        }
    }
    data_structure::retain_item{keyword:keyword,delval_list:vec}
}

fn parse_iopath_part1(pairs:Pair<Rule>)->(String,data_structure::port_spec,String){
    let mut keyword:String=String::new();
    let mut port_spec:data_structure::port_spec=data_structure::port_spec::port_instance(String::new());
    let mut port_instance:String=String::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::deltype_keyword=>keyword=i.as_str().to_string(),
            Rule::port_spec=>port_spec=common_fun::parse_port_spec(i.clone()),
            Rule::port_instance=>port_instance=i.as_str().to_string(),
            _=>(),
        }
    }
    (keyword,port_spec,port_instance)
}

fn parse_iopath_part2(pairs:Pair<Rule>)->(Vec<data_structure::retain_item>,Vec<Vec<data_structure::r_value>>){
    let mut vec1:Vec<data_structure::retain_item>=vec![];
    let mut vec2:Vec<Vec<data_structure::r_value>>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::iopath_part2_part=>{
                let tmp=parse_retain_item(i.clone());
                vec1.push(tmp);
            },
            Rule::delval_list=>{
                vec2=parse_delval_list(i.clone());
            },
            _=>(),
        }
    }
    (vec1,vec2)
}

fn parse_iopath_item(pairs:Pair<Rule>)->data_structure::iopath_item{
    let mut tuple1:(String,data_structure::port_spec,String)=(String::new(),data_structure::port_spec::port_instance(String::new()),String::new());
    let mut tuple2:(Vec<data_structure::retain_item>,Vec<Vec<data_structure::r_value>>)=(vec![],vec![]);
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::iopath_part1=>{
                tuple1=parse_iopath_part1(i.clone());
            },
            Rule::iopath_part2=>{
                tuple2=parse_iopath_part2(i.clone());
            },
            _=>(),
        }
    }
    let mut iopath_item:data_structure::iopath_item=data_structure::iopath_item::no_retain(tuple1.0.clone(),tuple1.1.clone(),
        tuple1.2.clone(),tuple2.1.clone());
    if tuple2.0.len()>0{
        iopath_item=data_structure::iopath_item::with_retain(tuple1.0,tuple1.1,tuple1.2,tuple2.0,tuple2.1);
    }
    iopath_item
}

fn parse_cond_item(pairs:Pair<Rule>)->data_structure::cond_item{
    let mut keyword:String=String::new();
    let mut qstr:String=String::new();
    let mut flag=false;
    let mut conditional_port_expr:String=String::new();
    let mut iopath_item:data_structure::iopath_item=data_structure::iopath_item::no_retain(String::new(),
        data_structure::port_spec::port_instance(String::new()),String::new(),vec![]);
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::deltype_keyword=>keyword=i.as_str().to_string(),
            Rule::ques_expr=>{
                qstr=i.as_str().to_string();
                flag=true;
            },
            Rule::conditional_port_expr=>conditional_port_expr=i.as_str().to_string(),
            Rule::iopath_item=>iopath_item=parse_iopath_item(i.clone()),
            _=>(),
        }
    }
    let mut cond_item=data_structure::cond_item::no_qstring(keyword.clone(),conditional_port_expr.clone(),iopath_item.clone());
    if flag{
        cond_item=data_structure::cond_item::with_qstring(keyword,qstr,conditional_port_expr,iopath_item);
    }
    cond_item
}

fn parse_condelse_item(pairs:Pair<Rule>)->data_structure::condelse_item{
    let mut keyword:String=String::new();
    let mut iopath_item:data_structure::iopath_item=data_structure::iopath_item::no_retain(String::new(),
        data_structure::port_spec::port_instance(String::new()),String::new(),vec![]);;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::deltype_keyword=>keyword=i.as_str().to_string(),
            Rule::iopath_item=>iopath_item=parse_iopath_item(i.clone()),
            _=>(),
        }
    }
    data_structure::condelse_item{keyword:keyword,iopath:iopath_item}
}

fn parse_port_item(pairs:Pair<Rule>)->data_structure::port_item{
    let mut keyword:String=String::new();
    let mut port_instance:String=String::new();
    let mut vec:Vec<Vec<data_structure::r_value>>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::deltype_keyword=>keyword=i.as_str().to_string(),
            Rule::port_instance=>port_instance=i.as_str().to_string(),
            Rule::delval_list=>vec=parse_delval_list(i.clone()),
            _=>(),
        }
    }
    data_structure::port_item{keyword:keyword,port_instance:port_instance,delval_list:vec}
}

fn parse_interconnect_item(pairs:Pair<Rule>)->data_structure::interconnect_item{
    let mut keyword:String=String::new();
    let mut input:String=String::new();
    let mut output:String=String::new();
    let mut delval_list:Vec<Vec<data_structure::r_value>>=vec![];
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::deltype_keyword=>keyword=i.as_str().to_string(),
            Rule::input_output_path=>{
                let tmp=parse_input_output_path(i.clone());
                input=tmp[0].clone();
                output=tmp[1].clone();
            },
            Rule::delval_list=>delval_list=parse_delval_list(i.clone()),
            _=>(),
        }
    }
    data_structure::interconnect_item{keyword:keyword,input:input,output:output,delval_list:delval_list}
}

fn parse_device_item(pairs:Pair<Rule>)->data_structure::device_item{
    let mut keyword:String=String::new();
    let mut port_instance:String=String::new();
    let mut delval_list:Vec<Vec<data_structure::r_value>>=vec![];
    let mut flag=false;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::deltype_keyword=>keyword=i.as_str().to_string(),
            Rule::port_instance=>{
                port_instance=i.as_str().to_string();
                flag=true;
            },
            Rule::delval_list=>delval_list=parse_delval_list(i.clone()),
            _=>(),
        }
    }
    let mut device_item:data_structure::device_item=data_structure::device_item::no_port_instance(keyword.clone(),delval_list.clone());
    if flag{
        device_item=data_structure::device_item::with_port_instance(keyword,port_instance,delval_list);
    }
    device_item
}

fn parse_del_def(pairs:Pair<Rule>)->data_structure::del_def{
    let mut item:data_structure::del_def=
        data_structure::del_def::iopath(data_structure::iopath_item::no_retain(String::new(),data_structure::port_spec::port_instance(String::new()),
        String::new(),vec![]));
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::iopath_item=>item=data_structure::del_def::iopath(parse_iopath_item(i.clone())),
            Rule::cond_item=>item=data_structure::del_def::cond(parse_cond_item(i.clone())),
            Rule::condelse_item=>item=data_structure::del_def::condelse(parse_condelse_item(i.clone())),
            Rule::port_item=>item=data_structure::del_def::port(parse_port_item(i.clone())),
            Rule::interconnect_item=>item=data_structure::del_def::interconnect(parse_interconnect_item(i.clone())),
            Rule::device_item=>item=data_structure::del_def::device(parse_device_item(i.clone())),
            _=>(),
        }
    }
    item
}

fn get_absolute_increment_item(pairs:Pair<Rule>)->data_structure::absolute_increment_item{
    let mut keyword:String=String::new();
    let mut del_defs=std::collections::LinkedList::<data_structure::del_def>::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::deltype_keyword=>keyword=i.as_str().to_string(),
            Rule::del_def=>{
                let tmp=parse_del_def(i.clone());
                del_defs.push_back(tmp);
            },
            _=>(),
        }
    }
    data_structure::absolute_increment_item{keyword:keyword,del_defs:del_defs}
}

fn parse_deltype(pairs:Pair<Rule>)->data_structure::deltype{
    let mut deltype:data_structure::deltype=data_structure::deltype::none;
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::path_pulse_percent=>{
                let temp=get_path_pulse_percent_item(i.clone());
                deltype=data_structure::deltype::path_pulse_percent(temp);
            },
            Rule::absolute_increment=>{
                let temp=get_absolute_increment_item(i.clone());
                deltype=data_structure::deltype::absolute_increment(temp);
            },
            _=>(),
        }
    }
    deltype
}

pub fn extract_del_spec(pairs:Pair<Rule>)->data_structure::del_spec{
    let mut keyword:String=String::new();
    let mut linkedlist=std::collections::LinkedList::<data_structure::deltype>::new();
    for i in pairs.into_inner(){
        match i.as_rule(){
            Rule::del_spec_keyword=>keyword=i.as_str().to_string(),
            Rule::deltype=>{
                let temp=parse_deltype(i.clone());
                linkedlist.push_back(temp);
            },
            _=>(),
        }
    }
    data_structure::del_spec{keyword:keyword,deltypes:linkedlist}
}