use crate::parse_sdf::data_structure;
use crate::parse_sdf::parse_header;
use crate::parse_sdf::parse_del_spec;
use crate::parse_sdf::parse_tc_spec;
use crate::parse_sdf::parse_te_spec;

/* use pest::Parser;
use pest_derive::Parser;
#[derive(Parser)]
#[grammar="sdf.pest"]
pub struct SDFParser;
use pest::iterators::Pair;
use std::fs; */
use crate::parse_sdf::SDFParser;
use crate::parse_sdf::Rule;
use pest::Parser;

//use pest::iterators::Pair;

pub fn test_sdf_header(){
    let text=SDFParser::parse(Rule::sdf_header,"(SDFVERSION \"3.0\")
    (DESIGN \"picorv32\")
    (DATE \"Fri Mar 24 09:04:43 2023\")
    (VENDOR \"Parallax\")
    (PROGRAM \"STA\")
    (VERSION \"2.4.0\")
    (DIVIDER .)
    (VOLTAGE 1.800::1.800)
    (PROCESS \"1.000::1.000\")
    (TEMPERATURE 25.000::25.000)
    (TIMESCALE 1ns)");
    assert_eq!(text.is_ok(),true);
    let temp=parse_header::extract_sdf_header_info(text.unwrap().next().unwrap().clone());
    for (key,val) in temp{
        print!("{} ",key);
        match val{
            data_structure::sdf_header_val::str(string)=>println!("{}",string),
            data_structure::sdf_header_val::val(num_arr)=>{
                match num_arr{
                    data_structure::r_triple_number::r_triple(arr)=>println!("{:?}",arr),
                    data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                    _=>(),
                }
            },
            data_structure::sdf_header_val::time_scale(time_scale)=>println!("{}{}",time_scale.number,time_scale.unit),
            _=>(),
        }
    }
}





pub fn test_del_spec(){
    let text=SDFParser::parse(Rule::del_spec,"(DELAY
        (ABSOLUTE
         (IOPATH A X (0.112:0.112:0.112) (34))
         (IOPATH B X (0.128:0.128:0.128) (0.286:0.286:0.286))
         (PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
        )
        (PATHPULSEPERCENT a y (25) (35))
       )");
    let tmp=parse_del_spec::extract_del_spec(text.unwrap().next().unwrap().clone());
    println!("{}",tmp.keyword);
    for i in tmp.deltypes.iter(){
        match i{
            data_structure::deltype::path_pulse_percent(path_pulse_percent_item)=>{
                match path_pulse_percent_item{
                    data_structure::path_pulse_percent_item::no_in_out_path(str1,vec)=>{
                        println!("{}",str1);
                        for i in 0..vec.len(){
                            match &vec[i]{
                                data_structure::r_value::val(val)=>{
                                    match val{
                                        data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                        data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                        _=>(),
                                    }
                                },
                                data_structure::r_value::none=>(),
                                _=>(),
                            }
                        }
                    },
                    data_structure::path_pulse_percent_item::with_in_out_path(str1,str2,vec)=>{
                        println!("{} {} {}",str1,str2[0],str2[1]);
                        for i in 0..vec.len(){
                            match &vec[i]{
                                data_structure::r_value::val(val)=>{
                                    match val{
                                        data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                        data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                        _=>(),
                                    }
                                },
                                data_structure::r_value::none=>(),
                                _=>(),
                            }
                        }

                    },
                    _=>(),
                }
            },
            data_structure::deltype::absolute_increment(absolute_increment_item)=>{
                println!("{}",absolute_increment_item.keyword);
                for j in absolute_increment_item.del_defs.iter(){
                    match j{
                        data_structure::del_def::iopath(iopath_item)=>{
                            match iopath_item{
                                data_structure::iopath_item::no_retain(str1,port_spec,str2,vec)=>{
                                    println!("{} {}",str1,str2);

                                    for i in vec.into_iter(){
                                        for j in i.into_iter(){
                                            match j{
                                                data_structure::r_value::val(val)=>{
                                                    match val{
                                                        data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                        data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                        _=>(),
                                                    }
                                                },
                                                data_structure::r_value::none=>(),
                                                _=>(),
                                            }
                                        }
                                    }

                                }
                                data_structure::iopath_item::with_retain(str1,port_spec,str2,a,vec)=>(),
                                _=>(),
                            }
                        }
                        data_structure::del_def::cond(cond_item)=>(),
                        data_structure::del_def::condelse(condelse_item)=>(),
                        data_structure::del_def::port(port_item)=>{
                            println!("{} {}",port_item.keyword,port_item.port_instance);
                            for i in 0..port_item.delval_list.len(){
                                for j in 0..port_item.delval_list[i].len(){
                                    match &port_item.delval_list[i][j]{
                                        data_structure::r_value::val(val)=>{
                                            match val{
                                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                _=>(),
                                            }
                                        },
                                        data_structure::r_value::none=>(),
                                        _=>(),
                                    }
                                }
                            }
                        },
                        data_structure::del_def::interconnect(interconnect_item)=>(),
                        data_structure::del_def::device(device_item)=>(),
                        _=>(),
                    }
                }
            },
            _=>(),
        }
    }
}




pub fn test_tc_spec(){
    let text=SDFParser::parse(Rule::tc_spec,"(TIMINGCHECK 
        (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
        (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
        (WIDTH (COND \"cybup8848\" !mem[5] cyb mem[3:5]) (2:3:))
        (SKEW (COND !mem[5] cyb mem[3:5]) mem[2] (-2:3:))
    )");
    let tmp=parse_tc_spec::extract_tc_spec(text.unwrap().next().unwrap().clone());
    println!("{}",tmp.keyword);
    for i in tmp.tchk_defs.iter(){
        match i{
            data_structure::tchk_def::setup_hold_recovery_removal(item)=>{
                println!("{}",item.keyword);
                for i in 0..2{
                    match &item.in_out[1]{
                        data_structure::port_tchk::port_spec(port_spec)=>{
                            match port_spec{
                                data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                _=>(),
                            }
                        },
                        data_structure::port_tchk::cond(cond)=>{
                            match cond{
                                data_structure::cond::no_qstring(str1,timing_check_condition_item,port_spec)=>{
                                    println!("{}",str1);
                                    match port_spec{
                                        data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                        data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                        _=>(),
                                    }
                                },
                                data_structure::cond::with_qstring(str1,str2,timing_check_condition_item,port_spec)=>{
                                    println!("{} {}",str1,str2);
                                    match port_spec{
                                        data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                        data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                        _=>(),
                                    }
                                },
                                _=>(),
                            }
                        },
                        _=>(),
                    }
                }
            },
            data_structure::tchk_def::setuphold_item1_recrem_item1_nochange(item)=>{
                println!("{}",item.keyword);
                for i in 0..2{
                    match &item.in_out[i]{
                        data_structure::port_tchk::port_spec(port_spec)=>{
                            match port_spec{
                                data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                _=>(),
                            }
                        },
                        data_structure::port_tchk::cond(cond)=>{
                            match cond{
                                data_structure::cond::no_qstring(str1,timing_check_condition_item,port_spec)=>{
                                    println!("{}",str1);
                                    match port_spec{
                                        data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                        data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                        _=>(),
                                    }
                                },
                                data_structure::cond::with_qstring(str1,str2,timing_check_condition_item,port_spec)=>{
                                    println!("{} {}",str1,str2);
                                    match port_spec{
                                        data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                        data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                        _=>(),
                                    }
                                },
                                _=>(),
                            }
                        },
                        _=>(),
                    }
                }
                for i in 0..2{
                    match &item.value[i]{
                        data_structure::r_value::val(val)=>{
                            match val{
                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                _=>(),
                            }
                        },
                        data_structure::r_value::none=>(),
                        _=>(),
                    }
                }
            },
            data_structure::tchk_def::setuphold_item2_recrem_item2(setuphold_item2_recrem_item2_item)=>{

            },
            data_structure::tchk_def::skew(item)=>{
                println!("{}",item.keyword);
                for i in 0..2{
                    match &item.in_out[i]{
                        data_structure::port_tchk::port_spec(port_spec)=>{
                            match port_spec{
                                data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                _=>(),
                            }
                        },
                        data_structure::port_tchk::cond(cond)=>{
                            match cond{
                                data_structure::cond::no_qstring(str1,timing_check_condition_item,port_spec)=>{
                                    println!("{}",str1);
                                    match port_spec{
                                        data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                        data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                        _=>(),
                                    }
                                },
                                data_structure::cond::with_qstring(str1,str2,timing_check_condition_item,port_spec)=>{
                                    println!("{} {}",str1,str2);
                                    match port_spec{
                                        data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                        data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                        _=>(),
                                    }
                                },
                                _=>(),
                            }
                        },
                        _=>(),
                    }
                }

                match item.val.clone(){
                    data_structure::r_value::val(val)=>{
                        match val{
                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                _=>(),
                            }
                    },
                    _=>()
                }
            },
            data_structure::tchk_def::width_period(item)=>{
                println!("{}",item.keyword);
                match &item.port{
                    data_structure::port_tchk::port_spec(port_spec)=>{
                        match port_spec{
                            data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                            data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                            _=>(),
                        }
                    },
                    data_structure::port_tchk::cond(cond)=>{
                        match cond{
                            data_structure::cond::no_qstring(str1,timing_check_condition_item,port_spec)=>{
                                println!("{}",str1);
                                match port_spec{
                                    data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                    data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                    _=>(),
                                }
                            },
                            data_structure::cond::with_qstring(str1,str2,timing_check_condition_item,port_spec)=>{
                                println!("{} {}",str1,str2);
                                match port_spec{
                                    data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                                    data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                                    _=>(),
                                }
                            },
                            _=>(),
                        }
                    },
                    _=>(),
                }
                match item.val.clone(){
                    data_structure::r_value::val(val)=>{
                        match val{
                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                _=>(),
                            }
                    },
                    _=>()
                }
            },
            _=>(),
        }
    }
}


pub fn test_te_spec(){
    let text=SDFParser::parse(Rule::te_spec,"(TIMINGENV 
        (PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] () (-2:3:4))
        (PERIODCONSTRAINTITEM mem[2] (2:3:4))
        (SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))
        (WAVEFORM a.c.v.mem 2.1535 
            (posedge 1.2) (negedge 1.2 2.3) (posedge 2.3 4.5) (negedge 2.3 4.5))
        (SKEWCONSTRAINT mem[2:3] (2))
        (DEPARTURE a.c.v.mem () (2::) (-2) (-2::0))
    )");
    let pairs=text.unwrap().next().unwrap();
    let tmp=parse_te_spec::extract_te_spec(pairs.clone());
    println!("{}",tmp.keyword);
    for i in tmp.te_defs.iter(){
        match i{
            data_structure::te_def::cns_def(cns_def_item)=>{
                match cns_def_item{
                    data_structure::cns_def_item::path_constraint(path_constraint_item)=>{
                        match path_constraint_item{
                            data_structure::path_constraint_item::no_name(str1,vec_str,r_vals)=>{
                                println!("{} {:?}",str1,vec_str);
                                for i in 0..2{
                                    match &r_vals[i]{
                                        data_structure::r_value::val(val)=>{
                                            match val{
                                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                _=>(),
                                            }
                                        },
                                        _=>(),
                                    }
                                }
                            },
                            data_structure::path_constraint_item::with_name(str1,name,vec_str,r_vals)=>{
                                println!("{} {:?}",str1,vec_str);
                                for i in 0..2{
                                    match &r_vals[i]{
                                        data_structure::r_value::val(val)=>{
                                            match val{
                                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                _=>(),
                                            }
                                        },
                                        _=>(),
                                    }
                                }
                                match name{
                                    data_structure::name::no_qstring(str1)=>println!("{}",str1),
                                    data_structure::name::with_qstring(str1,str2)=>println!("{} {}",str1,str2),
                                    _=>(),
                                }
                            },
                            _=>(),
                        }
                    },
                    data_structure::cns_def_item::period_constraint(period_constraint_item)=>{
                        match period_constraint_item{
                            data_structure::period_constraint_item::no_exception(str1,str2,val)=>{
                                println!("{} {}",str1,str2);
                                match val{
                                    data_structure::r_value::val(val)=>{
                                        match val{
                                            data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                            data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                            _=>(),
                                        }
                                    },
                                    _=>(),
                                }
                            },
                            data_structure::period_constraint_item::with_exception(str1,str2,val,exception)=>{
                                println!("{} {}",str1,str2);
                                match val{
                                    data_structure::r_value::val(val)=>{
                                        match val{
                                            data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                            data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                            _=>(),
                                        }
                                    },
                                    _=>(),
                                }
                                println!("{:?}",exception.keyword);
                            },
                            _=>(),
                        }   
                    },
                    data_structure::cns_def_item::sum(sum_item)=>{
                        println!("{}",sum_item.keyword);
                        for i in sum_item.constraint_paths.iter(){
                            for j in i.iter(){
                                print!("{} ",j);
                            }
                            println!();
                        }
                        for i in sum_item.vals.iter(){
                            match i{
                                data_structure::r_value::val(val)=>{
                                    match val{
                                        data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                        data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                        _=>(),
                                    }
                                },
                                _=>(),
                            }
                        }
                    },
                    data_structure::cns_def_item::diff(diff_item)=>{
                        println!("{}",diff_item.keyword);
                        for i in 0..2{
                            println!("{:?}",diff_item.constraint_paths[i]);
                        }
                        for i in diff_item.vals.iter(){
                            match i{
                                data_structure::r_value::val(val)=>{
                                    match val{
                                        data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                        data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                        _=>(),
                                    }
                                },
                                _=>(),
                            }
                        }
                    },
                    data_structure::cns_def_item::skew_constraint(skew_constraint_item)=>{
                        println!("{}",skew_constraint_item.keyword);
                        match skew_constraint_item.port.clone(){
                            data_structure::port_spec::port_instance(str1)=>println!("{}",str1),
                            data_structure::port_spec::port_edge(str1,str2)=>println!("{} {}",str1,str2),
                            _=>(),
                        }
                        match skew_constraint_item.val.clone(){
                            data_structure::r_value::val(val)=>{
                                match val{
                                    data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                    data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                    _=>(),
                                }
                            },
                            _=>(),
                        }
                    },
                    _=>(),
                }
            },
            data_structure::te_def::tenv_def(tenv_def_item)=>{
                match tenv_def_item{
                    data_structure::tenv_def_item::arrival_departure(arrival_departure_item)=>{
                        match arrival_departure_item{
                            data_structure::arrival_departure_item::no_port_edge(str1,str2,vec)=>{
                                println!("{} {}",str1,str2);
                                for i in 0..4{
                                    match &vec[i]{
                                        data_structure::r_value::val(val)=>{
                                            match val{
                                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                _=>(),
                                            }
                                        },
                                        _=>(),
                                    }
                                }
                            }, 
                            data_structure::arrival_departure_item::with_port_edge(str1,tuple,str2,vec)=>{
                                println!("{} {} ({} {})",str1,str2,tuple.0,tuple.1);
                                for i in 0..4{
                                    match &vec[i]{
                                        data_structure::r_value::val(val)=>{
                                            match val{
                                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                _=>(),
                                            }
                                        },
                                        _=>(),
                                    }
                                }
                            },
                            _=>(),
                        }
                    },
                    data_structure::tenv_def_item::slack(slack_item)=>{
                        match slack_item{
                            data_structure::slack_item::no_number(str1,str2,vec)=>{
                                println!("{} {}",str1,str2);
                                for i in 0..4{
                                    match &vec[i]{
                                        data_structure::r_value::val(val)=>{
                                            match val{
                                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                _=>(),
                                            }
                                        },
                                        _=>(),
                                    }
                                }
                            },
                            data_structure::slack_item::with_number(str1,str2,vec,num)=>{
                                println!("{} {} {}",str1,str2,num);
                                for i in 0..4{
                                    match &vec[i]{
                                        data_structure::r_value::val(val)=>{
                                            match val{
                                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                _=>(),
                                            }
                                        },
                                        _=>(),
                                    }
                                }
                            },
                            _=>(),
                        }
                    },
                    data_structure::tenv_def_item::waveform(waveform_item)=>{
                        println!("{} {} {}",waveform_item.keyword,waveform_item.port,waveform_item.num);
                    },
                    _=>(),
                }
            }
            _=>(),
        }
    }
}


use crate::parse_sdf;

fn print_del_spec(tmp:&data_structure::del_spec){
    for i in tmp.deltypes.iter(){
        match i{
            data_structure::deltype::path_pulse_percent(path_pulse_percent_item)=>{
                match path_pulse_percent_item{
                    data_structure::path_pulse_percent_item::no_in_out_path(str1,vec)=>{
                        println!("{}",str1);
                        for i in 0..vec.len(){
                            match &vec[i]{
                                data_structure::r_value::val(val)=>{
                                    match val{
                                        data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                        data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                        _=>(),
                                    }
                                },
                                data_structure::r_value::none=>(),
                                _=>(),
                            }
                        }
                    },
                    data_structure::path_pulse_percent_item::with_in_out_path(str1,str2,vec)=>{
                        println!("{} {} {}",str1,str2[0],str2[1]);
                        for i in 0..vec.len(){
                            match &vec[i]{
                                data_structure::r_value::val(val)=>{
                                    match val{
                                        data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                        data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                        _=>(),
                                    }
                                },
                                data_structure::r_value::none=>(),
                                _=>(),
                            }
                        }

                    },
                    _=>(),
                }
            },
            data_structure::deltype::absolute_increment(absolute_increment_item)=>{
                println!("{}",absolute_increment_item.keyword);
                for j in absolute_increment_item.del_defs.iter(){
                    match j{
                        data_structure::del_def::iopath(iopath_item)=>{
                            match iopath_item{
                                data_structure::iopath_item::no_retain(str1,port_spec,str2,vec)=>{
                                    println!("{} {}",str1,str2);

                                    for i in vec.into_iter(){
                                        for j in i.into_iter(){
                                            match j{
                                                data_structure::r_value::val(val)=>{
                                                    match val{
                                                        data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                        data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                        _=>(),
                                                    }
                                                },
                                                data_structure::r_value::none=>(),
                                                _=>(),
                                            }
                                        }
                                    }

                                }
                                data_structure::iopath_item::with_retain(str1,port_spec,str2,a,vec)=>(),
                                _=>(),
                            }
                        }
                        data_structure::del_def::cond(cond_item)=>(),
                        data_structure::del_def::condelse(condelse_item)=>(),
                        data_structure::del_def::port(port_item)=>{
                            println!("{} {}",port_item.keyword,port_item.port_instance);
                            for i in 0..port_item.delval_list.len(){
                                for j in 0..port_item.delval_list[i].len(){
                                    match &port_item.delval_list[i][j]{
                                        data_structure::r_value::val(val)=>{
                                            match val{
                                                data_structure::r_triple_number::r_triple(vec)=>println!("{:?}",vec),
                                                data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                                                _=>(),
                                            }
                                        },
                                        data_structure::r_value::none=>(),
                                        _=>(),
                                    }
                                }
                            }
                        },
                        data_structure::del_def::interconnect(interconnect_item)=>(),
                        data_structure::del_def::device(device_item)=>(),
                        _=>(),
                    }
                }
            },
            _=>(),
        }
    }
}

pub fn test_parse_sdf(){
    let tmp=parse_sdf::parse_sdf("C:/language_learn/rust/sdf_parse/tests/picorv32.sdf".to_string());
    println!("{}",tmp.keyword);
    for (key,val) in tmp.sdf_header{
        print!("{} ",key);
        match val{
            data_structure::sdf_header_val::str(string)=>println!("{}",string),
            data_structure::sdf_header_val::val(num_arr)=>{
                match num_arr{
                    data_structure::r_triple_number::r_triple(arr)=>println!("{:?}",arr),
                    data_structure::r_triple_number::r_number(num)=>println!("{}",num),
                    _=>(),
                }
            },
            data_structure::sdf_header_val::time_scale(time_scale)=>println!("{}{}",time_scale.number,time_scale.unit),
            _=>(),
        }
    }

    for i in tmp.cell.iter(){
        println!("{} {} {}",i.keyword,i.cell_type,i.cell_instance);
        for j in i.timing_spec.iter(){
            match j{
                data_structure::timing_spec::del_spec(del_spec)=>{
                    println!("del_spec:{}",del_spec.keyword);
                    print_del_spec(&del_spec);
                },
                data_structure::timing_spec::tc_spec(tc_spec)=>(),
                data_structure::timing_spec::te_spec(te_spec)=>(),
                _=>(),
            }
        }
    }
}





