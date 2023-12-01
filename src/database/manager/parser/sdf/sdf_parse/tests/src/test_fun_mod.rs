use pest::Parser;
use pest_derive::Parser;
use std::fs;
#[derive(Parser)]
#[grammar = "../src/sdf.pest"]
pub struct SDFParser;

pub fn rule_identifier() {
    let res = SDFParser::parse(Rule::identifier, "a \\ b");
    assert_eq!(res.is_ok(), true);

    let res = SDFParser::parse(Rule::identifier, "\\12a\\&");
    assert_eq!(res.is_err(), true);

    let res = SDFParser::parse(Rule::identifier, "meme[54]");
    assert_eq!(res.is_ok(), true);
}

pub fn rule_sdf_version_pair() {
    let mut result = SDFParser::parse(Rule::sdf_version_pair, "(SDFVERSION \"3.0\")")
        .unwrap() //vec<>
        .next() //Ok(Pairs)
        .unwrap() //Pairs
        .into_inner(); //一个个pair、pair
    let res=result.next();
    println!("{}", result.next().unwrap().as_str());
    //println!("{}",result.next().unwrap().as_str());
}

pub fn rule_design_name_pair() {
    let result = SDFParser::parse(Rule::design_name_pair, "(DESIGN \"picorv32\")")
        .unwrap() //vec<>
        .next() //Ok(Pairs)
        .unwrap() //Pairs
        .into_inner() //一个个pair、pair
        .next()
        .unwrap();
    println!("{}", result.as_str());
}

pub fn rule_date_pair() {
    let result = SDFParser::parse(Rule::date_pair, "(DATE \"Fri Mar 24 09:04:43 2023\")")
        .unwrap() //vec<>
        .next() //Ok(Pairs)
        .unwrap() //Pairs
        .into_inner() //一个个pair、pair
        .next()
        .unwrap();
    println!("{}", result.as_str());
}

pub fn rule_vendor_pair() {
    let result = SDFParser::parse(Rule::vendor_pair, "(VENDOR \"Parallax\")")
        .unwrap() //vec<>
        .next() //Ok(Pairs)
        .unwrap() //Pairs
        .into_inner() //一个个pair、pair
        .next()
        .unwrap();
    println!("{}", result.as_str());
}
pub fn rule_program_name_pair() {
    let result = SDFParser::parse(Rule::program_name_pair, "(PROGRAM \"STA\")")
        .unwrap() //vec<>
        .next() //Ok(Pairs)
        .unwrap() //Pairs
        .into_inner() //一个个pair、pair
        .next()
        .unwrap();
    println!("{}", result.as_str());
}
pub fn rule_version_pair() {
    let result = SDFParser::parse(Rule::version_pair, "(VERSION \"2.4.0\")")
        .unwrap() //vec<>
        .next() //Ok(Pairs)
        .unwrap() //Pairs
        .into_inner() //一个个pair、pair
        .next()
        .unwrap();
    println!("{}", result.as_str());
}

pub fn rule_divider_pair() {
    let result = SDFParser::parse(Rule::divider_pair, "(DIVIDER .)")
        .unwrap() //vec<>
        .next() //Ok(Pairs)
        .unwrap() //Pairs
        .into_inner() //一个个pair、pair
        .next()
        .unwrap();
    println!("{}", result.as_str());
}
pub fn rule_voltage_pair() {
    let result = SDFParser::parse(Rule::voltage_pair, "(VOLTAGE 1.800::1.800)");
    assert_eq!(result.is_ok(),true);
    println!("{}",result.unwrap().as_str());

    let result = SDFParser::parse(Rule::voltage_pair, "(VOLTAGE -1.800)");
    assert_eq!(result.is_ok(),true);
    println!("{}",result.unwrap().as_str());
}

pub fn rule_process_pair() {
    let result = SDFParser::parse(Rule::process_pair, "(PROCESS \"1.000::1.000\")")
        .unwrap() //vec<>
        .next() //Ok(Pairs)
        .unwrap() //Pairs
        .into_inner() //一个个pair、pair
        .next()
        .unwrap();
    println!("{}", result.as_str());
}

pub fn rule_temperature_pair() {
    let result = SDFParser::parse(Rule::temperature_pair, "(TEMPERATURE 25.000::25.000)");
    assert_eq!(result.is_ok(),true);
    //println!("{}",result.unwrap().as_str());
    for i in result.unwrap().next().unwrap().into_inner(){
        match i.as_rule(){
            Rule::sdf_header_keyword=>println!("key:{}",i.as_str()),
            Rule::rtriple=>println!("rtriple:{}",i.as_str()),
            Rule::rnumber=>println!("rnumber:{}",i.as_str()),
            _=>(),
        }
    }

    let result = SDFParser::parse(Rule::temperature_pair, "(TEMPERATURE -255)");
    assert_eq!(result.is_ok(),true);
    for i in result.unwrap().next().unwrap().into_inner(){
        match i.as_rule(){
            Rule::sdf_header_keyword=>println!("key:{}",i.as_str()),
            Rule::rtriple=>println!("rtriple:{}",i.as_str()),
            Rule::rnumber=>println!("rnumber:{}",i.as_str()),
            _=>(),
        }
    }
}

pub fn rule_time_scale_pair() {
    let result = SDFParser::parse(Rule::time_scale_pair, "(TIMESCALE 1ns)")
        .expect("parse failing")
        .next()
        .unwrap();
    //let data = fun::extract_time_scale(&result);
    //fun::print_time_scale(&data);

    let result = SDFParser::parse(Rule::time_scale_pair, "(TIMESCALE 100 ps)")
        .expect("parse failing")
        .next()
        .unwrap();
    //let data = fun::extract_time_scale(&result);
    //fun::print_time_scale(&data);
}

pub fn rule_number() {
    let res = SDFParser::parse(Rule::number, "2e-3");
    assert_eq!(res.is_ok(), true);
    let res = SDFParser::parse(Rule::number, "-23");
    assert_eq!(res.is_err(), true);
}

pub fn rule_temperature() {
    let mut res = SDFParser::parse(Rule::temperature, "100::100")
        .unwrap()
        .next()
        .unwrap();
    println!("{}", res.as_str());
    match res.as_rule() {
        Rule::rtriple => println!("3243"),
        Rule::rnumber => println!("sdgfd"),
        _ => (),
    }
    //let vec = fun::extract_triple_number(&res);
    //fun::print_triple_number(&vec);
}

pub fn rule_sdf_header() {
    let unparsed_text =
        std::fs::read_to_string("test_sdf_header.txt").expect("read file unsuccessfully");
    println!("{}", unparsed_text);
    let parsed_text = SDFParser::parse(Rule::sdf_header, &unparsed_text);
    assert_eq!(parsed_text.is_ok(), true);
}

pub fn rule_celltype() {
    let res = SDFParser::parse(Rule::celltype, "(CELLTYPE \"DFF\")");
    assert_eq!(res.is_ok(), true);
    println!(
        //"{}",
        //fun::parse_qstring_with_quotes(&(res.unwrap().next().unwrap()))
    );
}

pub fn rule_cell_instance() {
    let mut res = SDFParser::parse(Rule::cell_instance, "(INSTANCE *)")
        .unwrap()
        .next()
        .unwrap();
    //assert_eq!(res.is_ok(),true);
    println!("{}", res.as_str());
    for i in res.into_inner() {
        for j in i.into_inner() {
            println!("{}", j.as_str());
        }
    }

    let res = SDFParser::parse(Rule::cell_instance, "(INSTANCE a1.b1.c1)");
    assert_eq!(res.is_ok(), true);

    let res = SDFParser::parse(Rule::cell_instance, "(INSTANCE)");
    assert_eq!(res.is_ok(), true);
}

pub fn rule_input_output_path() {
    let res = SDFParser::parse(Rule::input_output_path, "a b");
    assert_eq!(res.is_ok(), true);
    for i in res.unwrap().next().unwrap().into_inner() {
        for j in i.into_inner() {
            println!("{}", j.as_str());
        }
    }

    let res = SDFParser::parse(Rule::input_output_path, "cyb mem[2]");
    assert_eq!(res.is_ok(), true);
    for i in res.unwrap().next().unwrap().into_inner() {
        for j in i.into_inner() {
            println!("{}", j.as_str());
        }
    }

    let res = SDFParser::parse(Rule::input_output_path, "mem[2:4] mem[2]");
    assert_eq!(res.is_ok(), true);
    for i in res.unwrap().next().unwrap().into_inner() {
        for j in i.into_inner() {
            println!("{}", j.as_str());
        }
    }

    let res = SDFParser::parse(Rule::input_output_path, "cyb ctyb.a.a.v.d.mem[2]");
    assert_eq!(res.is_ok(), true);
    for i in res.unwrap().next().unwrap().into_inner() {
        println!("{}", i.as_str());
    }

    let res=SDFParser::parse(Rule::input_output_path,"U1997.X U2620.S");
    assert_eq!(res.is_ok(),true);
    for i in res.unwrap().next().unwrap().into_inner(){
        println!("{}",i.as_str());
    }

    let res = SDFParser::parse(Rule::input_output_path, "ma[15]] res[0:15]]");
    assert_eq!(res.is_err(), true);
    //println!("{}",res.unwrap().next().unwrap().as_str());
}

pub fn rule_path_pulse_percent() {
    let mut res = SDFParser::parse(Rule::path_pulse_percent, "(PATHPULSE a y (13) (24))")
        .expect("parse unsuccessfully");
    //assert_eq!(res.is_ok(),true);
    for i in res.next().unwrap().into_inner() {
        match i.as_rule() {
            Rule::path_pulse_percent => println!("{}", i.as_str()),
            Rule::input_output_path => {
                for j in i.into_inner() {
                    println!("{}", j.as_str());
                }
            }
            Rule::value => {
                for j in i.into_inner() {
                    match j.as_rule() {
                        Rule::number => println!("{}", j.as_str().parse::<f64>().unwrap()),
                        Rule::triple => println!("{}", j.as_str()),
                        _ => (),
                    }
                }
            }
            _ => (),
        }
    }
    let res = SDFParser::parse(Rule::path_pulse_percent, "(PATHPULSE i1 o1 (13) (21))");
    assert_eq!(res.is_ok(), true);

    let res = SDFParser::parse(Rule::path_pulse_percent, "(PATHPULSEPERCENT a y (25) (35))");
    assert_eq!(res.is_ok(), true);
}

pub fn rule_scalar_port() {
    let parsed_text = SDFParser::parse(Rule::scalar_port, "mem$[12]");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    //let text=SDFParser::parse(Rule::scalar_port,"abc[2:5]");
    //assert_eq!(text.is_ok(),true);

    let parsed_text = SDFParser::parse(Rule::scalar_port, "mem[17]");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_bus_port() {
    let parsed_text = SDFParser::parse(Rule::bus_port, "mem[0:16]");
    assert_eq!(parsed_text.is_ok(), true);
    for i in parsed_text.unwrap().next().unwrap().into_inner() {
        match i.as_rule() {
            Rule::dnumber => println!("1{}", i.as_str().parse::<f64>().unwrap()),
            _ => println!("55{}", i.as_str()),
        }
    }
}

pub fn rule_port_instance() {
    let parsed_text = SDFParser::parse(Rule::port_instance, "mem[0:17]");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::port_instance, "abc.we.dgf/we");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::port_instance, "wete\\");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::port_instance, "cyb.a.mem[2]");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::port_instance, "cyb.a.mem[2].a.mem[2:3]");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::port_instance, "cyb");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_port_spec() {
    let parsed_text = SDFParser::parse(Rule::port_spec, "mem[0:17]]");
    assert_eq!(parsed_text.is_ok(), true);
    for i in parsed_text.unwrap().next().unwrap().into_inner() {
        for j in i.into_inner() {
            println!("{}", j.as_str())
        }
    }

    let parsed_text = SDFParser::parse(Rule::port_spec, "(posedge mem[18])");
    assert_eq!(parsed_text.is_ok(), true);
    for i in parsed_text.unwrap().next().unwrap().into_inner() {
        for j in i.into_inner() {
            println!("{}", j.as_str())
        }
    }

    let parsed_text = SDFParser::parse(Rule::port_spec, "(negedge mem[0:18])");
    assert_eq!(parsed_text.is_ok(), true);
    for i in parsed_text.unwrap().next().unwrap().into_inner() {
        for j in i.into_inner() {
            println!("{}", j.as_str())
        }
    }

    let parsed_text = SDFParser::parse(Rule::port_spec, "(z1 mem[12])");
    assert_eq!(parsed_text.is_ok(), true);
    for i in parsed_text.unwrap().next().unwrap().into_inner() {
        for j in i.into_inner() {
            println!("{}", j.as_str())
        }
    }
}

pub fn rule_iopath_part1() {
    let parsed_text = SDFParser::parse(Rule::iopath_part1, "IOPATH (negedge mem[12]) src[0:18]");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::iopath_part1, "IOPATH (negedge) src[0:18]");
    assert_eq!(parsed_text.is_err(), true);
}

pub fn rule_rvalue() {
    let parsed_text = SDFParser::parse(Rule::rvalue, "()");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::rvalue, "(12)");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::rvalue, "(12:23:5)");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_delval() {
    let parsed_text = SDFParser::parse(Rule::delval, "(12)");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::delval, "(12:23:5)");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::delval, "()");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::delval, "((12) (12:4:34))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::delval, "((12:23:5) (12) ())");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_delval_list() {
    let parsed_text = SDFParser::parse(Rule::delval_list, "((12) (12:4:34)) (12)");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(
        Rule::delval_list,
        "((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12)",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_iopath_part2_part() {
    let parsed_text = SDFParser::parse(Rule::iopath_part2_part, "(RETAIN (12))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::iopath_part2_part, "(RETAIN ((12) ()))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::iopath_part2_part, "(RETAIN ((12) (12:4:45)))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::iopath_part2_part, "(RETAIN ((12) (12:4:45)) ())");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_iopath_part2() {
    let parsed_text = SDFParser::parse(
        Rule::iopath_part2,
        "(RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_iopath_item() {
    let parsed_text = SDFParser::parse(
        Rule::iopath_item,
        "(IOPATH (negedge mem[12]) src[0:18] 
        (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45))",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_cond_item() {
    let parsed_text=SDFParser::parse(Rule::cond_item,
        "(COND \"asfd\" ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text=SDFParser::parse(Rule::cond_item,
        "(COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_condelse_item() {
    let parsed_text=SDFParser::parse(Rule::condelse_item,
        "(CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text=SDFParser::parse(Rule::condelse_item,"(CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_port_item() {
    let parsed_text = SDFParser::parse(
        Rule::port_item,
        "(PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(
        Rule::port_item,
        "(PORT nen[2:23]] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))",
    );
    assert_eq!(parsed_text.is_err(), true);
}

pub fn rule_interconnect_item() {
    let parsed_text=SDFParser::parse(Rule::interconnect_item,
        "(INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text=SDFParser::parse(Rule::interconnect_item,
        "(INTERCONNECT nen[2:23] (posedge mem[2]) ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))");
    assert_eq!(parsed_text.is_err(), true);

    test_right_exam(&Rule::interconnect_item,"(INTERCONNECT clk U7768.CLK (9.139:9.139:9.139) (4.401:4.401:4.401))");
    test_right_exam(&Rule::interconnect_item,"(INTERCONNECT clk U2620.S (0.000:0.000:0.000))");

    test_right_exam(&Rule::delval_list,"(0.000:0.000:0.000)");
    test_right_exam(&Rule::delval,"(0.000:0.000:0.000)");
    test_right_exam(&Rule::rnumber,"0.000");
    test_right_exam(&Rule::port_instance,"U1997.X U2610.S");
    test_right_exam(&Rule::input_output_path,"U1997.X U2620.S (0.000:0.000:0.000)");
    
    test_right_exam(&Rule::interconnect_item,"(INTERCONNECT U1997.X U2620.S (0.000:0.000:0.000))");
}

pub fn rule_device_item() {
    let parsed_text = SDFParser::parse(
        Rule::device_item,
        "(DEVICE ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(
        Rule::device_item,
        "(DEVICE mem[2:4] (12) ()) () (12) (12::) () () () () (34.5) (12))",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(
        Rule::device_item,
        "(DEVICE m[23] mem[4:6] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))",
    );
    assert_eq!(parsed_text.is_err(), true);
}

pub fn rule_del_def() {
    let parsed_text=SDFParser::parse(Rule::del_def,
        "(IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text=SDFParser::parse(Rule::del_def,
        "(COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text=SDFParser::parse(Rule::del_def,
        "(CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(
        Rule::del_def,
        "(PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text=SDFParser::parse(Rule::del_def,
        "(INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(
        Rule::del_def,
        "(DEVICE mem[2:4] (12) () () (12) (12::) () () () (() (34.5) (12)))",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_absolute_increment() {
    let parsed_text=SDFParser::parse(Rule::absolute_increment,
        "(ABSOLUTE 
            (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)) 
            (COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
            (COND \"asfd\" ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
            (CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
            (PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
            (INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
            (DEVICE mem[2:4] (12) () () (12) (12::) () () () () (34.5) (12))
        )");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text=SDFParser::parse(Rule::absolute_increment,
        "(INCREMENT
            (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45))
            (COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
            (CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
            (PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
            (INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
            (DEVICE mem[2:4] (12) () () (12) (12::) () () () () (34.5) (12)))");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_deltype() {
    let text=SDFParser::parse(Rule::deltype,"(INCREMENT
        (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45))
        (COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
        (CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
        (PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
        (INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
        (DEVICE mem[2:4] (12) () () (12) (12::) () () () () (34.5) (12))
    )");
    assert_eq!(text.is_ok(), true);
    println!("{}", text.unwrap().as_str());

    let text = SDFParser::parse(Rule::deltype, "(PATHPULSEPERCENT a y (25) (35))");
    assert_eq!(text.is_ok(), true);
    println!("{}", text.unwrap().as_str());

    test_right_exam(
        &Rule::deltype,
        "(ABSOLUTE
        (INTERCONNECT clk U7768.CLK (9.139:9.139:9.139) (4.401:4.401:4.401))
        (INTERCONNECT clk U7769.CLK (9.139:9.139:9.139) (4.401:4.401:4.401))
        (INTERCONNECT U1997.X U2392.S0 (0.000:0.000:0.000))
        (INTERCONNECT U1997.X U2620.S (0.000:0.000:0.000))
    (INTERCONNECT U2.X pcpi_rs2[5] (0.000:0.000:0.000))
    (INTERCONNECT U9395.LO eoi[31] (0.000:0.000:0.000))
    (INTERCONNECT U9396.LO mem_addr[0] (0.000:0.000:0.000))
    (INTERCONNECT U9397.LO mem_addr[1] (0.000:0.000:0.000))
    (INTERCONNECT U9398.LO mem_la_addr[0] (0.000:0.000:0.000))
    (INTERCONNECT U9399.LO mem_la_addr[1] (0.000:0.000:0.000))
    (INTERCONNECT U94.X U8690.D (0.000:0.000:0.000))
    (INTERCONNECT U940.X U941.C (0.000:0.000:0.000))
    (INTERCONNECT U9409.LO pcpi_insn[9] (0.000:0.000:0.000))
    (INTERCONNECT U9395.LO eoi[31] (0.000:0.000:0.000))
    (INTERCONNECT U9396.LO mem_addr[0] (0.000:0.000:0.000))
    (INTERCONNECT U9397.LO mem_addr[1] (0.000:0.000:0.000))
    (INTERCONNECT U9398.LO mem_la_addr[0] (0.000:0.000:0.000))
    (INTERCONNECT U9399.LO mem_la_addr[1] (0.000:0.000:0.000))
    (INTERCONNECT U94.X U8690.D (0.000:0.000:0.000))
    (INTERCONNECT U940.X U941.C (0.000:0.000:0.000))
    (INTERCONNECT U9409.LO pcpi_insn[9] (0.000:0.000:0.000))
    (INTERCONNECT U941.X U942.S (0.000:0.000:0.000))
    (INTERCONNECT U9410.LO pcpi_insn[10] (0.000:0.000:0.000))
    (INTERCONNECT U942.X U945.C1 (0.000:0.000:0.000))
    (INTERCONNECT U9429.LO pcpi_insn[29] (0.000:0.000:0.000))
    (INTERCONNECT U943.X U944.S (0.000:0.000:0.000))
    (INTERCONNECT U9430.LO pcpi_insn[30] (0.000:0.000:0.000))
    (INTERCONNECT U9431.LO pcpi_insn[31] (0.000:0.000:0.000))
    (INTERCONNECT U9432.LO pcpi_valid (0.000:0.000:0.000))
    (INTERCONNECT U9439.LO trace_data[6] (0.000:0.000:0.000))
    (INTERCONNECT U944.X U945.D1 (0.000:0.000:0.000))
    (INTERCONNECT U9440.LO trace_data[7] (0.000:0.000:0.000))
    (INTERCONNECT U9441.LO trace_data[8] (0.000:0.000:0.000))
    (INTERCONNECT U9449.LO trace_data[16] (0.000:0.000:0.000))
    (INTERCONNECT U945.Y U1492.A (0.000:0.000:0.000))
    (INTERCONNECT U945.Y U5911.A2 (0.000:0.000:0.000))
    (INTERCONNECT U945.Y U946.A (0.000:0.000:0.000))
    (INTERCONNECT U9450.LO trace_data[17] (0.000:0.000:0.000))
    (INTERCONNECT U9459.LO trace_data[26] (0.000:0.000:0.000))
    (INTERCONNECT U946.Y U5136.B2 (0.000:0.000:0.000))
    (INTERCONNECT U946.Y U957.A2 (0.000:0.000:0.000))
    (INTERCONNECT U9460.LO trace_data[27] (0.000:0.000:0.000))
    (INTERCONNECT U9468.LO trace_data[35] (0.000:0.000:0.000))
    (INTERCONNECT U9469.LO trace_valid (0.000:0.000:0.000))
    (INTERCONNECT U947.X U950.B (0.000:0.000:0.000))
    (INTERCONNECT U95.X U4489.A (0.000:0.000:0.000))
    (INTERCONNECT U95.X U96.A (0.000:0.000:0.000))
    (INTERCONNECT U950.Y U1053.A (0.000:0.000:0.000))
    (INTERCONNECT U953.X U956.B1 (0.000:0.000:0.000))
    (INTERCONNECT U954.X U1012.B1 (0.000:0.000:0.000))
    (INTERCONNECT U954.X U1031.B1 (0.000:0.000:0.000))
    (INTERCONNECT U954.X U1108.B1 (0.000:0.000:0.000))
    (INTERCONNECT U954.X U1119.A (0.000:0.000:0.000))
    (INTERCONNECT U954.X U1202.A1 (0.000:0.000:0.000))
    (INTERCONNECT U954.X U955.A (0.000:0.000:0.000))
    (INTERCONNECT U955.X U1034.B1 (0.000:0.000:0.000))
    (INTERCONNECT U955.X U1075.B1 (0.000:0.000:0.000))
    (INTERCONNECT U969.X U1045.A (0.000:0.000:0.000))
    (INTERCONNECT U969.X U1134.B1 (0.000:0.000:0.000))
        )"
    );
}

pub fn rule_del_spec() {
    let text=SDFParser::parse(Rule::del_spec,"(DELAY 
        (INCREMENT
            (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45))
            (COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
            (CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
            (PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
            (INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
            (DEVICE mem[2:4] (12) () () (12) (12::) () () () () (34.5) (12))
        )
        (PATHPULSEPERCENT a y (25) (35)))");
    assert_eq!(text.is_ok(), true);
    println!("{}", text.unwrap().as_str());
}

pub fn rule_simple_expression_type1() {
    let parsed_text = SDFParser::parse(Rule::simple_expression_type1, "(mem[12])");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    test_right_exam(&Rule::simple_expression_type1, "(!(!(~mem[2:3])))");
    test_right_exam(&Rule::simple_expression_type1, "(!(!(~1'B0)))");
}

pub fn rule_simple_expression_type2() {
    let parsed_text = SDFParser::parse(Rule::simple_expression_type2, "~&(mem[12:45])");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    test_right_exam(&Rule::simple_expression_type2, "~&((&1'b0))");
    test_right_exam(&Rule::simple_expression_type2, "!(abc[2:4])");
}

pub fn rule_simple_expression_type3() {
    let parsed_text = SDFParser::parse(Rule::simple_expression_type3, "mem[23:90]");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    test_right_exam(&Rule::simple_expression_type3, "mem[2]");
    test_right_exam(&Rule::simple_expression_type3, "mem");
}

pub fn rule_simple_expression_type4() {
    let parsed_text = SDFParser::parse(Rule::simple_expression_type4, "+mem");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    test_right_exam(&Rule::simple_expression_type4, "~&mem[2]");
    test_right_exam(&Rule::simple_expression_type4, "&mem");
}

pub fn rule_simple_expression_type5() {
    let parsed_text = SDFParser::parse(Rule::simple_expression_type5, "1'B0");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    test_right_exam(&Rule::simple_expression_type5, "1'b0");
    test_right_exam(&Rule::simple_expression_type5, "'B0");
}

pub fn rule_simple_expression_type6() {
    let parsed_text = SDFParser::parse(Rule::simple_expression_type6, "+b1");
    assert_eq!(parsed_text.is_err(), true);

    test_right_exam(&Rule::simple_expression_type6, "~&1'b0");
    test_right_exam(&Rule::simple_expression_type6, "&'b1");
}

pub fn rule_simple_expression_type7() {
    let parsed_text =
        SDFParser::parse(Rule::simple_expression_type7, "mem[12]? mem[23]:+mem[0:12]");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    test_right_exam(&Rule::simple_expression_type7, "~&mem[2]? ~&1'b0:mem[2:5]");
    test_right_exam(
        &Rule::simple_expression_type7,
        "~&mem[2]? ~&1'b0:(mem[2:5])",
    );
    test_right_exam(&Rule::simple_expression_type7, "mem[2:5]? cyb:up8848");
    test_right_exam(
        &Rule::simple_expression_type7,
        "mem[2]? cunsg:(mem[2:5]? cyb:up8848)",
    );

    let parsed_text = SDFParser::parse(
        Rule::simple_expression_type7,
        "mem[2]? cunsg:(mem[2:5]? cyb:up8848)",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    test_right_exam(
        &Rule::simple_expression_type7,
        "(~&mem[2]? ~&1'b0:mem[2:5])? mem[2]:1'b0",
    );

    test_right_exam(
        &Rule::simple_expression_type7,
        "(~&mem[2]? ~&1'b0:mem[2:5])? (mem[2]? cunsg:(mem[2:5]? cyb:up8848)):1'b0",
    );
    test_right_exam(
        &Rule::simple_expression_type7,
        "(~&mem[2]? ~&1'b0:mem[2:5])? mem[2]:1'b0",
    );
    test_wrong_exam(&Rule::simple_expression_type7, "&mem");
}

pub fn rule_simple_expression_type8() {
    let parsed_text = SDFParser::parse(Rule::simple_expression_type8, "{b1,mem[2]}");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::simple_expression_type8, "{b1}");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_simple_expression_type9() {
    let parsed_text = SDFParser::parse(Rule::simple_expression_type9, "{&b0{b1,mem[2]}}");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::simple_expression_type9, "{&b0{b1}}");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_simple_expression() {
    let parsed_text = SDFParser::parse(
        Rule::simple_expression,
        "{&b0{({mem[5]{mem[4:8],&mem[8]}}),mem[2]}}",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_concat_expression() {
    let parsed_text = SDFParser::parse(Rule::concat_expression, ",(mem[12])");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_conditional_port_expr_type1() {
    test_right_exam(
        &Rule::conditional_port_expr_type1,
        "{&b0{({mem[5]{mem[4:8],&mem[8]}}),mem[2]}}",
    );
}

pub fn rule_conditional_port_expr_type2() {
    test_right_exam(&Rule::conditional_port_expr_type2, "(~&(mem[3:5]))");
    test_right_exam(
        &Rule::conditional_port_expr_type2,
        "(~&(cyb? mem[2]:(mem? cyb:up77)))",
    );
    test_right_exam(&Rule::conditional_port_expr_type2, "(mem[2])");
    test_right_exam(
        &Rule::conditional_port_expr_type2,
        "((~&(cyb? mem[2]:(mem? cyb:up77)))+(~&(cyb? mem[2]:(mem? cyb:up77))))",
    );
}

pub fn rule_conditional_port_expr_type3() {
    test_right_exam(
        &Rule::conditional_port_expr_type3,
        "~&(((~&(cyb? mem[2]:(mem? cyb:up77)))+(~&(cyb? mem[2]:(mem? cyb:up77)))))",
    );
    test_right_exam(&Rule::conditional_port_expr_type3, "~&(mem[2])");
    test_right_exam(
        &Rule::conditional_port_expr_type3,
        "~&({&b0{({mem[5]{mem[4:8],&mem[8]}}),mem[2]}})",
    );
}

pub fn rule_conditional_port_expr_type4() {
    test_right_exam(
        &Rule::conditional_port_expr_type4,
        "~&(((~&(cyb? mem[2]:(mem? cyb:up77)))+(~&(cyb? mem[2]:(mem? cyb:up77)))))+
        {&b0{({mem[5]{mem[4:8],&mem[8]}}),mem[2]}}",
    );
}

pub fn rule_conditional_port_expr() {
    test_right_exam(
        &Rule::conditional_port_expr,
        "{&b0{({mem[5]{mem[4:8],&mem[8]}}),mem[2]}}",
    );
    test_right_exam(&Rule::conditional_port_expr, "(mem[2])");
    test_right_exam(
        &Rule::conditional_port_expr,
        "~&(((~&(cyb? mem[2]:(mem? cyb:up77)))+(~&(cyb? mem[2]:(mem? cyb:up77)))))",
    );
    test_right_exam(
        &Rule::conditional_port_expr,
        "~&(((~&(cyb? mem[2]:(mem? cyb:up77)))+(~&(cyb? mem[2]:(mem? cyb:up77)))))+
        {&b0{({mem[5]{mem[4:8],&mem[8]}}),mem[2]}}",
    );
}

fn test_right_exam(rule: &Rule, text: &str) {
    let parsed_text = SDFParser::parse(*rule, text);
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

fn test_wrong_exam(rule: &Rule, text: &str) {
    let parsed_text = SDFParser::parse(*rule, text);
    assert_eq!(parsed_text.is_err(), true);
}

pub fn rule_setup_hold_recovery_removal_item() {
    test_right_exam(
        &Rule::setup_hold_recovery_removal_item,
        "(SETUP (01 mem[2:3]) mem[3] ())",
    );
    test_right_exam(
        &Rule::setup_hold_recovery_removal_item,
        "(HOLD (01 mem[2:3]) mem[3] (12))",
    );
    test_right_exam(
        &Rule::setup_hold_recovery_removal_item,
        "(RECOVERY (01 mem[2:3]) mem[3] (12::45))",
    );
    test_right_exam(
        &Rule::setup_hold_recovery_removal_item,
        "(REMOVAL (01 mem[2]) mem[3] (12:3:34))",
    );
    test_right_exam(
        &Rule::setup_hold_recovery_removal_item,
        "(SETUP mem[2] mem[3] (12:3:34))",
    );
    test_right_exam(
        &Rule::setup_hold_recovery_removal_item,
        "(REMOVAL (01 mem[2]) mem[3] (12:3:34))",
    );

    test_wrong_exam(
        &Rule::setup_hold_recovery_removal_item,
        "(REMOVAL (01 mem[2]) (12:3:34))",
    );

    //test_right_exam(&Rule::setup_hold_recovery_removal_item,"(REMOVAL (COND fushgusdm mem[12] (negedge nen[23:6])) mem[3] (12:3:34))");
    //test_right_exam(&Rule::setup_hold_recovery_removal_item,"(REMOVAL (COND mem[12] (negedge nen[23:6])) mem[3] (12:3:34))");
}

pub fn rule_setuphold_item1_recrem_item1_nochange_item() {
    test_right_exam(
        &Rule::setuphold_item1_recrem_item1_nochange_item,
        "(SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())",
    );
    test_right_exam(
        &Rule::setuphold_item1_recrem_item1_nochange_item,
        "(SETUPHOLD (01 mem[2:2]) mem[3] (12:3:34) (12::3))",
    );
    test_right_exam(
        &Rule::setuphold_item1_recrem_item1_nochange_item,
        "(RECREM (01 mem[2]) mem[3] (12:3:34) (213))",
    );
    test_right_exam(
        &Rule::setuphold_item1_recrem_item1_nochange_item,
        "(RECREM (01 mem[2]) mem[3] (12) (14))",
    );

    test_wrong_exam(
        &Rule::setuphold_item1_recrem_item1_nochange_item,
        "(RECREM (01 mem[2]) mem[3] (12))",
    );
    test_wrong_exam(
        &Rule::setuphold_item1_recrem_item1_nochange_item,
        "(RECREM (01 mem[2]) (12::) (12))",
    );
}

//test failing
pub fn rule_setuphold_item2_recrem_item2_item() {
    test_right_exam(
        &Rule::setuphold_item2_recrem_item2_item,
        "(SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())",
    );
    test_right_exam(
        &Rule::setuphold_item2_recrem_item2_item,
        "(RECREM (01 mem[2:2]) mem[3] (12:3:34) (12::3))",
    );

    let parsed_text = SDFParser::parse(
        Rule::setuphold_item2_recrem_item2_item,
        "(RECREM (01 mem[2]) mem[3] (12:3:34) (213) (SCOND \"wet\" mem[3] cyb))",
    );
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    //test_right_exam(&Rule::setuphold_item2_recrem_item2_item,"(SETUPHOLD (01 mem[2]) mem[3] (12) (14) (SCOND \"wet\" mem[3] cyb) (CCOND mem[3] cyb))");
}

pub fn rule_skew_item() {
    test_right_exam(
        &Rule::skew_item,
        "(SKEW (COND \"cybup8848\" !mem[5] cyb mem[3:5]) mem[2] (-2:3:))",
    );
    test_right_exam(
        &Rule::skew_item,
        "(SKEW (COND !mem[5] cyb mem[3:5]) mem[2] (-2:3:))",
    );
    test_wrong_exam(&Rule::skew_item, "(SKEW mem[2] ())");
}

pub fn rule_width_period_item() {
    test_right_exam(
        &Rule::width_period_item,
        "(WIDTH (COND \"cybup8848\" !mem[5] cyb mem[3:5]) (2:3:))",
    );
    test_wrong_exam(
        &Rule::width_period_item,
        "(WIDTH (COND \"cybup8848\" !mem[5] cyb mem[3:5]) (-2:3:))",
    );
    test_right_exam(
        &Rule::width_period_item,
        "(WIDTH (COND !mem[5] cyb mem[3:5]) (2:3:))",
    );
    test_right_exam(&Rule::width_period_item, "(PERIOD mem[2] (2:3:))");
    test_right_exam(&Rule::width_period_item, "(WIDTH mem[2:3] ())");
    test_right_exam(&Rule::width_period_item, "(PERIOD mem ())");
}

pub fn rule_sccond() {
    let parsed_text = SDFParser::parse(Rule::sccond, "(SCOND \"cyb\" !mem[2] up8848)");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());

    let parsed_text = SDFParser::parse(Rule::sccond, "(SCOND mem[2] up8848)");
    assert_eq!(parsed_text.is_ok(), true);
    println!("{}", parsed_text.unwrap().as_str());
}

pub fn rule_port_tchk_type1() {
    test_right_exam(&Rule::port_tchk_type1, "mem[2:3]");
    test_right_exam(&Rule::port_tchk_type1, "mem[3]");
    test_right_exam(&Rule::port_tchk_type1, "cyb");
    test_right_exam(&Rule::port_tchk_type1, "(10 mem[2])");
}

pub fn rule_port_tchk_type2() {
    test_right_exam(
        &Rule::port_tchk_type2,
        "(COND \"wer\" ~mem[2] cyb (posedge mem[2]))",
    );
    test_right_exam(&Rule::port_tchk_type2, "(COND \"wer\" mem[2] cyb mem)");

    let text = SDFParser::parse(Rule::port_tchk_type2, "(COND !mem[2] cyb (posedge mem[2]))");
    assert_eq!(text.is_ok(), true);
    println!("{}", text.unwrap().as_str());

    let text = SDFParser::parse(
        Rule::port_tchk_type2,
        "(COND \"wer\" ~mem[2] cyb (posedge mem[2]))",
    );
    assert_eq!(text.is_ok(), true);
    println!("{}", text.unwrap().as_str());
}

pub fn rule_port_tchk() {
    let text = SDFParser::parse(Rule::port_tchk, "(COND !mem[2] cyb (posedge mem[2]))");
    assert_eq!(text.is_ok(), true);
    println!("{}", text.unwrap().as_str());

    let text = SDFParser::parse(Rule::port_tchk, "(10 mem[2])");
    assert_eq!(text.is_ok(), true);
    println!("{}", text.unwrap().as_str());
}

pub fn rule_timing_check_condition_type1() {
    test_right_exam(&Rule::timing_check_condition_type1, "mem mem");
    test_right_exam(&Rule::timing_check_condition_type1, "mem[2] mem");
}

pub fn rule_timing_check_condition_type2() {
    test_right_exam(&Rule::timing_check_condition_type2, "!mem[2] mem");
    test_right_exam(&Rule::timing_check_condition_type2, "~mem[2] cyb");
    test_wrong_exam(&Rule::timing_check_condition_type2, "~!mem[2] cyb");
    test_wrong_exam(&Rule::timing_check_condition_type2, "~!mem[2]");
    test_wrong_exam(&Rule::timing_check_condition_type2, ">mem[2] c");
}

pub fn rule_timing_check_condition_type3() {
    test_right_exam(&Rule::timing_check_condition_type3, "mem[2] mem !== 0");

    let text = SDFParser::parse(Rule::timing_check_condition_type3, "cyb cyb===0");
    assert_eq!(text.is_ok(), true);
    println!("{}", text.unwrap().as_str());

    test_right_exam(&Rule::timing_check_condition_type3, "mem cyb!=='B0");
    test_right_exam(&Rule::timing_check_condition_type3, "mem cyb==1'B0");
    test_right_exam(&Rule::timing_check_condition_type3, "mem cyb!=1'b1");
}

pub fn rule_timing_check_condition() {
    test_right_exam(&Rule::timing_check_condition, "mem mem");

    test_right_exam(&Rule::timing_check_condition, "!mem[2] mem");
    test_right_exam(&Rule::timing_check_condition, "~mem[2] cyb");
    test_wrong_exam(&Rule::timing_check_condition, "~!mem[2] cyb");

    test_right_exam(&Rule::timing_check_condition, "mem[2] cyb===0");
    test_right_exam(&Rule::timing_check_condition, "mem[2] sd!=='B0");
}

pub fn rule_scalar_node() {
    let text = SDFParser::parse(Rule::scalar_node, "mem[2] abc");
    assert_eq!(text.is_ok(), true);
    for i in text.unwrap().next().unwrap().into_inner() {
        match i.as_rule() {
            Rule::scalar_port => println!("scalar port:{}", i.as_str()),
            Rule::scalar_net => println!("scalar net:{}", i.as_str()),
            _ => (),
        }
    }
}

pub fn rule_scalar_net() {
    let text = SDFParser::parse(Rule::scalar_net, "abc");
    assert_eq!(text.is_ok(), true);
    let text = SDFParser::parse(Rule::scalar_net, "_abc");
    assert_eq!(text.is_ok(), true);
    let text = SDFParser::parse(Rule::scalar_net, "a$bc");
    assert_eq!(text.is_ok(), true);

    let text = SDFParser::parse(Rule::scalar_net, "a$bc cdf");
    assert_eq!(text.is_ok(), true);
    println!("{}", text.unwrap().as_str());
}

pub fn rule_tchk_def() {
    test_right_exam(
        &Rule::tchk_def,
        "(SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())",
    );
    test_right_exam(
        &Rule::tchk_def,
        "(SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())",
    );
    test_right_exam(
        &Rule::tchk_def,
        "(WIDTH (COND \"cybup8848\" !mem[5] cyb mem[3:5]) (2:3:))",
    );
    test_right_exam(
        &Rule::tchk_def,
        "(SKEW (COND !mem[5] cyb mem[3:5]) mem[2] (-2:3:))",
    );
}

pub fn rule_tc_spec() {
    test_right_exam(
        &Rule::tc_spec,
        "(TIMINGCHECK 
        (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
        (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
        (WIDTH (COND \"cybup8848\" !mem[5] cyb mem[3:5]) (2:3:))
        (SKEW (COND !mem[5] cyb mem[3:5]) mem[2] (-2:3:))
    )",
    );
}

pub fn rule_name() {
    test_right_exam(&Rule::name, "(NAME)");
    test_right_exam(&Rule::name, "(NAME \"cybup8848\")");
    test_wrong_exam(&Rule::name, "(NAME cybup8848)");
}

pub fn rule_exception() {
    test_right_exam(
        &Rule::exception,
        "(EXCEPTION (INSTANCE a.b.c) (INSTANCE *) (INSTANCE))",
    );
    test_right_exam(&Rule::exception, "(EXCEPTION (INSTANCE))");
    test_wrong_exam(&Rule::exception, "(EXCEPTION)");
}

pub fn rule_constraint_path() {
    test_right_exam(&Rule::constraint_path, "(mem[2] mem[2:4])");
    test_right_exam(&Rule::constraint_path, "(cyb.a.b.c.mem[3:2] a.b.mem[2])");
    let text = SDFParser::parse(Rule::constraint_path, "(cyb.a.b.c.mem[3:2] a.b.mem[2])");
    for i in text.unwrap().next().unwrap().into_inner() {
        match i.as_rule() {
            Rule::port_instance => println!("port_instance:{}", i.as_str()),
            _ => (),
        }
    }
    test_right_exam(&Rule::constraint_path, "(cyb a.b.mem[2:3])");

    let text = SDFParser::parse(Rule::constraint_path, "(cyb a.b.mem[2])");
    for i in text.unwrap().next().unwrap().into_inner() {
        match i.as_rule() {
            Rule::port_instance => println!("port_instance:{}", i.as_str()),
            _ => (),
        }
    }
    test_wrong_exam(&Rule::constraint_path, "(mem[2] mem[2\\])");
}

pub fn rule_path_constraint_item() {
    test_right_exam(&Rule::path_constraint_item,"(PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] () (-2:3:4))");
    test_right_exam(&Rule::path_constraint_item,"(PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] (2:4:) (-2:3:4))");
    test_right_exam(
        &Rule::path_constraint_item,
        "(PATHCONSTRAINTITEM mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] (2) (-2:3:4))",
    );
    test_wrong_exam(
        &Rule::path_constraint_item,
        "(PATHCONSTRAINTITEM mem[2] (2:4:) (-2:3:4))",
    );
}

pub fn rule_period_constraint_item() {
    test_right_exam(&Rule::period_constraint_item,
        "(PERIODCONSTRAINTITEM anc.c.d.mem[2:4] (2::) (EXCEPTION (INSTANCE a.b.c) (INSTANCE *) (INSTANCE cyb.c) (INSTANCE))))");
    test_right_exam(
        &Rule::period_constraint_item,
        "(PERIODCONSTRAINTITEM mem[2] (2:3:4))",
    );
    test_right_exam(
        &Rule::period_constraint_item,
        "(PERIODCONSTRAINTITEM mem[2] (2))",
    );
    test_wrong_exam(
        &Rule::period_constraint_item,
        "(PERIODCONSTRAINTITEM mem[2] () ())",
    );
}

pub fn rule_sum_item() {
    test_right_exam(
        &Rule::sum_item,
        "(SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (cyb amc.c.c.mem[2:3]) (2) (2::))",
    );
    test_right_exam(
        &Rule::sum_item,
        "(SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (cyb amc.c.c.mem[2:3]) (2))",
    );
    test_right_exam(
        &Rule::sum_item,
        "(SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))",
    );
    test_wrong_exam(&Rule::sum_item, "(SUM (mem[2] mem[2:4]) (2) (2::))");
}

pub fn rule_diff_item() {
    test_right_exam(
        &Rule::diff_item,
        "(DIFF (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))",
    );
    test_right_exam(
        &Rule::diff_item,
        "(DIFF (mem[2] mem[2:4]) (a.c.b.mem[2] cyb) (2::))",
    );
    test_right_exam(
        &Rule::diff_item,
        "(DIFF (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) ())",
    );
    test_wrong_exam(&Rule::diff_item, "(DIFF (mem[2] mem[2:4]) (2) (2::))");
}

pub fn rule_skew_constraint_item() {
    test_right_exam(&Rule::skew_constraint_item, "(SKEWCONSTRAINT mem[2:3] (2))");
    test_right_exam(
        &Rule::skew_constraint_item,
        "(SKEWCONSTRAINT a.c.v.mem[2] ())",
    );
    test_right_exam(
        &Rule::skew_constraint_item,
        "(SKEWCONSTRAINT mem[2:3] (2::))",
    );
    test_right_exam(
        &Rule::skew_constraint_item,
        "(SKEWCONSTRAINT (10 mem[2]) (2))",
    );
    test_right_exam(
        &Rule::skew_constraint_item,
        "(SKEWCONSTRAINT (negedge mem[2]) (2))",
    );
}

pub fn rule_cns_def() {
    test_right_exam(&Rule::cns_def,"(PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] () (-2:3:4))");
    test_right_exam(&Rule::cns_def, "(PERIODCONSTRAINTITEM mem[2] (2:3:4))");
    test_right_exam(
        &Rule::cns_def,
        "(SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))",
    );
    test_right_exam(
        &Rule::cns_def,
        "(DIFF (mem[2] mem[2:4]) (a.c.b.mem[2] cyb) (2::))",
    );
    test_right_exam(&Rule::cns_def, "(SKEWCONSTRAINT mem[2:3] (2))");
}

pub fn rule_arrival_departure_item() {
    test_right_exam(
        &Rule::arrival_departure_item,
        "(ARRIVAL (negedge mem[2:3]) a.c.v.mem () (2::) (-2) (-2::0))",
    );
    test_right_exam(
        &Rule::arrival_departure_item,
        "(DEPARTURE a.c.v.mem () (2::) (-2) (-2::0))",
    );
    test_right_exam(
        &Rule::arrival_departure_item,
        "(DEPARTURE mem[2] () (2::) (-2) (-2::0))",
    );
    test_right_exam(
        &Rule::arrival_departure_item,
        "(DEPARTURE (10 mem[2]) cyb () (2::) (-2) (-2::0))",
    );
    test_wrong_exam(
        &Rule::arrival_departure_item,
        "(DEPARTURE a.c.v.mem (2::) (-2) (-2::0))",
    );
    test_wrong_exam(
        &Rule::arrival_departure_item,
        "(DEPARTURE (10 mem[2]) cyb (-2) (-2::0))",
    );
}

pub fn rule_slack_item() {
    test_right_exam(
        &Rule::slack_item,
        "(SLACK a.c.v.mem () (2::) (-2) (-2::0) 2.1535)",
    );
    test_right_exam(
        &Rule::slack_item,
        "(SLACK a.c.v.mem () (2::) (-2) (-2::0) 2)",
    );
    test_right_exam(&Rule::slack_item, "(SLACK mem[2] () (2::) (-2) (-2::0))");
    test_wrong_exam(&Rule::slack_item, "(SLACK () (2::) (-2) (-2::0))");
}

pub fn rule_waveform_item() {
    test_right_exam(
        &Rule::waveform_item,
        "(WAVEFORM a.c.v.mem 2.1535 
    (posedge 1.2) (negedge 1.2 2.3) (posedge 2.3 4.5) (negedge 2.3 4.5))",
    );
    test_right_exam(
        &Rule::waveform_item,
        "(WAVEFORM a.c.v.mem 2.1535 
        (posedge 1.2) (negedge 1.2 2.3))",
    );
    test_right_exam(
        &Rule::waveform_item,
        "(WAVEFORM a.c.v.mem 2.1535 
        (negedge 1.2) (posedge 1.2 2.3) (negedge 2.3 4.5) (posedge 2.3 4.5))",
    );
    test_wrong_exam(
        &Rule::waveform_item,
        "(WAVEFORM a.c.v.mem 2.1535 
        (posedge 1.2) (negedge 1.2 2.3) (negedge 2.3 4.5) (posedge 2.3 4.5))",
    );
}

pub fn rule_tenv_def() {
    test_right_exam(
        &Rule::tenv_def,
        "(DEPARTURE a.c.v.mem () (2::) (-2) (-2::0))",
    );
    test_right_exam(
        &Rule::tenv_def,
        "(SLACK a.c.v.mem () (2::) (-2) (-2::0) 2.1535)",
    );
    test_right_exam(
        &Rule::tenv_def,
        "(WAVEFORM a.c.v.mem 2.1535 
        (posedge 1.2) (negedge 1.2 2.3) (posedge 2.3 4.5) (negedge 2.3 4.5))",
    );
}

pub fn rule_te_def() {
    test_right_exam(&Rule::te_def,"(PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] () (-2:3:4))");
    test_right_exam(&Rule::te_def, "(PERIODCONSTRAINTITEM mem[2] (2:3:4))");
    test_right_exam(
        &Rule::te_def,
        "(SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))",
    );
    test_right_exam(
        &Rule::te_def,
        "(DIFF (mem[2] mem[2:4]) (a.c.b.mem[2] cyb) (2::))",
    );
    test_right_exam(&Rule::te_def, "(SKEWCONSTRAINT mem[2:3] (2))");

    test_right_exam(&Rule::te_def, "(DEPARTURE a.c.v.mem () (2::) (-2) (-2::0))");
    test_right_exam(
        &Rule::te_def,
        "(SLACK a.c.v.mem () (2::) (-2) (-2::0) 2.1535)",
    );
    test_right_exam(
        &Rule::te_def,
        "(WAVEFORM a.c.v.mem 2.1535 
        (posedge 1.2) (negedge 1.2 2.3) (posedge 2.3 4.5) (negedge 2.3 4.5))",
    );
}

pub fn rule_te_spec() {
    test_right_exam(&Rule::te_spec,"(TIMINGENV 
        (PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] () (-2:3:4))
        (PERIODCONSTRAINTITEM mem[2] (2:3:4))
        (SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))
        (WAVEFORM a.c.v.mem 2.1535 
            (posedge 1.2) (negedge 1.2 2.3) (posedge 2.3 4.5) (negedge 2.3 4.5))
        (SKEWCONSTRAINT mem[2:3] (2))
        (DEPARTURE a.c.v.mem () (2::) (-2) (-2::0))
    )");
}

pub fn rule_pos_neg_pair_posedge() {
    test_right_exam(&Rule::pos_neg_pair_posedge, "(posedge 2.3 3.4)");
    test_right_exam(&Rule::pos_neg_pair_posedge, "(posedge 2.3)");
    test_wrong_exam(&Rule::pos_neg_pair_posedge, "(posedge)");
}

pub fn rule_pos_neg_pair_negedge() {
    test_right_exam(&Rule::pos_neg_pair_negedge, "(negedge 2.3 3.4)");
    test_right_exam(&Rule::pos_neg_pair_negedge, "(negedge 2.3)");
    test_wrong_exam(&Rule::pos_neg_pair_negedge, "(posedge)");
}

pub fn rule_pos_pair() {
    test_right_exam(&Rule::pos_pair, "(posedge -2.3)(negedge 2.3 3.4)");
    test_right_exam(&Rule::pos_pair, "(posedge -2.3 -34) (negedge 2.3)");
    test_wrong_exam(&Rule::pos_pair, "(posedge -2.3 -34) (posedge 2.3)");
}

pub fn rule_neg_pair() {
    test_right_exam(&Rule::neg_pair, "(negedge -2.3)(posedge 2.3 3.4)");
    test_right_exam(&Rule::neg_pair, "(negedge -2.3 -34) (posedge 2.3 -2.3)");
    test_wrong_exam(&Rule::neg_pair, "(posedge -2.3 -34) (posedge 2.3)");
}

pub fn rule_edge_list_type1() {
    test_right_exam(&Rule::edge_list_type1, "(posedge -2.3) (negedge 3.5 -3.4)");
    test_right_exam(
        &Rule::edge_list_type1,
        "(posedge -2.3)(negedge 2.3 3.4) (posedge 2.3) (negedge 3.5 -3.4)",
    );
    test_wrong_exam(
        &Rule::edge_list_type1,
        "(posedge -2.3)(posedge 2.3 3.4) (posedge 2.3) (negedge 3.5 -3.4)",
    );
}

pub fn rule_edge_list_type2() {
    test_right_exam(&Rule::edge_list_type2, "(negedge -2.3) (posedge 3.5 -3.4)");
    test_right_exam(&Rule::edge_list_type2,"(negedge 2.3 3.4) (posedge -2.3) (negedge 3.5 -3.4) (posedge 2.3) (negedge -2.3) (posedge 3.5 -3.4)");
    test_wrong_exam(
        &Rule::edge_list_type2,
        "(posedge -2.3)(posedge 2.3 3.4) (posedge 2.3) (negedge 3.5 -3.4)",
    );
}

pub fn rule_edge_list() {
    test_right_exam(&Rule::edge_list, "(negedge -2.3) (posedge 3.5 -3.4)");
    test_right_exam(&Rule::edge_list,"(negedge 2.3 3.4) (posedge -2.3) (negedge 3.5 -3.4) (posedge 2.3) (negedge -2.3) (posedge 3.5 -3.4)");
    test_wrong_exam(
        &Rule::edge_list,
        "(posedge -2.3)(posedge 2.3 3.4) (posedge 2.3) (negedge 3.5 -3.4)",
    );

    test_right_exam(&Rule::edge_list, "(posedge -2.3) (negedge 3.5 -3.4)");
    test_right_exam(
        &Rule::edge_list,
        "(posedge -2.3)(negedge 2.3 3.4) (posedge 2.3) (negedge 3.5 -3.4)",
    );
    test_wrong_exam(
        &Rule::edge_list,
        "(posedge -2.3)(posedge 2.3 3.4) (posedge 2.3) (negedge 3.5 -3.4)",
    );
}

pub fn rule_timing_spec() {
    test_right_exam(&Rule::timing_spec,"(DELAY 
        (INCREMENT
            (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45))
            (COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
            (CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
            (PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
            (INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
            (DEVICE mem[2:4] (12) () () (12) (12::) () () () () (34.5) (12))
        )
        (PATHPULSEPERCENT a y (25) (35)))
        ");

    test_right_exam(
        &Rule::timing_spec,
        "(TIMINGCHECK 
        (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
        (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
        (WIDTH (COND \"cybup8848\" !mem[5] cyb mem[3:5]) (2:3:))
        (SKEW (COND !mem[5] cyb mem[3:5]) mem[2] (-2:3:))
    )",
    );

    test_right_exam(&Rule::timing_spec,"(TIMINGENV 
        (PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] () (-2:3:4))
        (PERIODCONSTRAINTITEM mem[2] (2:3:4))
        (SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))
        (WAVEFORM a.c.v.mem 2.1535 
            (posedge 1.2) (negedge 1.2 2.3) (posedge 2.3 4.5) (negedge 2.3 4.5))
        (SKEWCONSTRAINT mem[2:3] (2))
        (DEPARTURE a.c.v.mem () (2::) (-2) (-2::0))
    )");
}

pub fn rule_cell() {
    test_right_exam(
        &Rule::cell,
        "(CELL
        (CELLTYPE \"DFF\")
        (INSTANCE *)
    )",
    );

    test_right_exam(&Rule::cell,"(CELL
        (CELLTYPE \"DFF\")
        (INSTANCE *)
        (TIMINGENV 
            (PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] () (-2:3:4))
            (PERIODCONSTRAINTITEM mem[2] (2:3:4))
            (SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))
            (WAVEFORM a.c.v.mem 2.1535 
                (posedge 1.2) (negedge 1.2 2.3) (posedge 2.3 4.5) (negedge 2.3 4.5))
            (SKEWCONSTRAINT mem[2:3] (2))
            (DEPARTURE a.c.v.mem () (2::) (-2) (-2::0))
        )
        (TIMINGCHECK 
            (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
            (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
            (WIDTH (COND \"cybup8848\" !mem[5] cyb mem[3:5]) (2:3:))
            (SKEW (COND !mem[5] cyb mem[3:5]) mem[2] (-2:3:))
        )
        (DELAY 
            (INCREMENT
                (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45))
                (COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
                (CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
                (PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
                (INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
                (DEVICE mem[2:4] (12) () () (12) (12::) () () () () (34.5) (12))
            )
            (PATHPULSEPERCENT a y (25) (35))
        )
    )");

    test_right_exam(&Rule::cell,"(CELL
        (CELLTYPE \"picorv32\")
        (INSTANCE)
        (DELAY
         (ABSOLUTE
          (INTERCONNECT clk U9363.CLK (9.139:9.139:9.139) (4.401:4.401:4.401))
          (INTERCONNECT mem_rdata[9] U367.B1 (0.022:0.022:0.022) (0.009:0.009:0.009))
          (INTERCONNECT mem_rdata[9] U6255.A1 (0.022:0.022:0.022) (0.009:0.009:0.009))
          (INTERCONNECT mem_ready U13.A (0.021:0.021:0.021) (0.009:0.009:0.009))
          (INTERCONNECT resetn U9.A (0.033:0.033:0.033) (0.015:0.015:0.015))
          (INTERCONNECT U0.X pcpi_rs2[7] (0.000:0.000:0.000))
          (INTERCONNECT U1.X pcpi_rs2[6] (0.000:0.000:0.000))
          (INTERCONNECT U10.X U5132.A1 (0.000:0.000:0.000))
          (INTERCONNECT U1027.Y U1028.A1 (0.000:0.000:0.000))
         )
        )
    )");

    test_right_exam(&Rule::iopath_item,"(IOPATH A X (0.258:0.258:0.258) (0.210:0.210:0.210))");
    test_right_exam(&Rule::cell_instance,"(INSTANCE U0)");

    test_right_exam(&Rule::cell,"(CELL
        (CELLTYPE \"sky130_fd_sc_hd__buf_2\")
        (INSTANCE U0)
        (DELAY
         (ABSOLUTE
          (IOPATH A X (0.258:0.258:0.258) (0.210:0.210:0.210))
         )
        )
       )");
}

pub fn rule_delay_file() {
    test_right_exam(&Rule::delay_file,"(DELAYFILE
        (SDFVERSION \"3.0\")
        (DESIGN \"picorv32\")
        (DATE \"Fri Mar 24 09:04:43 2023\")
        (VENDOR \"Parallax\")
        (PROGRAM \"STA\")
        (VERSION \"2.4.0\")
        (DIVIDER .)
        (VOLTAGE 1.800::1.800)
        (PROCESS \"1.000::1.000\")
        (TEMPERATURE 25.000::25.000)
        (TIMESCALE 1ns)
        (CELL
            (CELLTYPE \"DFF\")
            (INSTANCE *)
            (TIMINGENV 
                (PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] () (-2:3:4))
                (PERIODCONSTRAINTITEM mem[2] (2:3:4))
                (SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))
                (WAVEFORM a.c.v.mem 2.1535 
                    (posedge 1.2) (negedge 1.2 2.3) (posedge 2.3 4.5) (negedge 2.3 4.5))
                (SKEWCONSTRAINT mem[2:3] (2))
                (DEPARTURE a.c.v.mem () (2::) (-2) (-2::0))
            )
            (TIMINGCHECK 
                (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
                (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
                (WIDTH (COND \"cybup8848\" !mem[5] cyb mem[3:5]) (2:3:))
                (SKEW (COND !mem[5] cyb mem[3:5]) mem[2] (-2:3:))
            )
            (DELAY 
                (INCREMENT
                    (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45))
                    (COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
                    (CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
                    (PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
                    (INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
                    (DEVICE mem[2:4] (12) () () (12) (12::) () () () () (34.5) (12))
                )
                (PATHPULSEPERCENT a y (25) (35))
            )
        )
        (CELL
            (CELLTYPE \"DFF\")
            (INSTANCE *)
            (TIMINGENV 
                (PATHCONSTRAINTITEM (NAME \"cybup8848\") mem[2] mem[2:4] anc.c.d.mem[2:4] amc/mem[2] () (-2:3:4))
                (PERIODCONSTRAINTITEM mem[2] (2:3:4))
                (SUM (mem[2] mem[2:4]) (a.c.b.mem[2] a.c.v) (2) (2::))
                (WAVEFORM a.c.v.mem 2.1535 
                    (posedge 1.2) (negedge 1.2 2.3) (posedge 2.3 4.5) (negedge 2.3 4.5))
                (SKEWCONSTRAINT mem[2:3] (2))
                (DEPARTURE a.c.v.mem () (2::) (-2) (-2::0))
            )
            (TIMINGCHECK 
                (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
                (SETUPHOLD (01 mem[2]) mem[3] (12:3:34) ())
                (WIDTH (COND \"cybup8848\" !mem[5] cyb mem[3:5]) (2:3:))
                (SKEW (COND !mem[5] cyb mem[3:5]) mem[2] (-2:3:))
            )
            (DELAY 
                (INCREMENT
                    (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45))
                    (COND ~&((~&((mem[2:5])))) (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
                    (CONDELSE (IOPATH (negedge mem[12]) src[0:18] (RETAIN ((12) (12:4:45)) ()) (RETAIN ((12) (12:4:45))) (12::45)))
                    (PORT nen[2:23] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
                    (INTERCONNECT nen[2:23] mem[123] ((12:23:5) (12) ()) () (12) (12::) () () () () (34.5) (12))
                    (DEVICE mem[2:4] (12) () () (12) (12::) () () () () (34.5) (12))
                )
                (PATHPULSEPERCENT a y (25) (35))
            )
        )
    )");

    test_wrong_exam(
        &Rule::delay_file,
        "(DELAYFILE
        (SDFVERSION \"3.0\")
        (DESIGN \"picorv32\")
        (DATE \"Fri Mar 24 09:04:43 2023\")
        (VENDOR \"Parallax\")
        (PROGRAM \"STA\")
        (VERSION \"2.4.0\")
        (DIVIDER .)
        (VOLTAGE 1.800::1.800)
        (PROCESS \"1.000::1.000\")
        (TEMPERATURE 25.000::25.000)
        (TIMESCALE 1ns)
    )",
    );

    let unparsed_text = std::fs::read_to_string("picorv32.sdf").expect("read file unsuccessfully");
    //println!("{}", unparsed_text);
    test_right_exam(&Rule::delay_file, &unparsed_text);
}
