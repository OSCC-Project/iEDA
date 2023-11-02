#[derive(Clone)]
//designed data structure for rtriple、rnumber
pub enum r_triple_number{
    r_triple([f64;3]),
    r_number(f64),
}

//designed data structure for rvalue、value、rtriple、triple
#[derive(Clone)]
pub enum r_value{
    val(r_triple_number),
    none,
}


//designed data structure for time_scale
pub struct time_scale_structure{
    pub number:f64,
    pub unit:String,
}

//store sdf header as hash map
pub enum sdf_header_val{
    str(String),
    val(r_triple_number),
    time_scale(time_scale_structure),
}


//designed data struture for del_spec，from down to top
pub enum path_pulse_percent_item{
    no_in_out_path(String,Vec<r_value>),
    with_in_out_path(String,[String;2],Vec<r_value>),
}

//designed data struture for del_def
#[derive(Clone)]
pub enum port_spec{
    port_edge(String,String),
    port_instance(String),
}

#[derive(Clone)]
pub struct retain_item{
    pub keyword:String,
    pub delval_list:Vec<Vec<r_value>>,
}

#[derive(Clone)]
pub enum iopath_item{
    no_retain(String,port_spec,String,Vec<Vec<r_value>>),
    with_retain(String,port_spec,String,Vec<retain_item>,Vec<Vec<r_value>>),
}

pub enum cond_item{
    no_qstring(String,String,iopath_item),
    with_qstring(String,String,String,iopath_item),
}

pub struct condelse_item{
    pub keyword:String,
    pub iopath:iopath_item,
}

#[derive(Clone)]
pub struct port_item{
    pub keyword:String,
    pub port_instance:String,
    pub delval_list:Vec<Vec<r_value>>,
}

pub struct interconnect_item{
    pub keyword:String,
    pub input:String,
    pub output:String,
    pub delval_list:Vec<Vec<r_value>>,
}

pub enum device_item{
    no_port_instance(String,Vec<Vec<r_value>>),
    with_port_instance(String,String,Vec<Vec<r_value>>),
}

pub enum del_def{
    iopath(iopath_item),
    cond(cond_item),
    condelse(condelse_item),
    port(port_item),
    interconnect(interconnect_item),
    device(device_item),
}

pub struct absolute_increment_item{
    pub keyword:String,
    pub del_defs:std::collections::LinkedList::<del_def>,
}

pub enum deltype{
    path_pulse_percent(path_pulse_percent_item),
    absolute_increment(absolute_increment_item),
    none,
}

pub struct del_spec{
    pub keyword:String,
    pub deltypes:std::collections::LinkedList::<deltype>,
}

//designed data struture for tc_spec、tchk_def
#[derive(Clone)]
pub struct scalar_node_item{
    pub scalar_port:String,
    pub scalar_net:String,
}

#[derive(Clone)]
pub enum timing_check_condition_item{
    scalar_node(scalar_node_item),
    inv_scalar_node(String,scalar_node_item),
    node_equal_const(scalar_node_item,String,String),
}

pub enum cond{
    no_qstring(String,timing_check_condition_item,port_spec),
    with_qstring(String,String,timing_check_condition_item,port_spec),
}

pub enum port_tchk{
    port_spec(port_spec),
    cond(cond),
}


pub enum sccond_item{
    no_qstring(String,timing_check_condition_item),
    with_qstring(String,String,timing_check_condition_item),
}

pub struct setup_hold_recovery_removal_item{
    pub keyword:String,
    pub in_out:[port_tchk;2],
    pub val:r_value,
}

pub struct setuphold_item1_recrem_item1_nochange_item{
    pub keyword:String,
    pub in_out:[port_tchk;2],
    pub value:[r_value;2]
}

pub enum setuphold_item2_recrem_item2_item{
    no_sccond(String,[port_spec;2],[r_value;2]),
    with_sccond(String,[port_spec;2],[r_value;2],Vec<sccond_item>),
}

pub struct skew_item{
    pub keyword:String,
    pub in_out:[port_tchk;2],
    pub val:r_value,
}

pub struct width_period_item{
    pub keyword:String,
    pub port:port_tchk,
    pub val:r_value,
}

pub enum tchk_def{
    setup_hold_recovery_removal(setup_hold_recovery_removal_item),
    setuphold_item1_recrem_item1_nochange(setuphold_item1_recrem_item1_nochange_item),
    setuphold_item2_recrem_item2(setuphold_item2_recrem_item2_item),
    skew(skew_item),
    width_period(width_period_item),
    none,
}

pub struct tc_spec{
    pub keyword:String,
    pub tchk_defs:std::collections::LinkedList::<tchk_def>,
}

//designed data struture for te_spec
pub enum name{
    no_qstring(String),
    with_qstring(String,String),
}

pub enum path_constraint_item{
    no_name(String,Vec<String>,[r_value;2]),
    with_name(String,name,Vec<String>,[r_value;2])
}

pub enum cell_instance{
    no_path(String),
    with_path(String,String),
}

pub struct exception{
    pub keyword:String,
    pub cell_instances:Vec<cell_instance>,
}

pub enum period_constraint_item{
    no_exception(String,String,r_value),
    with_exception(String,String,r_value,exception),
}

pub struct sum_item{
    pub keyword:String,
    pub constraint_paths:Vec<[String;2]>,
    pub vals:Vec<r_value>,
}

pub struct diff_item{
    pub keyword:String,
    pub constraint_paths:[[String;2];2],
    pub vals:Vec<r_value>,
}

pub struct skew_constraint_item{
    pub keyword:String,
    pub port:port_spec,
    pub val:r_value,
}

pub enum cns_def_item{
    path_constraint(path_constraint_item),
    period_constraint(period_constraint_item),
    sum(sum_item),
    diff(diff_item),
    skew_constraint(skew_constraint_item),
    none,
}


pub enum arrival_departure_item{
    no_port_edge(String,String,[r_value;4]),
    with_port_edge(String,(String,String),String,[r_value;4]),
}

pub enum slack_item{
    no_number(String,String,[r_value;4]),
    with_number(String,String,[r_value;4],f64),
}

pub struct pos_neg{
    pub keyword:String,
    pub nums:Vec<f64>,
}

pub struct pos_neg_pair{
    pub part1:pos_neg,
    pub part2:pos_neg,
}

pub struct waveform_item{
    pub keyword:String,
    pub port:String,
    pub num:f64,
    pub edge_list:Vec<pos_neg_pair>,
}

pub enum tenv_def_item{
    arrival_departure(arrival_departure_item),
    slack(slack_item),
    waveform(waveform_item),
    none,
}

pub enum te_def{
    cns_def(cns_def_item),
    tenv_def(tenv_def_item),
    none,
}

pub struct te_spec{
    pub keyword:String,
    pub te_defs:std::collections::LinkedList::<te_def>,
}

pub enum timing_spec{
    del_spec(del_spec),
    tc_spec(tc_spec),
    te_spec(te_spec),
    none,
}

pub struct cell{
    pub keyword:String,
    pub cell_type:String,
    pub cell_instance:String,
    pub timing_spec:std::collections::LinkedList::<timing_spec>,   
}


pub struct sdf_data{
    pub keyword:String,
    pub sdf_header:std::collections::HashMap::<String,sdf_header_val>,
    pub cell:std::collections::LinkedList::<cell>,
} 