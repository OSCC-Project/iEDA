#!/bin/bash

echo "Start evaluation..."

input_def_name=$1
input_guide_name=$2
base_def_name=$3
base_guide_name=$4
max_area_inc=$5

evaluate_dir=$(cd "$(dirname "$0")"; pwd)
run_evaluation_tcl_file=${evaluate_dir}/../../script/contest_script/run_evaluation.tcl

# 修改input_def
input_def_search_string="def_init -path "
input_def_replace_string="def_init -path ${input_def_name}"
sed -i "s#${input_def_search_string}.*#${input_def_replace_string}#g" ${run_evaluation_tcl_file}

# 修改input_guide
input_guide_search_string="run_contest_evaluation -guide "
input_guide_replace_string="run_contest_evaluation -guide ${input_guide_name}"
sed -i "s#${input_guide_search_string}.*#${input_guide_replace_string}#g" ${run_evaluation_tcl_file}

# 执行input
input_log_file=${evaluate_dir}/input.log
# cd ${evaluate_dir}/../.. ; ./IncTO_2023 -script script/contest_script/run_evaluation.tcl > ${input_log_file}

# 分析input_log
input_requirement_check=0
if [ `grep -c "Overlap check successful!" ${input_log_file}` -ne '0' ];then
    if [ `grep -c "Connectivity check successful!" ${input_log_file}` -ne '0' ];then
        if [ `grep -c "Overflow check successful!" ${input_log_file}` -ne '0' ];then
            input_requirement_check=1
        fi
    fi
fi

input_instance_area=0
input_wns=0
input_tns=0
if [ ${input_requirement_check} -eq 0 ]; then
    echo "input_requirement_check failed!"
    exit
else
    input_instance_area=$(echo $(grep "total instance area: .*" ${input_log_file}) | awk '{print $4}')  
    input_wns=$(echo $(grep "wns: .*" ${input_log_file}) | awk '{print $2}')  
    input_tns=$(echo $(grep "tns: .*" ${input_log_file}) | awk '{print $2}')  
fi

# 修改base_def
base_def_search_string="def_init -path "
base_def_replace_string="def_init -path ${base_def_name}"
sed -i "s#${base_def_search_string}.*#${base_def_replace_string}#g" ${run_evaluation_tcl_file}

# 修改base_guide
base_guide_search_string="run_contest_evaluation -guide "
base_guide_replace_string="run_contest_evaluation -guide ${base_guide_name}"
sed -i "s#${base_guide_search_string}.*#${base_guide_replace_string}#g" ${run_evaluation_tcl_file}

# 执行base
base_log_file=${evaluate_dir}/base.log
# cd ${evaluate_dir}/../.. ; ./IncTO_2023 -script script/contest_script/run_evaluation.tcl > ${base_log_file}

# 分析base_log
base_requirement_check=0
if [ `grep -c "Overlap check successful!" ${base_log_file}` -ne '0' ];then
    if [ `grep -c "Connectivity check successful!" ${base_log_file}` -ne '0' ];then
        if [ `grep -c "Overflow check successful!" ${base_log_file}` -ne '0' ];then
            base_requirement_check=1
        fi
    fi
fi

base_instance_area=0
base_wns=0
base_tns=0
if [ ${base_requirement_check} -eq 0 ]; then
    echo "base_requirement_check failed!"
    exit
else
    base_instance_area=$(echo $(grep "total instance area: .*" ${base_log_file}) | awk '{print $4}')  
    base_wns=$(echo $(grep "wns: .*" ${base_log_file}) | awk '{print $2}')  
    base_tns=$(echo $(grep "tns: .*" ${base_log_file}) | awk '{print $2}')  
fi

# calc calc calc
# calc calc calc
inc_instance_area_check=$(echo "$input_instance_area > (1 + $max_area_inc) * $base_instance_area" | bc -l)

if [ $inc_instance_area_check -eq 1 ]; then
    echo "inc_instance_area_check failed!"
    exit
fi

echo "input_wns: " $input_wns
echo "input_tns: " $input_tns
echo "base_wns: " $base_wns
echo "base_tns: " $base_tns
w_wns=1.0
w_tns=2.0
final_score=$(echo "scale=2; ($w_wns * (1 - ($input_wns / $base_wns)) + $w_tns * (1 - ($input_tns / $base_tns))) / ($w_wns + $w_tns) * 100.0" | bc)

echo "############################################################"
echo "############################################################"
echo "############################################################"
echo "final_score: ${final_score}"  
echo "############################################################"
echo "############################################################"
echo "############################################################"
echo "End evaluation"






