#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File : generate_doc.py
@Author : yell
@Desc : generate python api description doc
'''

######################################################################################
# import ai-eda as root

######################################################################################
import sys
import os
import csv
import re
      
def extract_info(root_dir, output_csv):  
        # 遍历根目录下的所有子文件夹
    apis_dict = {}
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        module_name = os.path.basename(dirpath)
        if not module_name.startswith('py_'):
            continue
        
        module_name = module_name.removeprefix("py_")
        
        apis_dict[module_name] = {}
        
        # 只处理包含py_register_*.py的文件
        for filename in filenames:
            if filename.startswith('py_register_') and filename.endswith('.h'):
                file_path = os.path.join(dirpath, filename)
                print(f"处理文件: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                lines = [line.split('//', 1)[0] for line in content.split('\n')]
                processed_content = '\n'.join(lines)
                
                # 查找所有m.def(...)的定义
                pattern = r'm\.def\s*\(\s*"([^"]+)"\s*,\s*[^,]+(?:,\s*py::arg\s*\(\s*"([^"]+)"\s*\))*'
                matches = re.finditer(pattern, processed_content)
                print(matches)
                
                for match in matches:
                    function_name = match.group(1)
                    
                    apis_dict[module_name][function_name] = {}
                    
                    # 提取参数列表
                    # args = []
                    arg_pattern = r'py::arg\s*\(\s*"([^"]+)"\s*\)'
                    arg_matches = re.findall(arg_pattern, match.group(0))
                    # args.extend(arg_matches)
                    
                    for parameter in arg_matches:
                        apis_dict[module_name][function_name][parameter] = {}
                    
                    # parameters = ', '.join(args) if args else 'None'
                    # apis.append((module_name, function_name, parameters))
                    
    return apis_dict
     
def save_csv(api_list : dict, output_csv):
    if api_descriptions is None:
        print(f"data is empty")
        return
    
        # 准备CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['modules', 'apis', 'parameters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for module_key, module_value in api_list.items():
            for function_key, function_value in module_value.items():
                parameters = ""
                if len(function_value) > 0:
                    for param_key, param_value in function_value.items(): 
                        parameters = "{}{}\n".format(parameters, param_key)
                else:
                    parameters = "None"
        
        # for(module_name, function_name, parameters) in api_list:
                writer.writerow({
                        'modules': module_key,
                        'apis': function_key,
                        'parameters': parameters
                    })
        
    print(f"处理完成，结果已保存到 {output_csv}")

if __name__ == "__main__":   
    current_dir = os.path.split(os.path.abspath(__file__))[0]
    sys.path.append(current_dir)

    python_api_dir = current_dir.rsplit('/', 3)[0] + "/src/interface/python"
    csv_file = "{}/api.csv".format(current_dir)

    api_descriptions = None
    # 输出CSV文件路径
    if not os.path.exists(python_api_dir):
        print(f"错误：目录 {python_api_dir} 不存在")
    else:
        api_descriptions = extract_info(python_api_dir, csv_file)
    
    save_csv(api_descriptions, csv_file)

