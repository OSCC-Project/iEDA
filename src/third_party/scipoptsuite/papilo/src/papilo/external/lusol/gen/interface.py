#!/usr/bin/env python3

import io

def parse_org_table(table_lines):
    # remove separator row
    table_lines.pop(1)
    table_list = [[b.strip() for b in a[1:-2].split('|')] for a in table_lines]
    # get column list
    column_list = table_list.pop(0)
    #print(column_names)
    # organize table data
    table_data = []
    for param in table_list:
        param_dict = {}
        for column, value in zip(column_list,param):
            param_dict[column] = value
        table_data.append(param_dict)
    #print(table_data)
    return table_data

def read_org_file(file_name):
    # read lines
    file = open(file_name,'r')
    file_lines = file.readlines()
    file.close()
    # get function name
    function_name = file_lines[0].strip()
    #print(function_name)
    # parse remaining lines as table
    table_data = parse_org_table(file_lines[1:])
    return function_name, table_data

def load_interface_files(file_name):
    # read lines
    file = open(file_name,'r')
    file_lines = file.readlines()
    file.close()
    interface_list = parse_org_table(file_lines)
    interface_data = []
    for interface_function in interface_list:
        function_name, argument_data = read_org_file(interface_function['interface_file'])
        d = {}
        d['function_name'] = function_name
        d['argument_data'] = argument_data
        d['format'] = interface_function['format']
        interface_data.append(d)
    #print(interface_data)
    return interface_data

def function_declaration(function_dict,prefix='',suffix=''):
    f = io.StringIO()
    f.write('void ')
    f.write(prefix + function_dict['function_name'] + suffix)
    f.write('(\n')
    for arg in function_dict['argument_data']:
        f.write('  {0}* {1},\n'.format(arg['c_type'],arg['var_name']))
    func_dec = f.getvalue()[:-2] + ')'
    #print(func_dec)
    return func_dec

def function_call(function_dict,prefix='',suffix=''):
    f = io.StringIO()
    f.write('  ' + prefix + function_dict['function_name'] + suffix)
    f.write('(')
    for arg in function_dict['argument_data']:
        f.write('{},'.format(arg['var_name']))
    func_dec = f.getvalue()[:-1] + ')'
    #print(func_dec)
    return func_dec

def get_header(file_name):
    interface_data = load_interface_files(file_name)
    # start the header buffer
    f = io.StringIO()
    # start the header file
    f.write('#ifndef CLUSOL_H_\n')
    f.write('#define CLUSOL_H_\n')
    f.write('\n')
    # include directives
    f.write('#include <stdint.h>')
    f.write('\n\n')
    # function declarations
    for interface_func in interface_data:
        f.write(function_declaration(interface_func,prefix='c'))
        f.write(';\n\n')
    # end the headerfile
    f.write('#endif // CLUSOL_H_\n')
    # clean up and return
    header_str = f.getvalue()
    f.close();
    return header_str

def get_source(file_name):
    interface_data = load_interface_files(file_name)
    # start the source buffer
    f = io.StringIO()
    # include directives
    f.write('#include "clusol.h"\n')
    f.write('\n')
    # fortran function declarations
    f.write('// declarations for fortran function calls\n')
    for interface_func in interface_data:
        if interface_func['format'] == 'f90':
            f.write(function_declaration(interface_func,prefix='__lusol_MOD_'))
        if interface_func['format'] == 'f77':
            f.write(function_declaration(interface_func,suffix='_'))
        f.write(';\n\n')
    # function calls in c
    f.write('// c interface function definitions\n')
    for interface_func in interface_data:
        f.write(function_declaration(interface_func,prefix='c'))
        f.write(' {\n')
        if interface_func['format'] == 'f90':
            f.write(function_call(interface_func,prefix='__lusol_MOD_'))
        if interface_func['format'] == 'f77':
            f.write(function_call(interface_func,suffix='_'))
        f.write(';\n')
        f.write('}\n\n')
    # clean up and return
    source_str = f.getvalue()
    f.close();
    return source_str

# for testing
if __name__ == '__main__':
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate C interface to LUSOL.')
    parser.add_argument('-i','--input',
                        help='input file name',
                        required=True)
    parser.add_argument('-o','--output',
                        help='output file name',
                        required=True)
    parser.add_argument('-t','--type',
                        help='output file type',
                        required=True,
                        choices=['header','source'])
    args = parser.parse_args()
    # generate code
    if args.type == 'header':
        file_str = get_header(args.input)
    elif args.type == 'source':
        file_str = get_source(args.input)
    else:
        raise Exception('uknown type')
    # write code
    f = open(args.output,'w')
    f.write(file_str)
    f.close()
