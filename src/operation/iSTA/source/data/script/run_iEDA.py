import os
import shutil
import sys
import subprocess

# 检查输入参数数量
if len(sys.argv) < 2:
    print("请输入至少一个参数！")
    print("使用方法: python script.py 参数1 参数2 参数3 ...")
    sys.exit(1)

input_params = sys.argv[1:]

def copy_and_rename_tcl(param, current_dir):
    copy_to_dir = os.path.join(os.path.dirname(current_dir), param)

    # 构建源文件路径
    source_file = os.path.join(current_dir, 'run_iEDA.tcl')
    destination_file = os.path.join(copy_to_dir, f'run_{param}.tcl')
    
    # 复制并重命名文件
    shutil.copyfile(source_file, destination_file)
    print(f"复制并重命名成功：run_iEDA.tcl -> {destination_file}")

    # 修改新文件的内容，更新设计名称
    with open(destination_file, 'r') as file:
        content = file.read()
    
    new_content = content.replace('set design_name s1238', f'set design_name {param}')
    
    with open(destination_file, 'w') as file:
        file.write(new_content)
        print(f"已更新文件 {destination_file} 中的设计名称")

    # 运行 iEDA 脚本
    iEDA_script = f"/data/yexinyu/iEDA/bin/iSTA {destination_file}"
    subprocess.run(iEDA_script, shell=True)
    print("已运行 iEDA 脚本")

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.realpath(__file__))

# 为每个参数生成文件
for param in input_params:
    copy_and_rename_tcl(param, current_dir)
