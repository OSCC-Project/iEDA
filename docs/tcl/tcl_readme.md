注：ScriptEngine 和 UserShell 头文件和实现在 ```iEDA/src/auxiliary/tcl``` 路径下

## 1 使用 ScriptEngine 自定义 Tcl 命令

ScriptEngine 是 Tcl 命令解析器，包含命令、命令选项、解析器等一系列工具。用户可以使用 ScriptEngine 中的接口轻松实现自定义 Tcl 命令

1. **文件结构：**

   在你的点工具下添加独立的 Tcl 命令模块便于组织文件结构，参考 ```iEDA/src/iSTA/sdc-cmd```的组织形式：

   ```shell
   ~$ tree iEDA/src/iSTA/sdc-cmd/
   iEDA/src/iSTA/sdc-cmd/
   ├── CMakeLists.txt
   ├── CmdCreateClock.cc
   ├── CmdCreateGenerateClock.cc
   ...
   └── Cmd.hh
   
   0 directories, 16 files
   ```

   其中

   ```CMakeLists.txt``` 用于生成静态库，示例见```iEDA/src/iSTA/sdc-cmd/CMakeLists.txt```；

   ```Cmd.hh``` 用于用户自定义命令声明，解析见2；

   ```CmdCreateClock.cc``` 等cpp文件用于实现命令。

    **Build 注意事项**

    1. 安装 Tcl 8.6  

      在你的工作系统环境中安装 [Tcl 8.6](https://github.com/tcltk/tcl/tree/core-8-6-branch)

    2. 在点工具CMakeList.txt下加入链接库  

      在点工具CMakeList.txt文件中添加tcl模块需要用到的头文件和链接库 ```include(${PROJECT_SOURCE_DIR}/cmake/tcl.cmake)```;  

      并添加命令模块（或者直接添加cpp文件，取决于你的构建方式）;  


2. **声明用户命令类：**

   命令类需继承 ```TclCmd```类。内容可参考 ```iEDA/src/iSTA/sdc-cmd/Cmd.hh``` 或如下实例，下列代码段 8~13 行建议直接复制。

   ```C++
   #include "tcl/ScriptEngine.hh"
   
   namespace ieda {
   /**
    * @brief your command.
    */
   class YourCommand : public TclCmd {
    public:
     explicit YourCommand(const char* cmd_name);
     ~YourCommand() override = default;
   
     unsigned check() override;
     unsigned exec() override;
   
    private:
     // private function
     // private data 
   };
   
   }
   ```

   

3. **命令实现：**

   命令类必须实现其 **构造函数**、**check()** 和 **exec()** 

   - **1. 构造函数**

     构造函数用于添加命令的 **选项(Option)** 或 **参数(Arg)**（均为```TclOption```类）。目前命令的option选项及arg参数支持 *String, String List, Int, Int List, Double, Double List, Switch(仅option)* 类（均为```TclOption```子类）

     - 选项（Option）是一条 Tcl 命令的**可选项**。option 无固定的顺序，用户需要输入提示符（如示例中```-divide_by```）来引导option的值（如```divide_factor```）。另外，Switch类型的option无值；
     - 参数（Arg）是一条 Tcl 命令的**必选项**。开发者需要**按照顺序**在构造函数中添加 arg，用户需要**按照相同的顺序**输入参数值，无需提示符引导；

     在构造函数中添加命令选项的过程包括：1 ```new Tcl{Int/Double/String/...}Option("选项/参数名",0(option选项)/1(arg参数),默认值)```; 2 使用 ```addOption``` 传入1中生成的选项指针，以注册命令。
     
     以 ```iEDA/src/iSTA/sdc-cmd/CmdGetClocks.cc``` 为例：

   ```c++
   /**
    * @brief add options and args for "create_generate_clock" command
    * @usage in Tcl-shell
    * % create_generated_clock
    *     [-name clock_name]
    *     [-source master_pin]
    *     [-divide_by divide_factor | -multiply_by multiply_factor | -edges edge_list ]
    *     ...
    *     source_objects
    */
   CmdCreateGenerateClock::CmdCreateGenerateClock(const char* cmd_name): TclCmd(cmd_name) {
     // creat an int option
     // option initialization list:
     // "-divide_by" -- option name
     //       0      -- option
     //       0      -- default value
     auto* period_option = new TclIntOption("-divide_by", 0, 0);
     addOption(period_option);
     
     // ...
   
     // creat a string list arg
     // option(arg) initialization list:
     // "source_objects" -- option name
     //       1          -- arg
     //       {}         -- default value
     auto* pin_port_arg = new TclStringListOption("source_objects", 1, {});
     addOption(pin_port_arg);
   }
   ```
   
   - **2. check()**
   
     根据命令需要，设置选项/参数的合法性检查，
     
     一般可通过 ```getOptionOrArg("选项/参数名")``` 获得对应的 ```TclOption``` 指针。通过调用 ```TclOption``` 的 ```is_set_val()``` 方法来判断此选项/参数是否被赋值，若选项/参数有值可用 ```TclOption``` 的 ```get{String/Int/Double}Val()```, ```get{String/Int/Double}List()``` 方法获得相应的值。
     
     例如 ```iEDA/src/iSTA/sdc-cmd/CmdCreateGenerateClock.cc ```:
   
   ```c++
   /**
    * @brief The create_generate_clock cmd legally check.
    * @return 1 if all options/args are legal, 0 if not
    */
   unsigned CmdCreateGenerateClock::check() {
     // rule: "-source" and "source_objects" are required
     TclOption* source_option = getOptionOrArg("-source");
     TclOption* source_obj_option = getOptionOrArg("source_objects");
     if (!(source_option->is_set_val() && source_obj_option->is_set_val())) {
       LOG_ERROR << "'-source' 'source_objects' are missing.";
       return 0;
     }
   
     // rule: "-edges", "-divide_by", "-multiply_by" are exclusive
     TclOption* edges_option = getOptionOrArg("-edges");
     TclOption* divide_by_option = getOptionOrArg("-divide_by");
     TclOption* multiply_by_option = getOptionOrArg("multiply_by");
     unsigned period_val_count = divide_by_option->is_set_val() +
                                 multiply_by_option->is_set_val() +
                                 edges_option->is_set_val();
     if (period_val_count > 1) {
       LOG_ERROR << "'-divide_by'  '-multiply_by'  '-edges' are exclusive.";
       return 0;
     }
   
     // ...
   }
   ```
   
   - **3. exec()**
   
     根据命令需要，实现命令逻辑。在 ```exec()``` 中应当先进行 ```check()``` 检查。获得参数值的方法通 2 中的描述，通过 ```getOptionOrArg("选项/参数名")``` 获得对应的 ```TclOption``` 指针，在确定有值（```is_set_val()```）的情况下调用 ```get{String/Int/Double}Val()```, ```get{String/Int/Double}List()``` 方法取得相应类型的值。
     
     例如 ```iEDA/src/iSTA/sdc-cmd/CmdCreateGenerateClock.cc``` :
   
   ```c++
   /**
    * @brief The create_generate_clock execute body.
    * @return 1 if execution success, 0 if not
    */
   unsigned CmdCreateGenerateClock::exec() {
     if (!check()) {
       return 0;
     }
   
     // do something ...
   
     return 1;
   }
   ```
   
   

## 2 UserShell 接入 main 函数

1. **在main函数中添加 用户命令类 的声明**：例如 ```#include "sdc-cmd/Cmd.hh"``` 
2. **定义初始化函数：**初始化函数可用于注册你的命令（也可执行别的初始化语句）。初始化函数形式：参数为空，返回值为int (返回0为执行成功，返回1则不会打开tcl解释器)。可使用```registerTclCmd(命令类名，命令string)```宏快速注册自定义命令。

3. 向UserShell中添加步骤*2*中的初始化函数，此初始化函数将会在 Tcl 解释器运行前执行初始化，否则将无法运行用户自定义命令
4. 使用userMain打开Tcl解释器，若参数为文件路径，则执行并退出；若参数为nullptr，将打开可交互的Tcl命令行，可用于Tcl命令输入执行

```C++
// 示例文件：iEDA/src/iSTA/main.cc
// 1.在main函数中添加用户命令类的声明
#include "sdc-cmd/Cmd.hh"
#include "shell-cmd/ShellCmd.hh"
#include "tcl/UserShell.hh"

using namespace ieda;

// 2.定义初始化函数：初始化函数可用于注册你的命令。初始化函数形式：参数为空，返回值为int (返回0为执行成功，返回1则不会打开tcl解释器)
int registerCommands() {
  // 可使用 registerTclCmd(命令类名，命令string) 宏快速注册自定义命令
  registerTclCmd(CmdReadVerilog, "read_verilog");
  // ...
  registerTclCmd(CmdGetPins, "get_pins");

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  auto shell = UserShell::getShell();
  // 3.向UserShell中添加步骤2中的初始化函数，此初始化函数将会在 Tcl 解释器运行前执行初始化，否则将无法运行用户自定义命令
  shell->set_init_func(registerCommands);

  // get Tcl file path from main args
  char* tcl_file_path = nullptr;
  if (argc == 2) {
    tcl_file_path = argv[1];
  } else {
    shell->displayHelp();
  }

  // 4.使用userMain打开Tcl解释器，若参数为文件路径，则执行并退出；若参数为nullptr，将打开可交互的Tcl命令行，可用于Tcl命令输入执行
  shell->userMain(tcl_file_path);
}
```


## 常见 BUG 

1. 命令/选项明明已经注册了，但提示命令没找到/获取选项值出现core dump

   命令没找到：检查Tcl初始化函数中，命令名是否与输入的命令名相符

   选项出现core dump：检查在命令构造函数中注册选项时选项名是否正确，检查使用命令时```getOptionOrArg``` 的选项名是否正确

2. 有待提供...

