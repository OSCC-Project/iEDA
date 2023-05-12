<div align="center">

<!-- <img src="docs/resources/iEDA-logo-new.png" width="15%" height="15%" alt="iEDA-logo" /> -->

<img src="docs/resources/iEDA.png" width="15%" height="15%" alt="iEDA-logo" />
 <h1>iEDA</h1>
 <h3> An pen-source from netlist to GDS EDA platform for ASIC design.</h3>

[License][License-url]

[Chinese][README-CN-path] | **English**

</div>

iEDA Homepage：[ieda.oscc.cc][iEDA-OSCC-url]

## **About iEDA**

- **About “i”in iEDA**
  - Meaning 1: Infrastructure
  - Meaning 2: Intelligent
- **The goal of the iEDA project**
  - EDA Infrastructure
  - High quality and performance EDA tool
- **Open-source is not a goal but a way**

## **Main Contents and Plans**

- Enhance the **infrastructure** to support more design requirement
- Complete the EDA tool chain from **RTL-GDS II**
- Improve the **quality and performance** of all EDA tool operations
- Construct **AI for EDA** platform and introduce trained **AI model** to the EDA platform
- Build **data system** with enough chip design and labeling process data
- Achieve the adaptability of the EDA platform for **cloud-native**

<div align="center">
 <img src= "docs/resources/iEDA_framework_new.png" width="550" height="50%" alt="iEDA_framework" />
</div>

## **iEDA Structure**

<div align="center">
 <img src= "docs/resources/iEDA-structure.png" width="650" height="65%"  alt="iEDA_structure" />
</div>

## **iEDA Infrastructure**

<div align="center">
 <img src= "docs/resources/iEDA-infrastructure.png" width="650" height="60%" alt="iEDA_infrastructure" />
</div>

## **iEDA Operations (Tools)**

<div align="center">
 <img src= "docs/resources/iEDA-operation.png" width="650" height="60%"  alt="iEDA_operation" />
</div>

## **iEDA Tapeout**

<div align="center">
 <img src= "docs/resources/iEDA-tapeout.png" width="550" height="55%"  alt="iEDA_tapeout" />
</div>

---

🎉 **News:**

Presentation in ISEDA-2023, May-09 2023, Nanjing, China, Session 2: [iEDA: An Open-Source Intelligent Physical lmplementation Toolkit and Library][ISEDA-2023-iEDA-url] \[[paper][iEDA-paper], [slides][iEDA-slides]\]，and Panel 4 [Is ML a real cure for modern design automation?][ISEDA-2023-panel4-url], and Panel 6 [Open-source EDA and Standards][ISEDA-2023-panel6-url]

---

## **Papers and Presentations**

- ISEDA 2023: iEDA：An Open-Source Intelligent Physical Implementation Toolkit and Library \[[paper][iEDA-paper], [slides][iEDA-slides]\]

# **iEDA User Guide**

Before designing chip by iEDA，you need to obtain the execution binary of iEDA.

If you need to modify iEDA through source code construction, please read in sequence.

You can also directly  use the latest release docker， then you can skip "1. Source code construction of iEDA".

PS: about how to install Docker，you can refer [Docker install and initilization](https://www.cnblogs.com/harrypotterisdead/p/17223606.html)。

## 1. Build iEDA from source code

We provide two methods for source code construction of iEDA as examples.


### Method 1. Using  the iEDA mirror (Redommended)

Download the latest “iedaopensource/base” mirror from Dockerhub, which includes the latest master branch code and dependencies (build tools and dependency libraries). You can also use the ‘-v’ command to mount your own downloaded iEDA code repository. The build will only use the compile tools and dependency libraries provided by the mirror.


he current directory after entering the container is the iEDA master branch code, refering the following commands:

```bash
# iedaopensource/base:(latest, ubuntu, debian)
docker run -it --rm iedaopensource/base:latest bash 
# 进入容器后执行 build.sh 进行构建
bash build.sh
# 若能够正常输出 "Hello iEDA!" 则编译成功
./bin/iEDA -script scripts/hello.tcl
```

根据个人使用习惯，有 ubuntu（基于Ubuntu20.04）和 debian（基于Debian11）两种不同镜像tag可选。

### Method 2. 手动安装依赖并编译

在 Ubuntu 20.04 下执行如下命令：

```bash
# 下载iEDA仓库
git clone https://gitee.com/oscc-project/iEDA.git iEDA && cd iEDA
# 通过apt安装编译依赖，需要root权限
sudo bash build.sh -i apt
# 编译 iEDA
bash build.sh
# 若能够正常输出 "Hello iEDA!" 则编译成功
./bin/iEDA -script scripts/hello.tcl
```

## 2. 使用 iEDA 完成芯片设计

这里提供两种 iEDA 的运行方法作为参考。

关于 iEDA 的使用，参考 [Tcl 命令手册][Tcl-menu-xls] 和 `src/operation` 下各工具的说明文档readme。

### Method 1. release 或者 demo 镜像运行（推荐）

若需要使用自定义的工艺和设计，可将相关的文件挂载到容器中运行。关于目录结构和相关配置文件，可参考 `scripts/sky130` 中的示例。

```
docker run -it -v ${工艺和设计目录}:${容器内目录} --rm iedaopensource/release:latest
```

### Method 2.  自行创建文件运行

参考 `scripts/sky130` 中的文件目录格式，添加 iEDA 可执行文件路径到系统$PATH变量，运行 `sh run_iEDA.sh`，在 `result` 文件夹中查看运行结果。

```
iEDA/scripts/sky130
├── common        # common scripts
├── iEDA_config   # iEDA parameters configuration files
├── lef           # lef files
├── lib           # lib files
├── result        # iEDA result output files
├── script        # Tcl script files
├── sdc           # sdc files
├── run_iEDA.py   # Python3 script for running iEDA
└── run_iEDA.sh   # POSIX shell script for running iEDA
```

<!-- # 未来路线图

Roadmap -->

## **Contribution Guide**

Fork 此 iEDA 仓库，修改代码后提交 [Pull Request](https://gitee.com/oscc-project/iEDA/pulls)。

Please note the using [Coding Style][Code-conduct-md] of iEDA。

## **Discussion**

- Create [issue](https://gitee.com/oscc-project/iEDA/issues) in repo
- QQ Group：**793409748**
- WeChat Group：

<div align="center">
 <img src="docs/resources/WeChatGroup.png" width="20%" height="20%" alt="微信讨论群" />
</div>

## **License**

[木兰宽松许可证, 第2版][License-url]

## 致谢

在iEDA的开发过程中，我们采用了来自开源社区的子模块。具体情况如下：

| 子模块     | 来源                                                                                                  | 详细用途                                                          |
| ---------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| flute3     | [FastRoute](http://home.eng.iastate.edu/~cnchu/FastRoute)                                                | 借助flute3来产生rectange steiner tree.                            |
| abseil-cpp | [Google abseil](https://github.com/abseil/abseil-cpp.git)                                                | 使用Google的高性能C++容器和算法库提升性能，相比STL会改进程序性能. |
| json       | [JSON for Modern C++](https://github.com/nlohmann/json)                                                  | Json C++库，用来解析程序Json配置文件.                             |
| magic_enum | [Static reflection for enums (to string, from string, iteration)](https://github.com/Neargye/magic_enum) | 支持 enum 值和字符串的相互转换.                                   |
| libfort    | [Seleznev Anton libfort](https://github.com/seleznevae/libfort.git)                                      | C/C++ library 产生格式化的 ASCII tables.                          |
| pegtl      | [PEGTL（Parsing Expression Grammar Template Library）](https://github.com/taocpp/PEGTL/)                 | 使用PEGTL来方便的解析SPEF文件.                                    |
| pybind11   | [pybind 11](https://github.com/pybind/pybind11.git)                                                      | 方便python调用C++.                                                |
| VCDParser  | [ben-marshall verilog-vcd-parser](https://github.com/ben-marshall/verilog-vcd-parser.git)                | 解析功耗VCD波形文件.                                              |
| def lef    | [def lef parser](https://github.com/asyncvlsi/lefdef.git)                                                | 解析物理设计DEF/LEF文件.                                          |
| ThreadPool | [Jakob Progsch, Václav Zeman threadpool](https://github.com/progschj/ThreadPool.git)                    | C++11模板库实现的多线程池.                                        |
| fft        | [ fft](https://github.com/progschj/ThreadPool.git)                                                       | 快速傅立叶变换库.                                                 |
| hmetics    | [hmetics](https://github.com/progschj/ThreadPool.git)                                                    | 高效的图划分算法.                                                 |

我们深深地感谢来自开源社区的支持，我们也鼓励其他开源项目在[木兰宽松许可证](LICENSE)的范围下复用我们的代码。

<!-- links -->

<!-- [README-CN-path]: README.zh-cn.md -->

<!-- links -->

[License-icon]: https://s2.d2scdn.com/static/imgs/favicon.ico
[License-img]: docs/resources/license-Mulan%20PSL%20v2-blue.svg
[License-url]: LICENSE
[README-path]: README-En.md
[README-CN-path]: README.md
[Code-conduct-md]: docs/tbd/CodeConduct.md
[Tcl-menu-xls]: docs/tbd/TclMenu.xls
[iEDA-OSCC-url]: https://ieda.oscc.cc/
[ISEDA-2023-iEDA-url]: https://www.eda2.com/conferenceHome/program/detail?key=s2
[ISEDA-2023-panel4-url]: https://www.eda2.com/conferenceHome/program/detail?key=panel4
[ISEDA-2023-panel6-url]: https://www.eda2.com/conferenceHome/program/detail?key=panel6
[iEDA-paper]: docs/paper/ISEDA'23-iEDA-final.pdf
[iEDA-slides]: docs/ppt/ISEDA'23-iEDA-lxq-v8.pptx
[iFlow-repo-url]: https://gitee.com/oscc-project/iFlow
