# iRCX ICS55 SDK

本目录只放置 ecc-tools 侧的 CMake 集成配置。ICS55 专用 iRCX SDK 需要作为 CMake package 安装到外部 prefix，供 `ecc-tools` 通过 `find_package(IrcxIcs55 CONFIG REQUIRED)` 查找。

## CMake 查找方式

配置 ecc-tools 时通过 CMake CLI 指定 SDK prefix 或 package config 目录：

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/ircx_ics55_sdk_prefix
# 或
cmake -S . -B build \
  -DIrcxIcs55_DIR=/path/to/ircx_ics55_sdk_prefix/lib/cmake/IrcxIcs55
```

`CMakeLists.txt` 会定义 ecc-tools 内部兼容 target：

```cmake
ircx_ics55
```

`src/operation/iRCX/CMakeLists.txt` 已经包含：

```cmake
add_subdirectory(ics55)
```

## 在 iRCX 中调用

需要使用该 SDK 的 target 链接：

```cmake
target_link_libraries(your_target
  PRIVATE
    ircx_ics55
)
```

C/C++ 代码中调用：

```c
#include "ircx_ics55.h"

if (!ircx_ics55_extract("/path/to/rcx.json")) {
  const char* error = ircx_ics55_last_error();
}
```

如果需要拆开执行：

```c
ircx_ics55_init("/path/to/rcx.json");
ircx_ics55_run();
ircx_ics55_report();
```

所有接口返回非 0 表示成功，返回 0 表示失败。

## rcx.json

内置 ICS55 corner 模式下，只保留：

```json
{
  "thread_num": 64,
  "output": "/RCX_ecc/output"
}
```

不要写 `mapping_file` 和 `corners`。如果写了这些字段，库会切换到外部文件模式，要求外部 mapping 和 corner 文件存在。

## 运行依赖

该 SDK 不依赖内部 iEDA/iRCX 动态库。glog、gflags、OpenMP/libgomp、libunwind、libcurl、Tcl、TBB、libstdc++、libgcc_s 等运行时依赖由 ecc-tools 的统一构建/运行环境提供。

`CMakeLists.txt` 会在配置阶段检查直接依赖是否存在；缺失时会 fail fast。

## 不能使用的情况

1. 调用方没有初始化 iEDA/idb 设计数据库时，`run`、`report`、`extract` 不能完成抽取。
2. 设计工艺、layer 名、routing 数据和内置 ICS55 mapping/corner 不匹配时不能使用。
3. 需要绝对防反编译时不能使用。当前库不暴露明文 corner 文件，也避免常规 `strings` 直接看到 ITF 内容，但不能防止运行时内存提取或高级逆向。
4. 不适合同一进程内并发跑多个 extraction job，内部仍有全局单例状态。
5. Windows、macOS 或 ABI 不兼容的 Linux 环境不能直接使用，需要重新构建。
6. 如果目标环境不是 Linux x86-64，或 glibc/动态加载器版本不兼容，程序不能正常加载该 `.so`。
