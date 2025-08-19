// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once
/**
 * @Brief :
 */

namespace tcl {

#define TCL_HELP "-help"
#define TCL_JSON_OPTION "-discard"
#define TCL_CONFIG "-config"
#define TCL_PATH "-path"
#define TCL_DIRECTORY "-dir"
#define TCL_NAME "-name"
#define TCL_TYPE "-type"
#define EXCLUDE_CELL_NAMES "-exclude_cell_names"
#define TCL_OUTPUT_PATH "-output"
#define TCL_VERILOG_TOP "-top"
#define TCL_MAX_NUM "-max_num"
#define TCL_WORK_DIR "-work_dir"
#define TCL_STEP "-step"
#define TCL_ADD_SPACE "-add_space"
#define TCL_PATCH_ROW_STEP "-patch_row_step"
#define TCL_PATCH_COL_STEP "-patch_col_step"

const char* const EMPTY_STR = "";

#define DEFINE_CMD_CLASS(CLASSNAME)                \
  class Cmd##CLASSNAME : public TclCmd             \
  {                                                \
   public:                                         \
    explicit Cmd##CLASSNAME(const char* cmd_name); \
    ~Cmd##CLASSNAME() override = default;          \
    unsigned check() override;                     \
    unsigned exec() override;                      \
  }

#define CMD_CLASS_DEFAULT_DEFINITION(CLASSNAME)                 \
  CLASSNAME::CLASSNAME(const char* cmd_name) : TclCmd(cmd_name) \
  {                                                             \
  }                                                             \
  unsigned CLASSNAME::check()                                   \
  {                                                             \
    return 1;                                                   \
  }
#define CMD_CLASS_DEFAULT_EXEC(CLASSNAME, EXEC) \
  unsigned CLASSNAME::exec()                    \
  {                                             \
    if (not check()) {                          \
      return 0;                                 \
    }                                           \
    {                                           \
      EXEC;                                     \
    }                                           \
    return 1;                                   \
  }
}  // namespace tcl