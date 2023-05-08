#pragma once
/**
 * @Brief :
 */

namespace tcl {

#define TCL_HELP "-help"
#define TCL_CONFIG "-config"
#define TCL_PATH "-path"
#define TCL_DIRECTORY "-dir"
#define TCL_NAME "-name"
#define TCL_TYPE "-type"
#define EXCLUDE_CELL_NAMES "-exclude_cell_names"
#define TCL_OUTPUT_PATH "-output"
#define TCL_VERILOG_TOP "-top"
#define TCL_MAX_NUM "-max_num"

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