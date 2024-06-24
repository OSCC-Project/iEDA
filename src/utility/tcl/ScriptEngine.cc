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
/**
 * @file ScriptEngine.cpp
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the implementation of the script engine based on tcl.
 * @version 0.1
 * @date 2020-11-18
 */

#include "ScriptEngine.hh"

#include <cstring>
#include <utility>

namespace ieda {
ScriptEngine* ScriptEngine::_instance = nullptr;

ScriptEngine::ScriptEngine()
{
  _interp = Tcl_CreateInterp();
}
ScriptEngine::~ScriptEngine()
{
  Tcl_DeleteInterp(_interp);
}

/**
 * @brief Get the script engine or create one.
 *
 * @return ScriptEngine* The script engine.
 */
ScriptEngine* ScriptEngine::getOrCreateInstance()
{
  static std::mutex mt;
  if (_instance == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_instance == nullptr) {
      _instance = new ScriptEngine();
    }
  }
  return _instance;
}

/**
 * @brief Close the script engine.
 *
 */
void ScriptEngine::destroyInstance()
{
  delete _instance;
  _instance = nullptr;
}

/**
 * @brief Create a Cmd object
 *
 * @param cmd_name The user defined cmd name.
 * @param proc The cmd callback function.
 * @param cmd_Data The cmd data that will be copied to proc.
 * @param delete_proc The deleteProc will be invoked before the command is
 * deleted through a call to Tcl_DeleteCommand.
 * @return int
 */
Tcl_Command ScriptEngine::createCmd(const char* cmd_name, Tcl_ObjCmdProc* proc, void* cmd_data, Tcl_CmdDeleteProc* delete_proc)
{
  return Tcl_CreateObjCommand(_interp, cmd_name, proc, cmd_data, delete_proc);
}

/**
 * @brief Call the tcl interpreter to execuate the tcl file.
 *
 * @param file_name The script file.
 * @return int The return code.
 */
int ScriptEngine::evalScriptFile(const char* file_name)
{
  return Tcl_EvalFile(_interp, file_name);
}

/**
 * @brief Call the tcl interpreter to execuate the cmd string.
 *
 * @param cmd_str The cmd string.
 * @return int The return code.
 */
int ScriptEngine::evalString(const char* cmd_str)
{
  return Tcl_Eval(_interp, cmd_str);
}

const char* ScriptEngine::getTclFileName()
{
  evalString("set fileName [dict get [info frame 2] file]");
  const char* file_name = Tcl_GetVar(_interp, "fileName", 0);
  return file_name;
}

/**
 * @brief Get the current tcl line no.
 *
 * @return unsigned The tcl file line no.
 */
unsigned ScriptEngine::getTclLineNo()
{
  evalString("set lineNum [dict get [info frame 2] line]");
  const char* line_no = Tcl_GetVar(_interp, "lineNum", 0);
  return Str::toUnsigned(line_no);
}

void ScriptEngine::setResult(char* result)
{
  Tcl_SetResult(_interp, result, nullptr);
}

/**
 * @brief Append the cmd execuate result to tcl interpretr.
 *
 * @param result The cmd execuate result.
 */
void ScriptEngine::appendResult(char* result)
{
  Tcl_AppendResult(_interp, result, nullptr);
}

/**
 * @brief Get the result from interpreter.
 *
 * @return const char* The tcl result that is string format.
 */
const char* ScriptEngine::getResult()
{
  return Tcl_GetStringResult(_interp);
}

TclOption::TclOption(const char* option_name, unsigned is_arg) : _option_name(Str::copy(option_name)), _is_arg(is_arg)
{
}

TclOption::~TclOption()
{
  Str::free(_option_name);
  _option_name = nullptr;
}

TclSwitchOption::TclSwitchOption(const char* option_name) : TclOption(option_name, 0)
{
}

TclSwitchOption::~TclSwitchOption() = default;

TclDoubleOption::TclDoubleOption(const char* option_name, unsigned is_arg, float default_val)
    : TclOption(option_name, is_arg), _default_val(default_val)
{
}

TclDoubleOption::~TclDoubleOption() = default;

TclStringOption::TclStringOption(const char* option_name, unsigned is_arg, const char* default_val)
    : TclOption(option_name, is_arg), _default_val(Str::copy(default_val))
{
}

TclStringOption::~TclStringOption()
{
  Str::free(_default_val);
  _default_val = nullptr;
}

TclStringListListOption::TclStringListListOption(const char* option_name, unsigned is_arg, std::vector<StrList>&& default_val)
    : TclOption(option_name, is_arg), _default_val(std::move(default_val))
{
}

template <char c>
inline int IgnoreNext(const char val[], int pos = 0)
{
  static_assert(c != '\0');
  for (; val[pos] == c; ++pos)
    ;
  return pos;
}

template <char c>
inline int FindNext(const char val[], int pos = 0)
{
  for (; val[pos] != c; ++pos)
    if (val[pos] == '\0')
      break;
  return pos;
}

template <char c0, char c1, class Itr = std::string::iterator>
inline Itr IgnoreNext(Itr start)
{
  for (; *start == c0 || *start == c1; ++start)
    ;
  return start;
}

template <char c, class Itr = std::string::iterator>
inline Itr FindNext(Itr start, Itr end)
{
  for (; *start != c; ++start)
    if (start == end)
      break;
  return start;
}

inline std::vector<std::pair<int, int>> GetStrListPosLen(const char val[])
{
  std::vector<std::pair<int, int>> pos_len_list;
  int pos = 0;
  while (val[pos] != '\0') {
    int start = FindNext<'{'>(val, pos);
    int end = FindNext<'}'>(val, start);
    start = IgnoreNext<' '>(val, start + 1);
    pos = IgnoreNext<' '>(val, end + 1);
    pos_len_list.emplace_back(start, end - start);
  }
  return pos_len_list;
}

template <char delim0, char delim1>
inline TclStringListListOption::StrList Split(const char* val, int len)
{
  TclStringListListOption::StrList str_list;
  std::string str(val, len);
  auto substr_begin = str.begin();
  auto substr_end = str.begin();
  while (substr_end != str.end()) {
    switch (*substr_end) {
      case '\"':
        substr_begin = ++substr_end;
        substr_end = FindNext<'\"'>(substr_end, str.end());
        *substr_end = ' ';

      case delim0:
      case delim1:
        str_list.emplace_back(substr_begin, substr_end);
        substr_begin = substr_end = IgnoreNext<delim0, delim1>(substr_end);
        break;

      default:
        ++substr_end;
        break;
    }
  }
  if (substr_begin != substr_end) {
    str_list.emplace_back(substr_begin, substr_end);
  }
  return str_list;
}

/**
 * @brief set string list val.
 *
 * @param val
 */
void TclStringListListOption::setVal(const char* val)
{
  val += IgnoreNext<' '>(val);

  if (*val == '{') {
    auto pos_len_list = GetStrListPosLen(val);
    for (auto& [pos, len] : pos_len_list) {
      _val.emplace_back(Split<' ', ','>(val + pos, len));
    }
  } else {
    _val.emplace_back(Str::split(val, " "));
  }

  _is_set_val = 1;
}

TclCmd::TclCmd(const char* cmd_name) : _cmd_name(Str::copy(cmd_name))
{
}

TclCmd::~TclCmd()
{
  Str::free(_cmd_name);
  _cmd_name = nullptr;
}

/**
 * @brief Reset the option and arg value.
 *
 */
void TclCmd::resetOptionArgValue()
{
  for (auto& [option_name, option] : _options) {
    option->resetVal();
  }
}

StrMap<std::unique_ptr<TclCmd>> TclCmds::_cmds;

/**
 * @brief The tcl cmd process callback function.
 *
 * @param clientData The callback data, which transparent from regiester
 * function.
 * @param interp The tcl interp.
 * @param objc The tcl cmd option and arg num count.
 * @param objv The tcl cmd option and arg obj.
 * @return int The process result, success return TCL_OK, else return
 * TCL_ERROR.
 */
int CmdProc(ClientData clientData, Tcl_Interp* interp, int objc, struct Tcl_Obj* const* objv)
{
  const char* cmd_name = Tcl_GetString(objv[0]);
  TclCmd* cmd = TclCmds::getTclCmd(cmd_name);
  cmd->resetOptionArgValue();

  bool next_is_option_val = false;
  TclOption* curr_option = nullptr;
  int arg_index = 0;
  for (int cnt = 1; cnt < objc; ++cnt) {
    struct Tcl_Obj* obj = objv[cnt];
    // get option lead string or arg
    const char* obj_str = Tcl_GetString(obj);
    if (!next_is_option_val) {
      TclOption* option = cmd->getOptionOrArg(obj_str);
      curr_option = option;
      if (option) {
        if (!option->isSwitchOption()) {
          // It is option, next should be option value if it is not switch
          // option,
          next_is_option_val = true;
        } else {
          // switch option
          option->setVal(nullptr);
        }
      } else {
        // should be arg, arg is need keep order.
        TclOption* arg = cmd->getArg(arg_index);
        ++arg_index;
        if (!arg) {
          LOG_ERROR << "The cmd " << cmd->get_cmd_name() << " syntax has error.";
          return TCL_ERROR;
        }

        arg->setVal(obj_str);
      }
    } else {
      curr_option->setVal(obj_str);
      next_is_option_val = false;
    }
  }

  if (next_is_option_val) {
    LOG_ERROR << "The cmd syntax has error " << curr_option->get_option_name() << " need val.";
  }

  unsigned result = cmd->exec();
  return result ? TCL_OK : TCL_ERROR;
}

/**
 * @brief Registe the tcl cmd.
 *
 * @param cmd
 */
void TclCmds::addTclCmd(std::unique_ptr<TclCmd> cmd)
{
  ScriptEngine::getOrCreateInstance()->createCmd(cmd->get_cmd_name(), CmdProc, cmd.get());
  _cmds.emplace(cmd->get_cmd_name(), std::move(cmd));
}

/**
 * @brief Get tcl cmd accord to cmd name.
 *
 */
TclCmd* TclCmds::getTclCmd(const char* cmd_name)
{
  auto it = _cmds.find(cmd_name);
  if (it != _cmds.end()) {
    return it->second.get();
  }
  return nullptr;
}

/**
 * @brief Encode the pointer for transmit.
 *
 * @param pointer
 * @return char*
 */
char* TclEncodeResult::encode(void* pointer)
{
  return Str::printf("%s%p", _encode_preamble, pointer);
}

/**
 * @brief decode the encode string to pointer.
 *
 * @param encode_str
 */
void* TclEncodeResult::decode(const char* encode_str)
{
  std::string pointer_str = Str::stripPrefix(encode_str, _encode_preamble);
  const int hex = 16;
  auto pointer_address = static_cast<uintptr_t>(std::stoull(pointer_str, nullptr, hex));
  return reinterpret_cast<void*>(pointer_address);
}

bool containWildcard(const char* pattern)
{
  return (pattern[0] == '-') && (strpbrk(pattern, "*?") != nullptr);
}

bool matchWildcardWithtarget(const char* const pattern, const char* const target)
{
  const char* p = pattern;
  const char* t = target;

  while (1) {
    while (*p && *t && (*p == *t)) {
      ++p;
      ++t;
    }

    if (*p == '\0') {
      return (*t == '\0') ? true : false;
    }

    if (*t == '\0') {
      if (*p == '\0') {
        return true;
      } else {
        while (*p == '*') {
          ++p;
          if (*p == '\0') {
            return true;
          } else if (*p == '*') {
            continue;
          } else {
            return false;
          }
        }
      }
    }

    if (*p == '?') {
      ++p;
      ++t;
    } else if (*p == '*') {
      if (*(p + 1) == '\0') {
        return true;
      } else {
        if (*(p + 1) == *t) {
          ++p;
        } else {
          ++t;
        }
      }
    } else if (*p != *t) {
      return false;
    }
  }
}

}  // namespace ieda
