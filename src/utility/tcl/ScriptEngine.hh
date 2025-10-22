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
 * @file ScriptEnginer.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the class of the script engine based on tcl.
 * @version 0.1
 * @date 2020-11-18
 */

#pragma once

#if __has_include(<tcl8.6/tcl.h>)
  #include <tcl8.6/tcl.h>
#else
  #include <tcl.h>
#endif

#include <memory>
#include <mutex>

#include "Vector.hh"
#include "log/Log.hh"
#include "string/Str.hh"
#include "string/StrMap.hh"

namespace ieda {

bool matchWildcardWithtarget(const char* const pattern, const char* const target);
bool containWildcard(const char* pattern);
/**
 * @brief The ScriptEngine is used for tcl file process such as sdc file.
 *
 */
class ScriptEngine
{
 public:
  static ScriptEngine* getOrCreateInstance();
  static void destroyInstance();

  Tcl_Interp* get_interp() { return _interp; }

  Tcl_Command createCmd(const char* cmd_name, Tcl_ObjCmdProc* proc, void* cmd_data = nullptr, Tcl_CmdDeleteProc* delete_proc = nullptr);

  int evalScriptFile(const char* file_name);
  int evalString(const char* cmd_str);

  const char* getTclFileName();
  unsigned getTclLineNo();

  void setResult(char* result);
  void appendResult(char* result);
  const char* getResult();

 private:
  ScriptEngine();
  virtual ~ScriptEngine();

  ScriptEngine(const ScriptEngine&) = delete;
  ScriptEngine& operator=(const ScriptEngine&) = delete;

  static ScriptEngine* _instance;  //!< The singleton instance.
  Tcl_Interp* _interp;             //!< The tcl interpreter.
};

/**
 * @brief The tcl option base class.
 *
 */
class TclOption
{
 public:
  TclOption(const char* option_name, unsigned is_arg);
  virtual ~TclOption();

  const char* get_option_name() { return _option_name; }
  unsigned is_arg() const { return _is_arg; }

  virtual unsigned isSwitchOption() { return 0; }
  virtual unsigned isDoubleOption() { return 0; }
  virtual unsigned isStringOption() { return 0; }
  virtual unsigned isIntOption() { return 0; }

  virtual unsigned isDoubleListOption() { return 0; }
  virtual unsigned isStringListOption() { return 0; }
  virtual unsigned isIntListOption() { return 0; }

  virtual unsigned isStringListListOption() { return 0; }

  virtual double getDoubleVal()
  {
    LOG_FATAL << "The option do not has float val.";
    return 0.0;
  }

  virtual double getDefaultDoubleVal()
  {
    LOG_FATAL << "The option do not has float val.";
    return 0.0;
  }

  virtual char* getStringVal()
  {
    LOG_FATAL << "The option do not has string val";
    return nullptr;
  }

  virtual char* getDefaultStringVal()
  {
    LOG_FATAL << "The option do not has string val.";
    return nullptr;
  }

  virtual bool getSwitchVal()
  {
    LOG_FATAL << "The option do not has switch val.";
    return 0;
  }

  virtual int getIntVal()
  {
    LOG_FATAL << "The option do not has int val.";
    return 0;
  }

  virtual int getDefaultIntVal()
  {
    LOG_FATAL << "The option do not has int val.";
    return 0;
  }

  virtual std::vector<int> getIntList()
  {
    LOG_FATAL << "The option do not has int list.";
    return {};
  }

  virtual std::vector<int> getDefaultIntList()
  {
    LOG_FATAL << "The option do not has int list.";
    return {};
  }

  virtual std::vector<double> getDoubleList()
  {
    LOG_FATAL << "The option do not has double list.";
    return {};
  }

  virtual std::vector<double> getDefaultDoubleList()
  {
    LOG_FATAL << "The option do not has double list.";
    return {};
  }

  virtual std::vector<std::string> getStringList()
  {
    LOG_FATAL << "The option do not has string list.";
    return {};
  }

  virtual std::vector<std::string> getDefaultStringList()
  {
    LOG_FATAL << "The option do not has string list.";
    return {};
  }

  virtual std::vector<std::vector<std::string>> getStringListList()
  {
    LOG_FATAL << "The option do not has string list list.";
    return {};
  }

  virtual std::vector<std::vector<std::string>> getDefaultStringListList()
  {
    LOG_FATAL << "The option do not has string list list.";
    return {};
  }

  virtual void setVal(const char* /*val*/) { LOG_FATAL << "The option can not set float val."; }

  virtual void resetVal() { LOG_FATAL << "The option has not reset value."; }

  unsigned is_set_val() { return _is_set_val; }

 protected:
  unsigned _is_set_val = 0;

 private:
  const char* _option_name;
  unsigned _is_arg;
};

/**
 * @brief The tcl switch option.
 *
 */
class TclSwitchOption : public TclOption
{
 public:
  explicit TclSwitchOption(const char* option_name);
  ~TclSwitchOption() override;

  unsigned isSwitchOption() override { return 1; }

  void setVal(const char* /*val*/) override { _is_set_val = 1; }

  void resetVal() override { _is_set_val = 0; }
};

/**
 * @brief The tcl float option.
 *
 */
class TclDoubleOption : public TclOption
{
 public:
  TclDoubleOption(const char* option_name, unsigned is_arg, float default_val = 0.0);
  ~TclDoubleOption() override;

  unsigned isDoubleOption() override { return 1; }

  double getDoubleVal() override { return _is_set_val ? _val : _default_val; }
  void setVal(const char* val) override
  {
    _val = Str::toDouble(val);
    _is_set_val = 1;
  }
  double getDefaultDoubleVal() override { return _default_val; }

  void resetVal() override { _is_set_val = 0; }

 private:
  double _default_val = 0.0;
  double _val;
};

/**
 * @brief The tcl string option.
 *
 */
class TclStringOption : public TclOption
{
 public:
  TclStringOption(const char* option_name, unsigned is_arg, const char* default_val = nullptr);
  ~TclStringOption() override;

  unsigned isStringOption() override { return 1; }

  char* getStringVal() override { return _is_set_val ? (char*) (_val.c_str()) : _default_val; }
  void setVal(const char* val) override
  {
    _val = val;
    _is_set_val = 1;
  }
  void resetVal() override { _is_set_val = 0; }
  char* getDefaultStringVal() override { return _default_val; }

 private:
  char* _default_val = nullptr;
  std::string _val;
};

/**
 * @brief The tcl int option.
 *
 */
class TclIntOption : public TclOption
{
 public:
  TclIntOption(const char* option_name, unsigned is_arg, int default_val = 0) : TclOption(option_name, is_arg), _default_val(default_val) {}
  ~TclIntOption() override = default;

  unsigned isIntOption() override { return 1; }

  int getIntVal() override { return _is_set_val ? _val : _default_val; }
  void setVal(const char* val) override
  {
    _val = Str::toInt(val);
    _is_set_val = 1;
  }
  void resetVal() override { _is_set_val = 0; }
  int getDefaultIntVal() override { return _default_val; }

 private:
  int _default_val = 0;
  int _val;
};

/**
 * @brief The tcl int list option.
 *
 */
class TclIntListOption : public TclOption
{
 public:
  TclIntListOption(const char* option_name, unsigned is_arg, std::vector<int> default_val = {})
      : TclOption(option_name, is_arg), _default_val(default_val)
  {
  }
  ~TclIntListOption() override = default;

  unsigned isIntListOption() override { return 1; }

  std::vector<int> getIntList() override { return _is_set_val ? _val : _default_val; }
  void setVal(const char* val) override
  {
    _val = Str::splitInt(val, " ");
    _is_set_val = 1;
  }
  void resetVal() override
  {
    std::vector<int>().swap(_val);
    _is_set_val = 0;
  }
  std::vector<int> getDefaultIntList() override { return _default_val; }

 private:
  std::vector<int> _default_val = {};
  std::vector<int> _val;
};

/**
 * @brief The tcl string list option.
 *
 */
class TclStringListOption : public TclOption
{
 public:
  TclStringListOption(const char* option_name, unsigned is_arg, std::vector<std::string> default_val = {})
      : TclOption(option_name, is_arg), _default_val(default_val)
  {
  }
  ~TclStringListOption() override = default;

  unsigned isStringOption() override { return 1; }

  std::vector<std::string> getStringList() override { return _is_set_val ? _val : _default_val; }
  void setVal(const char* val) override
  {
    _val = Str::split(val, " ");
    _is_set_val = 1;
  }
  void resetVal() override
  {
    // clear string vector, and release memory(is it needed for string?)
    std::vector<std::string>().swap(_val);
    _is_set_val = 0;
  }
  std::vector<std::string> getDefaultStringList() override { return _default_val; }

 private:
  std::vector<std::string> _default_val = {};
  std::vector<std::string> _val;
};

/**
 * @brief The tcl double list option.
 *
 */
class TclDoubleListOption : public TclOption
{
 public:
  TclDoubleListOption(const char* option_name, unsigned is_arg, std::vector<double> default_val = {})
      : TclOption(option_name, is_arg), _default_val(default_val)
  {
  }
  ~TclDoubleListOption() override = default;

  unsigned isStringOption() override { return 1; }

  std::vector<double> getDoubleList() override { return _is_set_val ? _val : _default_val; }
  void setVal(const char* val) override
  {
    while (*val == '{' || *val == ' ') {
      val++;
    }

    _val = Str::splitDouble(val, " ");
    _is_set_val = 1;
  }
  void resetVal() override
  {
    std::vector<double>().swap(_val);
    _is_set_val = 0;
  }
  std::vector<double> getDefaultDoubleList() override { return _default_val; }

 private:
  std::vector<double> _default_val = {};
  std::vector<double> _val;
};

/**
 * @brief The tcl string list list option.
 * such as set_clock_group -group xxx -group xxx.
 *
 */
class TclStringListListOption : public TclOption
{
 public:
  using StrList = std::vector<std::string>;
  TclStringListListOption(const char* option_name, unsigned is_arg, std::vector<StrList>&& default_val = {});
  ~TclStringListListOption() override = default;

  unsigned isStringListListOption() override { return 1; }

  void setVal(const char* val) override;
  void resetVal() override
  {
    std::vector<StrList>().swap(_val);
    _is_set_val = 0;
  }

  std::vector<StrList> getStringListList() override { return _is_set_val ? _val : _default_val; }
  std::vector<StrList> getDefaultStringListList() override { return _default_val; }

 private:
  std::vector<StrList> _default_val;
  std::vector<StrList> _val;
};

/**
 * @brief The tcl cmd base class.
 *
 */
class TclCmd
{
 public:
  explicit TclCmd(const char* cmd_name);
  virtual ~TclCmd();

  const char* get_cmd_name() { return _cmd_name; }
  TclOption* getOptionOrArg(const char* option_name)
  {
    if (containWildcard(option_name)) {
      return findOptionWithWildcard(option_name);
    } else {
      if (auto it = _options.find(option_name); it != _options.end()) {
        return it->second.get();
      }
    }

    return nullptr;
  }
  void addOption(TclOption* option)
  {
    // The arg need keep order.
    if (option->is_arg()) {
      _args.push_back(option);
    }

    _options.emplace(option->get_option_name(), option);
  }

  TclOption* getArg(int index) { return (static_cast<int>(_args.size()) > index) ? _args[index] : nullptr; }

  void resetOptionArgValue();

  virtual unsigned printHelp() {
    LOG_FATAL << "This cmd has not define print help body.";
    return 0;
  }

  virtual unsigned check()
  {
    LOG_FATAL << "This cmd has not define check body.";
    return 0;
  }
  virtual unsigned exec()
  {
    LOG_FATAL << "This cmd has not define exe body.";
    return 0;
  }

 private:
  TclOption* findOptionWithWildcard(const char* option_name)
  {
    int match_times = 0;
    auto res = _options.end();
    for (auto it = _options.begin(); it != _options.end(); ++it) {
      if (matchWildcardWithtarget(option_name, it->second.get()->get_option_name())) {
        if (++match_times > 1) {
          LOG_ERROR << "invalid option wildcard(s), multiple options matched.";
          assert(0);
        }
        res = it;
      }
    }
    return res == _options.end() ? nullptr : res->second.get();
  }
  const char* _cmd_name;
  StrMap<std::unique_ptr<TclOption>> _options;  //!< The tcl option do not need keep order.
  Vector<TclOption*> _args;                     //!< The tcl arg need keep order.
};

/**
 * @brief The all tcl cmd container.
 *
 */
class TclCmds
{
 public:
  static void addTclCmd(std::unique_ptr<TclCmd> cmd);
  static TclCmd* getTclCmd(const char* cmd_name);

 private:
  static StrMap<std::unique_ptr<TclCmd>> _cmds;
};

/**
 * @brief Encode/decode tcl pointer result.
 *
 */
class TclEncodeResult
{
 public:
  static char* encode(void* pointer);
  static void* decode(const char* encode_str);
  static const char* get_encode_preamble() { return _encode_preamble; }

 private:
  inline static const char* _encode_preamble = "@ptr";
};

/**
 * @brief initialization -- register a user defined command
 * @param type-class type command class type (a derived classe of TclCmd)
 * @param name-const char* command's name
 */
#define registerTclCmd(type, name)               \
  do {                                           \
    auto cmd_ptr = std::make_unique<type>(name); \
    TclCmds::addTclCmd(std::move(cmd_ptr));      \
  } while (0)

}  // namespace ieda
