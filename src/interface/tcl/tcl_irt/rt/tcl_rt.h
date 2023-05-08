#pragma once

#include "tcl_util.h"

namespace tcl {

class TclDestroyRT : public TclCmd
{
 public:
  explicit TclDestroyRT(const char* cmd_name);
  ~TclDestroyRT() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;
};

class TclInitRT : public TclCmd
{
 public:
  explicit TclInitRT(const char* cmd_name);
  ~TclInitRT() override = default;

  unsigned check() override { return 1; };
  
  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};
class TclRunRT : public TclCmd
{
 public:
  explicit TclRunRT(const char* cmd_name);
  ~TclRunRT() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

}  // namespace tcl
