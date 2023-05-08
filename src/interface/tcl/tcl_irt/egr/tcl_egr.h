#pragma once

#include "tcl_util.h"

namespace tcl {

class TclRunEGR : public TclCmd
{
 public:
  explicit TclRunEGR(const char* cmd_name);
  ~TclRunEGR() override = default;

  unsigned check() override { return 1; };

  unsigned exec() override;

 private:
  std::vector<std::pair<std::string, ValueType>> _config_list;
};

}  // namespace tcl
