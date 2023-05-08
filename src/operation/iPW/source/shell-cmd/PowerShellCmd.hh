/**
 * @file PowerShellCmd.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief the classes of ipower shell cmd.
 * @version 0.1
 * @date 2023-05-04
 */

#pragma once

#include "api/Power.hh"
#include "tcl/ScriptEngine.hh"

namespace ipower {

using ieda::ScriptEngine;
using ieda::TclCmd;
using ieda::TclOption;
using ieda::TclStringOption;

/**
 * @brief The class of read in a VCD file.
 *
 */
class CmdReadVcd : public TclCmd {
 public:
  explicit CmdReadVcd(const char* cmd_name);
  ~CmdReadVcd() override = default;

  unsigned check() override;
  unsigned exec() override;
};

/**
 * @brief report_power command reports power.
 *
 */
class CmdReportPower : public TclCmd {
 public:
  explicit CmdReportPower(const char* cmd_name);
  ~CmdReportPower() override = default;

  unsigned check() override;
  unsigned exec() override;
};

}  // namespace ipower
