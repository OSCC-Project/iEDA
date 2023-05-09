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
 * @File Name: tcl_placer.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-05-08
 *
 * @last change :zhangmz 2022-12-1
 *
 */

#include <any>
#include <iostream>
#include <map>
#include <string>

#include "DrcAPI.hpp"
#include "ScriptEngine.hh"
#include "tcl_definition.h"

using ieda::TclCmd;
using ieda::TclOption;
using ieda::TclStringOption;

namespace tcl {

class CmdDRCAutoRun : public TclCmd
{
 public:
  explicit CmdDRCAutoRun(const char* cmd_name);
  ~CmdDRCAutoRun() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class TclCheckDrc : public TclCmd
{
 public:
  explicit TclCheckDrc(const char* cmd_name) : TclCmd(cmd_name){};
  ~TclCheckDrc() override = default;

  unsigned check() { return 1; };
  unsigned exec() override
  {
    if (!check()) {
      return 0;
    }
    std::cout << "init DRC ......" << std::endl;
    DrcInst.initDRC();

    std::cout << "init DRC check module ......" << std::endl;
    DrcInst.initCheckModule();
    std::cout << "run DRC check module ......" << std::endl;
    DrcInst.run();
    std::cout << "report check result ......" << std::endl;
    DrcInst.report();
    return 1;
  };
};

class TclInitDrcAPI : public TclCmd
{
 public:
  explicit TclInitDrcAPI(const char* cmd_name) : TclCmd(cmd_name){};
  ~TclInitDrcAPI() override = default;

  unsigned check() { return 1; };
  unsigned exec() override
  {
    if (!check()) {
      return 0;
    }
    idrc::DrcAPIInst.initDRC();
    return 1;
  };

 private:
  // private function
  // private data
};

class TclInitDrc : public TclCmd
{
 public:
  explicit TclInitDrc(const char* cmd_name) : TclCmd(cmd_name){};
  ~TclInitDrc() override = default;

  unsigned check() { return 1; };
  unsigned exec() override
  {
    if (!check()) {
      return 0;
    }
    DrcInst.initDRC();
    return 1;
  };

 private:
  // private function
  // private data
};

class TclDrcCheckDef : public TclCmd
{
 public:
  explicit TclDrcCheckDef(const char* cmd_name);
  ~TclDrcCheckDef() override = default;

  unsigned check() { return 1; };
  unsigned exec() override;

 private:
  // private function
  bool initConfigMapByTCL(std::map<std::string, std::any>& config_map);
  void addOptionForTCL();

  // private data
};

class TclDestroyDrcAPI : public TclCmd
{
 public:
  explicit TclDestroyDrcAPI(const char* cmd_name) : TclCmd(cmd_name){};
  ~TclDestroyDrcAPI() override = default;

  unsigned check() { return 1; };
  unsigned exec() override
  {
    if (!check()) {
      return 0;
    }
    DrcInst.destroyInst();
    return 1;
  };

  // private data
};

class TclDestroyDrc : public TclCmd
{
 public:
  explicit TclDestroyDrc(const char* cmd_name) : TclCmd(cmd_name){};
  ~TclDestroyDrc() override = default;

  unsigned check() { return 1; };
  unsigned exec() override
  {
    if (!check()) {
      return 0;
    }
    DrcInst.destroyInst();
    return 1;
  };

  // private data
};

class TclDrcCheckNet : public TclCmd
{
 public:
  explicit TclDrcCheckNet(const char* cmd_name);
  ~TclDrcCheckNet() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class TclDrcCheckAllNet : public TclCmd
{
 public:
  explicit TclDrcCheckAllNet(const char* cmd_name);
  ~TclDrcCheckAllNet() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdDRCSaveDetailFile : public TclCmd
{
 public:
  explicit CmdDRCSaveDetailFile(const char* cmd_name);
  ~CmdDRCSaveDetailFile() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

class CmdDRCReadDetailFile : public TclCmd
{
 public:
  explicit CmdDRCReadDetailFile(const char* cmd_name);
  ~CmdDRCReadDetailFile() override = default;

  unsigned check() override;
  unsigned exec() override;

 private:
  // private function
  // private data
};

}  // namespace tcl