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
 * @file UserShell.hh
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-13
 */

#pragma once

#include <map>

namespace ista {

enum CMD_RET_CODE {
  CMD_SUCCESS,
  CMD_FAILURE,
};

enum SHELL_OPTION {
  SHELL_HELP,
  SHELL_FILE,
};

/**
 * @brief singleton UserShell
 *
 */
class UserShell {
 public:
  UserShell(const UserShell&) = delete;
  UserShell(const UserShell&&) = delete;

  static UserShell* getShell();

  int userMain(int argc, char** argv);
  int execCMD();
  int execFile(const char* file_path);

  int parseOptionArgs(int argc, char** argv);

  void initCommandList();
  void initShellLog(const char** argv);
  void displayHelp();
  void displayHello();
  void exitWithMessage(const char* message);

 private:
  UserShell() = default;
  ~UserShell() = default;
};

}  // namespace ista