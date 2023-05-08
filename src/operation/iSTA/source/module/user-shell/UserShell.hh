/**
 * @file UserShell.hh
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-13
 *
 * @copyright Copyright (c) 2021
 *
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