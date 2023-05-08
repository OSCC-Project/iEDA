/**
 * @file UserShell.hh
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-22
 */

#pragma once
#include <iostream>

#include "tcl/ScriptEngine.hh"

namespace ieda {
/**
 * @brief singleton UserShell
 *
 */
class UserShell
{
 public:
  UserShell(const UserShell&) = delete;
  UserShell(const UserShell&&) = delete;

  /**
   * @brief Get the Shell object (singleton)
   *
   * @return UserShell*
   */
  static UserShell* getShell();

  /**
   * @brief set a callback initialization function.
   * It will be executed before userMain()
   *
   * @param initFunction int (*)() Note: the return value of
   * initFunction() need to be 0(EXIT_SUCCESS) or 1(EXIT_FAILURE)
   */
  static void set_init_func(int (*initFunction)());

  /**
   * @brief read and execute a Tcl script file, or open an interative tcl shell
   * @param file_path char* - If nullptr, it would open an interative tcl shell
   * @return 0(EXIT_SUCCESS) if succeed, 1(EXIT_FAILURE) if failed
   * @note If an initialization function has been set by set_init_func(), it
   * would invoke this initialization function before execution of tcl script
   * or start of the interative shell. Otherwise no initialization will be done
   */
  static int userMain(const char* file_path);

  static void displayHello(const std::string& hello_info) { std::cout << hello_info << std::endl; }

  /**
   * @brief display how to exit user shell
   *
   */
  static void displayHelp();

 private:
  UserShell() = default;
  ~UserShell() = default;
  static int initUserSetting(Tcl_Interp* interp) { return _user_init(); }

  static int (*_user_init)();
};

}  // namespace ieda