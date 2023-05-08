#pragma once
/**
 * @File Name: tcl_main.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include <chrono>
#include <thread>

#include "UserShell.hh"
#include "tool_manager.h"
#include "tcl_register.h"

namespace tcl {

int tcl_start(char* path = nullptr)
{
#ifdef BUILD_GUI
  iplf::tmInst->guiInit();
#endif

  // this_thread::sleep_for(chrono::seconds(10));

  auto shell = ieda::UserShell::getShell();
  shell->set_init_func(registerCommands);
  shell->userMain(path);

  return 0;
}

}  // namespace tcl