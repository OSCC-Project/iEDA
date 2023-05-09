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
#include "EvalLog.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>

namespace eval {

/**
 * @description: The SIGSEGV signal handle.
 * @param {char*} data
 * @param {int} size
 * @return {*}
 */
void SignalHandle(const char* data, int size)
{
  std::ofstream fs("glog_dump.log", std::ios::app);
  std::string str = std::string(data, size);
  fs << str;
  fs.close();
  LOG_ERROR << str;
}

/**
 * @description: The init of log module.
 * @param {char*} argv, The gflag config from main function.
 * @return {*}
 */
void Log::init(char* argv[])
{
  /*init google logging.*/
  google::InitGoogleLogging(argv[0]);

  /*config the log path.*/
  std::string home = "/var/tmp/Eval/";

  makeSureDirectoryExist(home);

  std::string info_log = home + "info_";
  google::SetLogDestination(google::INFO, info_log.c_str());

  std::string warning_log = home + "warning_";
  google::SetLogDestination(google::WARNING, warning_log.c_str());

  std::string error_log = home + "error_";
  google::SetLogDestination(google::ERROR, error_log.c_str());

  std::string fatal_log = home + "fatal_";
  google::SetLogDestination(google::FATAL, fatal_log.c_str());

  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = true;

  /*print stack trace when received SIGSEGV signal. */
  google::InstallFailureSignalHandler();

  /*print core dump SIGSEGV signal*/
  google::InstallFailureWriter(&SignalHandle);
}

/**
 * @description: The end of log module.
 * @param {*}
 * @return {*}
 */
void Log::end()
{
  google::ShutdownGoogleLogging();
}

void Log::makeSureDirectoryExist(std::string directory_path)
{
  std::filesystem::create_directories(directory_path.c_str());
}
}  // namespace eval
