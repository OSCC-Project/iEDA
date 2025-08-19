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
 * @file Log.cpp
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of log utility tool.
 * @version 0.1
 * @date 2020-11-12
 */

#include "Log.hh"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>

using std::string;


namespace ieda {

bool Log::_is_init = false;

/**
 * @brief The SIGSEGV signal handle.
 *
 * @param data
 * @param size
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
 * @brief The init of log module.
 *
 * @param argv The gflag config from main function.
 */
void Log::init(char* argv[], std::string log_dir)
{
  std::filesystem::create_directories(log_dir.c_str());
  // Check if glog is already initialized
  if (isInit()) {
    LOG_WARNING << "Google logging is already initialized, re-initialization to log dir: " << log_dir;
    end();
  }

  /*init google logging.*/
  google::InitGoogleLogging(argv[0]);

  string info_log = log_dir + "info_";
  google::SetLogDestination(google::INFO, info_log.c_str());

  string warning_log = log_dir + "warning_";
  google::SetLogDestination(google::WARNING, warning_log.c_str());

  string error_log = log_dir + "error_";
  google::SetLogDestination(google::ERROR, error_log.c_str());

  string fatal_log = log_dir + "fatal_";
  google::SetLogDestination(google::FATAL, fatal_log.c_str());

  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = true;

  /*print stack trace when received SIGSEGV signal. */
  google::InstallFailureSignalHandler();

  /*print core dump SIGSEGV signal*/
  google::InstallFailureWriter(&SignalHandle);

  set_is_init();
}

/**
 * @brief The end of log module.
 *
 */
void Log::end()
{
  google::ShutdownGoogleLogging();
}

/**
 * @brief Set verbose log level.
 *
 * @param module_name
 * @param level
 */
void Log::setVerboseLogLevel(const char* module_name, int level)
{
  google::SetVLOGLevel(module_name, level);
}

void Log::makeSureDirectoryExist(std::string directory_path)
{
  std::filesystem::create_directories(directory_path.c_str());
}

}  // namespace ieda
