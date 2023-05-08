/*
 * @Author: S.J Chen
 * @Date: 2022-01-19 19:27:41
 * @LastEditTime: 2022-01-20 17:32:40
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/module/logger/Log.cc
 * Contact : https://github.com/sjchanson
 */

#include "Log.hh"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>

namespace ipl {

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
  std::string home = "./result/pl/log/";

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

}  // namespace ipl