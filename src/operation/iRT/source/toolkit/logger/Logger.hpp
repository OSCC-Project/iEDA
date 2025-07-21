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

#include <thread>

#include "LogLevel.hpp"
#include "RTHeader.hpp"

namespace irt {

using Loc = std::experimental::source_location;

#define RTLOG (irt::Logger::getInst())

class Logger
{
 public:
  static void initInst();
  static Logger& getInst();
  static void destroyInst();
  // function
  void openLogFileStream(const std::string& log_file_path)
  {
    _log_file_path = log_file_path;
    _log_file = new std::ofstream(_log_file_path);
  }

  void closeLogFileStream()
  {
    if (_log_file != nullptr) {
      _log_file->close();
      delete _log_file;
    }
  }

  void printLogFilePath()
  {
    if (!_log_file_path.empty()) {
      info(Loc::current(), "The log file path is '", _log_file_path, "'!");
    }
  }

  template <typename T, typename... Args>
  void info(Loc location, const T& value, const Args&... args)
  {
    printLog(LogLevel::kInfo, location, value, args...);
  }

  template <typename T, typename... Args>
  void warn(Loc location, const T& value, const Args&... args)
  {
    printLog(LogLevel::kWarn, location, value, args...);
  }

  template <typename T, typename... Args>
  void error(Loc location, const T& value, const Args&... args)
  {
    printLog(LogLevel::kError, location, value, args...);
    closeLogFileStream();
    exit(0);
  }

 private:
  // self
  static Logger* _log_instance;
  // config & database
  std::string _log_file_path;
  std::ofstream* _log_file = nullptr;
  std::vector<std::string> _temp_log_list;

  Logger() = default;
  Logger(const Logger& other) = delete;
  Logger(Logger&& other) = delete;
  ~Logger() { closeLogFileStream(); }
  Logger& operator=(const Logger& other) = delete;
  Logger& operator=(Logger&& other) = delete;
  // function
  template <typename T, typename... Args>
  void printLog(LogLevel log_level, Loc location, const T& value, const Args&... args)
  {
    const char* log_color_start;
    const char* log_level_char;

    switch (log_level) {
      case LogLevel::kInfo:
        log_color_start = "\033[1;34m";
        log_level_char = "Info";
        break;
      case LogLevel::kWarn:
        log_color_start = "\033[1;33m";
        log_level_char = "Warn";
        break;
      case LogLevel::kError:
        log_color_start = "\033[1;31m";
        log_level_char = "Error";
        break;
      default:
        log_color_start = "\033[1;32m";
        log_level_char = "None";
        break;
    }
    const char* log_color_end = "\033[0m";

    std::string file_name = std::filesystem::absolute(getString(location.file_name(), ":", location.line()));
    file_name.erase(remove(file_name.begin(), file_name.end(), '\"'), file_name.end());
    if (log_level != LogLevel::kError) {
      std::string::size_type pos = file_name.find_last_of('/') + 1;
      file_name = file_name.substr(pos, file_name.length() - pos);
    }
    std::string prefix = getString("[RT ", getTimestamp(), " ", getCompressedBase62(std::stoul(getString(std::this_thread::get_id()))), " ", file_name, " ");
    std::string suffix = getString(" ", location.function_name());
    std::string message = getString(value, args...);

    std::string origin_log = getString(prefix, log_level_char, suffix, "] ", message, "\n");
    std::string color_log = getString(prefix, log_color_start, log_level_char, log_color_end, suffix, "] ", message, "\n");

    if (_log_file != nullptr) {
      if (!_temp_log_list.empty()) {
        for (std::string& temp_log : _temp_log_list) {
          pushStream(*_log_file, temp_log);
        }
        _temp_log_list.clear();
      }
      pushStream(*_log_file, origin_log);
      _log_file->flush();
    } else {
      _temp_log_list.push_back(origin_log);
    }
    pushStream(std::cout, color_log);
  }

  template <typename T, typename... Args>
  std::string getString(T value, Args... args)
  {
    std::stringstream oss;
    pushStream(oss, value, args...);
    std::string string = oss.str();
    oss.clear();
    return string;
  }

  template <typename Stream, typename T, typename... Args>
  void pushStream(Stream& stream, T t, const Args&... args)
  {
    stream << t;
    pushStream(stream, args...);
  }

  template <typename Stream, typename T>
  void pushStream(Stream& stream, T t)
  {
    stream << t;
  }

  std::string getTimestamp()
  {
    std::string timestamp;

    time_t now = time(nullptr);
    tm* t = localtime(&now);
    char* buffer = new char[32];
    strftime(buffer, 32, "%Y%m%d %H:%M:%S", t);
    timestamp = buffer;
    delete[] buffer;
    buffer = nullptr;
    return timestamp;
  }

  std::string getCompressedBase62(uint64_t origin)
  {
    std::string base = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::string result = "";
    while (origin != 0) {
      result.push_back(base[origin % base.size()]);
      origin /= base.size();
    }
    return result;
  }
};

}  // namespace irt
