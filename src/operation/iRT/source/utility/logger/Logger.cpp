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
#include "Logger.hpp"

namespace irt {

// public

void Logger::initInst()
{
  if (_log_instance == nullptr) {
    _log_instance = new Logger();
  }
}

Logger& Logger::getInst()
{
  if (_log_instance == nullptr) {
    initInst();
  }
  return *_log_instance;
}

void Logger::destroyInst()
{
  if (_log_instance != nullptr) {
    delete _log_instance;
    _log_instance = nullptr;
  }
}

// function

void Logger::openLogFileStream(const std::string& log_file_path)
{
  _log_file_path = log_file_path;
  _log_file = new std::ofstream(_log_file_path);
}

void Logger::closeLogFileStream()
{
  if (_log_file != nullptr) {
    _log_file->close();
    delete _log_file;
  }
}

void Logger::printLogFilePath()
{
  if (!_log_file_path.empty()) {
    info(Loc::current(), "The log file path is '", _log_file_path, "'!");
  }
}

// private

Logger* Logger::_log_instance = nullptr;

}  // namespace irt
