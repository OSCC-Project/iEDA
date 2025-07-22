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

namespace idrc {

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

// private

Logger* Logger::_log_instance = nullptr;

}  // namespace idrc
