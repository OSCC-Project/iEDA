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

#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#include "StringUtils.hh"
#include "Types.hh"
#include "log/Log.hh"

namespace ircx {
namespace path {

inline auto absolute(const std::filesystem::path& path) -> std::filesystem::path
{
  return std::filesystem::absolute(path).lexically_normal();
}

inline auto resolve(const std::filesystem::path& base_dir, std::string_view raw_path) -> Str
{
  const Str trimmed_path = string::trim(raw_path);
  if (trimmed_path.empty()) {
    return "";
  }

  std::filesystem::path path(trimmed_path);
  if (path.is_relative()) {
    path = base_dir / path;
  }

  return path.lexically_normal().string();
}

inline auto file_exists(const std::filesystem::path& file, std::string_view field_name) -> bool
{
  const Str file_string = file.string();
  if (!string::require_non_empty(file_string, field_name)) {
    return false;
  }

  if (std::filesystem::exists(file)) {
    return true;
  }

  LOG_ERROR << "RCX file not found for " << field_name << ": " << file_string;
  return false;
}

inline auto ensure_dir(const std::filesystem::path& dir, std::string_view field_name) -> bool
{
  const Str dir_string = dir.string();
  if (!string::require_non_empty(dir_string, field_name)) {
    return false;
  }

  std::error_code ec;
  const bool exists = std::filesystem::exists(dir, ec);
  if (ec) {
    LOG_ERROR << "Failed to access RCX directory for " << field_name
              << " " << dir_string << ": " << ec.message();
    return false;
  }

  if (exists) {
    if (std::filesystem::is_directory(dir, ec)) {
      return true;
    }
    if (ec) {
      LOG_ERROR << "Failed to inspect RCX directory for " << field_name
                << " " << dir_string << ": " << ec.message();
      return false;
    }
    LOG_ERROR << "RCX path for " << field_name
              << " exists but is not a directory: " << dir_string;
    return false;
  }

  std::filesystem::create_directories(dir, ec);
  if (ec) {
    LOG_ERROR << "Failed to create RCX directory for " << field_name
              << " " << dir_string << ": " << ec.message();
    return false;
  }

  if (std::filesystem::is_directory(dir, ec)) {
    return true;
  }
  if (ec) {
    LOG_ERROR << "Failed to inspect created RCX directory for " << field_name
              << " " << dir_string << ": " << ec.message();
    return false;
  }

  LOG_ERROR << "Failed to create RCX directory for " << field_name
            << " " << dir_string;
  return false;
}

inline auto file_under_dir(const std::filesystem::path& dir, std::string_view stem, std::string_view extension) -> std::filesystem::path
{
  std::string file_name(stem);
  if (file_name.empty()) {
    file_name = "output";
  }
  if (!extension.empty() && extension.front() != '.') {
    file_name.push_back('.');
  }
  file_name.append(extension);
  return dir / file_name;
}

}  // namespace path
}  // namespace ircx
