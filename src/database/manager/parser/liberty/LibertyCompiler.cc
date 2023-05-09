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
 * @file LibertyCompiler.cc
 * @author shy long (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2022-07-14
 */
#include "LibertyCompiler.hh"

#include "ThreadPool/ThreadPool.h"

namespace ista {

LibertyCompiler* LibertyCompiler::_liberty_compiler = nullptr;

LibertyCompiler::LibertyCompiler()
{
}
LibertyCompiler::~LibertyCompiler() = default;

/**
 * @brief Get the LibertyCompiler instance, if not, create one.
 *
 * @return LibertyCompiler*
 */
LibertyCompiler* LibertyCompiler::getOrCreateLibertyCompiler()
{
  static std::mutex mt;
  if (_liberty_compiler == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_liberty_compiler == nullptr) {
      _liberty_compiler = new LibertyCompiler();
    }
  }
  return _liberty_compiler;
}

/**
 * @brief Destory the LibertyCompiler.
 *
 */
void LibertyCompiler::destroyLibertyCompiler()
{
  delete _liberty_compiler;
  _liberty_compiler = nullptr;
}

/**
 * @brief read one lib file.
 *
 * @param lib_file
 * @return unsigned
 */
unsigned LibertyCompiler::readLiberty(const char* lib_file)
{
  Liberty lib;
  auto load_lib = lib.loadLiberty(lib_file);
  addLib(std::move(load_lib));

  return 1;
}

/**
 * @brief read liberty files.
 *
 * @param lib_files
 * @return unsigned
 */
unsigned LibertyCompiler::readLiberty(const std::vector<const char*>& lib_files)
{
  LOG_INFO << "load lib start";

#if 0
  for (const auto *lib_file : lib_files) {
    readLiberty(lib_file);
  }

#else

  {
    ThreadPool pool(get_num_threads());

    for (const auto* lib_file : lib_files) {
      pool.enqueue([this, lib_file]() { readLiberty(lib_file); });
    }
  }

#endif

  LOG_INFO << "load lib end";

  return 1;
}

/**
 * @brief Find the liberty cell from the lib.
 *
 * @param cell_name
 * @return LibertyCell*
 */
LibertyCell* LibertyCompiler::findLibertyCell(const char* cell_name)
{
  LibertyCell* found_cell = nullptr;
  for (auto& lib : _libs) {
    if (found_cell = lib->findCell(cell_name); found_cell) {
      break;
    }
  }
  return found_cell;
}

/*
std::string LibertyCompiler::getLibTechnology(const char* lib_name) {
  for (auto& lib : _libs) {
    if (lib->get_lib_name().c_str() == lib_name) {
    }
  }
}
std::string LibertyCompiler::getLibDelaymodel(const char* lib_name) {
  for (auto& lib : _libs) {
    if (lib->get_lib_name().c_str() == lib_name) {
    }
  }
}
std::string LibertyCompiler::getLibDefaultwireload(const char* lib_name) {
  std::string default_wire_load;
  for (auto& lib : _libs) {
    if (lib->get_lib_name().c_str() == lib_name) {
      default_wire_load = lib->get_default_wire_load();
      break;
    }
  }
  return default_wire_load;
}

std::string LibertyCompiler::getLibDefaultwireloadmode(const char* lib_name) {
  for (auto& lib : _libs) {
    if (lib->get_lib_name().c_str() == lib_name) {
    }
  }
}*/

/**
 * @brief Make the function equivalently liberty cell map.
 *
 * @param equiv_libs
 * @param map_libs
 */
void LibertyCompiler::makeEquivCells(std::vector<LibertyLibrary*>& equiv_libs, std::vector<LibertyLibrary*>& map_libs)
{
  if (_equiv_cells) {
    _equiv_cells.reset();
  }

  _equiv_cells = std::make_unique<LibertyEquivCells>(equiv_libs, map_libs);
}

/**
 * @brief Get the equivalently liberty cell.
 *
 * @param cell
 * @return Vector<LibertyCell *>*
 */
Vector<LibertyCell*>* LibertyCompiler::equivCells(LibertyCell* cell)
{
  if (_equiv_cells)
    return _equiv_cells->equivs(cell);
  else
    return nullptr;
}

}  // namespace ista
