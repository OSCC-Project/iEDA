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
 * @file LibertyCompiler.hh
 * @author shy long (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2022-07-13
 */
#pragma once

#include "Liberty.hh"
#include "mLibertyEquivCells.hh"

namespace ista {
/**
 * @brief The top Liberty class, would provide the API to other tools.
 *
 */
class LibertyCompiler
{
 public:
  static LibertyCompiler* getOrCreateLibertyCompiler();
  static void destroyLibertyCompiler();

  void set_num_threads(unsigned num_thread) { _num_threads = num_thread; }
  [[nodiscard]] unsigned get_num_threads() const { return _num_threads; }

  unsigned readLiberty(const char* lib_file);
  unsigned readLiberty(const std::vector<const char*>& lib_files);

  LibertyCell* findLibertyCell(const char* cell_name);

  void addLib(std::unique_ptr<LibertyLibrary> lib)
  {
    std::unique_lock<std::mutex> lk(_mt);
    _libs.emplace_back(std::move(lib));
  }

  // get library name: getOneLib()-> get_lib_name() { return _lib_name; }
  LibertyLibrary* getOneLib() { return _libs.empty() ? nullptr : _libs.back().get(); }
  Vector<std::unique_ptr<LibertyLibrary>>& getAllLib() { return _libs; }

  // attributes
  std::string getLibTechnology(const char* lib_name);
  std::string getLibDelaymodel(const char* lib_name);
  std::string getLibDefaultwireload(const char* lib_name);
  std::string getLibDefaultwireloadmode(const char* lib_name);

  // equivCells
  void makeEquivCells(std::vector<LibertyLibrary*>& equiv_libs, std::vector<LibertyLibrary*>& map_libs);
  Vector<LibertyCell*>* equivCells(LibertyCell* cell);

 private:
  LibertyCompiler();
  ~LibertyCompiler();

  unsigned _num_threads;                            //!< The num of thread for propagation.
  Vector<std::unique_ptr<LibertyLibrary>> _libs;    //!< The design libs of different corners.
  std::unique_ptr<LibertyEquivCells> _equiv_cells;  //!< The function equivalently liberty cell.

  std::mutex _mt;

  // Singleton sta.
  static LibertyCompiler* _liberty_compiler;

  DISALLOW_COPY_AND_ASSIGN(LibertyCompiler);
};

}  // namespace ista