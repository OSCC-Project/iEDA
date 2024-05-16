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
#ifndef IMP_DBPARSER_H
#define IMP_DBPARSER_H

#include <memory>
#include <string>
#include <unordered_map>
namespace imp {
class Instance;
class Block;
class Row;
class Cell;

class ParserEngine
{
 public:
  ParserEngine() = default;
  ParserEngine(const ParserEngine&) = delete;
  ParserEngine(ParserEngine&&) = delete;
  virtual ~ParserEngine() = default;

  ParserEngine& operator=(const ParserEngine&) = delete;
  ParserEngine& operator=(ParserEngine&&) = delete;

  virtual bool read() = 0;
  virtual bool write() = 0;

  Block& get_design() { return *_design; }
  const Block& get_design() const { return *_design; }
  const std::unordered_map<std::string, std::shared_ptr<Cell>>& get_cells()const { return _cells;}
  const std::unordered_map<std::string, std::shared_ptr<Instance>>& get_instances()const { return _instances;}

  std::shared_ptr<Block> get_design_ptr() { return _design; }

 private:
  friend class IDBParser;
  std::shared_ptr<Block> _design;
  std::unordered_map<std::string, std::shared_ptr<Cell>> _cells;
  std::unordered_map<std::string, std::shared_ptr<Row>> _rows;
  std::unordered_map<std::string, std::shared_ptr<Instance>> _instances;
};

}  // namespace imp

#endif