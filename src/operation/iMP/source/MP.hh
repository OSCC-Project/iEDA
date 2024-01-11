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
#ifndef IMP_MACROPLACER_H
#define IMP_MACROPLACER_H
#include "Block.hh"
#include "ParserEngine.hh"
namespace imp {

class MP
{
 public:
  MP(ParserEngine* parser) : _parser(parser) { _root = _parser->get_design_ptr(); }
  ~MP() = default;
  Block& root() { return *_root; }
  const Block& root() const { return *_root; }
  std::shared_ptr<Block> root_ptr() { return _root; }
  void runMP();

 private:
  std::shared_ptr<Block> _root;
  std::unique_ptr<ParserEngine> _parser;
};
}  // namespace imp
#endif