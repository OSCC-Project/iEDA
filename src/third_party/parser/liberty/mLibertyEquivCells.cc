// OpenSTA, Static Timing Analyzer
// Copyright (c) 2021, Parallax Software, Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
/**
 * @file LibertyEquivCells.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of find the function equivalently cells.
 * @version 0.1
 * @date 2021-09-24
 */
#include "mLibertyEquivCells.hh"

namespace ista {

using std::max;

const size_t hash_init_value = 5381;

static size_t hashString(const char* str)
{
  size_t hash = hash_init_value;
  size_t length = strlen(str);
  for (size_t i = 0; i < length; i++) {
    hash = ((hash << 5) + hash) ^ str[i];
  }
  return hash;
}

static unsigned hashCell(LibertyCell* cell);
static unsigned hashCellPorts(LibertyCell* cell);
// static unsigned hashCellSequentials(LibertyCell *cell);
static unsigned hashFuncExpr(LibertyExpr* expr);
static unsigned hashPort(LibertyPort* port);

static bool equiv(LibertyPort* port1, LibertyPort* port2)
{
  return (port1 == nullptr && port2 == nullptr)
         || (port1 != nullptr && port2 != nullptr && Str::equal(port1->get_port_name(), port2->get_port_name())
             && port1->get_port_type() == port2->get_port_type());
}

static bool equiv(LibertyArcSet* set1, LibertyArcSet* set2)
{
  return Str::equal(set1->front()->get_src_port(), set2->front()->get_src_port())
         && Str::equal(set1->front()->get_snk_port(), set2->front()->get_snk_port())
         && set1->front()->get_timing_type() == set2->front()->get_timing_type();
}

static bool equiv(LibertyExpr* expr1, LibertyExpr* expr2)
{
  if (expr1 == nullptr && expr2 == nullptr) {
    return true;
  }

  if (expr1 != nullptr && expr2 != nullptr && expr1->get_op() == expr2->get_op()) {
    switch (expr1->get_op()) {
      case LibertyExpr::Operator::kBuffer:
        return Str::equal(expr1->get_port(), expr2->get_port());
      case LibertyExpr::Operator::kNot:
        return equiv(expr1->get_left(), expr2->get_left());
      default:
        return equiv(expr1->get_left(), expr2->get_left()) && equiv(expr1->get_right(), expr2->get_right());
    }
  }

  return false;
}

bool equivCells(LibertyCell* cell1, LibertyCell* cell2)
{
  return equivCellPortsAndFuncs(cell1, cell2) && equivCellTimingArcSets(cell1, cell2);
}

double cellDriveResistance(LibertyCell* cell)
{
  LibertyCellPortIterator port_iter(cell);
  while (port_iter.hasNext()) {
    auto* port = port_iter.next();
    if (port->isOutput()) {
      return port->driveResistance();
    }
  }
  return 0.0;
}

class CellDriveResistanceGreater
{
 public:
  bool operator()(LibertyCell* cell1, LibertyCell* cell2) const { return cellDriveResistance(cell1) > cellDriveResistance(cell2); }
};

static unsigned hashCell(LibertyCell* cell)
{
  return hashCellPorts(cell);
}

static unsigned hashCellPorts(LibertyCell* cell)
{
  unsigned hash = 0;
  LibertyCellPortIterator port_iter(cell);
  while (port_iter.hasNext()) {
    LibertyPort* port = port_iter.next();
    hash += hashPort(port);
    hash += hashFuncExpr(port->get_func_expr()) * 3;
  }
  return hash;
}

static unsigned hashPort(LibertyPort* port)
{
  return hashString(port->get_port_name()) * 3 + static_cast<int>(port->get_port_type()) * 5;
}

static unsigned hashFuncExpr(LibertyExpr* expr)
{
  if (!expr) {
    return 0;
  }

  switch (expr->get_op()) {
    case LibertyExpr::Operator::kBuffer:
      return hashString(expr->get_port()) * 17;
      break;
    case LibertyExpr::Operator::kNot:
      return hashFuncExpr(expr->get_left()) * 31;
      break;
    default:
      return (hashFuncExpr(expr->get_left()) + hashFuncExpr(expr->get_right())) * ((1 << static_cast<unsigned>(expr->get_op())) - 1);
  }
}

bool equivCellPortsAndFuncs(LibertyCell* cell1, LibertyCell* cell2)
{
  bool ret_value = true;
  if (cell1->get_num_port() != cell2->get_num_port()) {
    ret_value = false;
  } else {
    LibertyCellPortIterator port_iter1(cell1);
    while (port_iter1.hasNext()) {
      LibertyPort* port1 = port_iter1.next();
      const char* name = port1->get_port_name();
      LibertyPort* port2 = cell2->get_cell_port_or_port_bus(name);
      if (!(port2 && equiv(port1, port2) && equiv(port1->get_func_expr(), port2->get_func_expr()))) {
        ret_value = false;
      }
    }
  }
  return ret_value;
}

bool equivCellPorts(LibertyCell* cell1, LibertyCell* cell2)
{
  bool ret_value = true;
  if (cell1->get_num_port() != cell2->get_num_port()) {
    ret_value = false;
  } else {
    LibertyCellPortIterator port_iter1(cell1);
    while (port_iter1.hasNext()) {
      auto* port1 = port_iter1.next();
      const char* name = port1->get_port_name();
      auto* port2 = cell2->get_cell_port_or_port_bus(name);
      if (!(port2 && equiv(port1, port2))) {
        return false;
      }
    }
  }
  return ret_value;
}

bool equivCellTimingArcSets(LibertyCell* cell1, LibertyCell* cell2)
{
  bool ret_value = true;
  if (cell1->getCellArcSetCount() != cell2->getCellArcSetCount()) {
    ret_value = false;
  } else {
    LibertyCellTimingArcSetIterator set_iter1(cell1);
    while (set_iter1.hasNext()) {
      auto* set1 = set_iter1.next();
      auto set2 = cell2->findLibertyArcSet(set1->front()->get_src_port(), set1->front()->get_snk_port(), set1->front()->get_timing_type());
      if (!(set2 && equiv(set1, *set2))) {
        ret_value = false;
      }
    }
  }
  return ret_value;
}

LibertyEquivCells::LibertyEquivCells(std::vector<LibertyLibrary*>& equiv_libs, std::vector<LibertyLibrary*>& map_libs)
{
  LibertyCellHashMap hash_matches;
  for (auto* lib : equiv_libs) {
    findEquivCells(lib, hash_matches);
  }

  // Sort the equiv sets by drive resistance.
  for (auto* cell : _unique_equiv_cells) {
    auto* equivs = _equiv_cells[cell];
    std::stable_sort(equivs->begin(), equivs->end(), CellDriveResistanceGreater());
  }

  for (auto* lib : map_libs) {
    mapEquivCells(lib, hash_matches);
  }

  for (auto hash_match : hash_matches) {
    delete hash_match.second;
  }
}

LibertyEquivCells::~LibertyEquivCells()
{
  for (auto cell : _unique_equiv_cells) {
    delete _equiv_cells[cell];
  }
}

LibertyCellSeq* LibertyEquivCells::equivs(LibertyCell* cell)
{
  return _equiv_cells[cell];
}

// Use a comprehensive hash on cell properties to segregate
// cells into groups of potential matches.
void LibertyEquivCells::findEquivCells(LibertyLibrary* library, LibertyCellHashMap& hash_matches)
{
  LibertyCellIterator cell_iter(library);
  while (cell_iter.hasNext()) {
    LibertyCell* cell = cell_iter.next();
    if (!cell->isDontUse()) {
      unsigned hash = hashCell(cell);
      LibertyCellSeq* matches = hash_matches[hash];
      if (matches) {
        for (auto* match : *matches) {
          if (equivCells(match, cell)) {
            LibertyCellSeq* equivs = _equiv_cells[match];
            if (equivs == nullptr) {
              equivs = new LibertyCellSeq;
              equivs->push_back(match);
              _unique_equiv_cells.push_back(match);
              _equiv_cells[match] = equivs;
            }
            equivs->push_back(cell);
            _equiv_cells[cell] = equivs;
            break;
          }
        }
        matches->push_back(cell);
      } else {
        matches = new LibertyCellSeq;
        hash_matches[hash] = matches;
        matches->push_back(cell);
      }
    }
  }
}

// Map library cells to equiv cells.
void LibertyEquivCells::mapEquivCells(LibertyLibrary* library, LibertyCellHashMap& hash_matches)
{
  LibertyCellIterator cell_iter(library);
  while (cell_iter.hasNext()) {
    LibertyCell* cell = cell_iter.next();
    if (!cell->isDontUse()) {
      unsigned hash = hashCell(cell);
      LibertyCellSeq* matches = hash_matches[hash];
      if (matches) {
        for (auto match : *matches) {
          if (equivCells(match, cell)) {
            LibertyCellSeq* equivs = _equiv_cells[match];
            _equiv_cells[cell] = equivs;
            break;
          }
        }
      }
    }
  }
}

}  // namespace ista
