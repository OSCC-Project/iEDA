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

#pragma once

#include <unordered_map>

#include "Liberty.hh"
#include "Map.hh"
#include "Vector.hh"

namespace ista {

using LibertyCellSeq = Vector<LibertyCell*>;
using EquivCellMap = ieda::Map<LibertyCell*, LibertyCellSeq*>;
using LibertyCellHashMap = std::unordered_map<unsigned, LibertyCellSeq*>;

// Predicate that is true when the ports, functions, sequentials and
// timing arcs match.
bool equivCells(LibertyCell* cell1, LibertyCell* cell2);

// Predicate that is true when the ports match.
bool equivCellPorts(LibertyCell* cell1, LibertyCell* cell2);

// Predicate that is true when the ports and their functions match.
bool equivCellPortsAndFuncs(LibertyCell* cell1, LibertyCell* cell2);

// Predicate that is true when the timing arc sets match.
bool equivCellTimingArcSets(LibertyCell* cell1, LibertyCell* cell2);

/**
 * @brief
 *
 */
class LibertyEquivCells
{
 public:
  // Find equivalent cells in equiv_libs.
  // Optionally add mappings for cells in map_libs.
  LibertyEquivCells(std::vector<LibertyLibrary*>& equiv_libs, std::vector<LibertyLibrary*>& map_libs);
  ~LibertyEquivCells();
  // Find equivalents for cell (member of from_libs) in to_libs.
  LibertyCellSeq* equivs(LibertyCell* cell);

 protected:
  void findEquivCells(LibertyLibrary* library, LibertyCellHashMap& hash_matches);
  void mapEquivCells(LibertyLibrary* library, LibertyCellHashMap& hash_matches);

  EquivCellMap _equiv_cells;
  // Unique cell for each equiv cell group.
  LibertyCellSeq _unique_equiv_cells;
};

}  // namespace ista
