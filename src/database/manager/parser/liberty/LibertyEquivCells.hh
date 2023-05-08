/**
 * @file LibertyEquivCells.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Find the function equivalently cells.
 * @version 0.1
 * @date 2021-09-24
 */

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
