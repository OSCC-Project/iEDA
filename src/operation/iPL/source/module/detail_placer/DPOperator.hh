#ifndef IPL_DPOPERATOR_H
#define IPL_DPOPERATOR_H

#include <string>

#include "GridManager.hh"
#include "TopologyManager.hh"
#include "data/Rectangle.hh"
#include "database/DPDatabase.hh"

namespace ipl {

class DPOperator
{
 public:
  DPOperator();

  DPOperator(const DPOperator&) = delete;
  DPOperator(DPOperator&&) = delete;
  ~DPOperator();

  DPOperator& operator=(const DPOperator&) = delete;
  DPOperator& operator=(DPOperator&&) = delete;

  TopologyManager* get_topo_manager() const { return _topo_manager; }
  GridManager* get_grid_manager() const { return _grid_manager; }

  void initDPOperator(DPDatabase* database);
  void updateTopoManager();
  void updateGridManager();

  std::pair<int32_t, int32_t> obtainOptimalXCoordiLine(DPInstance* inst);
  std::pair<int32_t, int32_t> obtainOptimalYCoordiLine(DPInstance* inst);
  Rectangle<int32_t> obtainOptimalCoordiRegion(DPInstance* inst);

  int64_t calInstAffectiveHPWL(DPInstance* inst);
  int64_t calInstPairAffectiveHPWL(DPInstance* inst_1, DPInstance* inst_2);

  bool checkIfClustered();
  void updateInstClustering();
  void pickAndSortMovableInstList(std::vector<DPInstance*>& movable_inst_list);
  DPCluster* createClsuter(DPInstance* inst, DPInterval* interval);

  bool checkOverlap(int32_t boundary_min, int32_t boundary_max, int32_t query_min, int32_t query_max);
  bool checkInNest(Rectangle<int32_t>& inner_box, Rectangle<int32_t>& outer_box);
  Rectangle<int32_t> obtainOverlapRectangle(Rectangle<int32_t>& box_1, Rectangle<int32_t>& box_2);
  std::pair<int32_t, int32_t> obtainOverlapRange(int32_t boundary_min, int32_t boundary_max, int32_t query_min, int32_t query_max);
  bool checkInBox(int32_t boundary_min, int32_t boundary_max, int32_t query_min, int32_t query_max);

  int64_t calTotalHPWL();

 private:
  DPDatabase* _database;
  TopologyManager* _topo_manager;
  GridManager* _grid_manager;

  void initTopoManager();
  void initGridManager();
  void initGridManagerFixedArea();
  bool isCoreOverlap(DPInstance* inst);
  void cutOutShape(Rectangle<int32_t>& shape);
};
}  // namespace ipl

#endif