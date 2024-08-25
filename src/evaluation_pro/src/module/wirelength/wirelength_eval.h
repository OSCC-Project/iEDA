
#include "wirelength_db.h"

namespace ieval {

class WirelengthEval
{
 public:
  WirelengthEval();
  ~WirelengthEval();

  int32_t evalTotalHPWL(PointSets point_sets);
  int32_t evalTotalFLUTE(PointSets point_sets);
  int32_t evalTotalHTree(PointSets point_sets);
  int32_t evalTotalVTree(PointSets point_sets);

  int32_t evalNetHPWL(PointSet point_set);
  int32_t evalNetFLUTE(PointSet point_set);
  int32_t evalNetHTree(PointSet point_set);
  int32_t evalNetVTree(PointSet point_set);

  int32_t evalPathHPWL(PointSet point_set, PointPair point_pair);
  int32_t evalPathFLUTE(PointSet point_set, PointPair point_pair);
  int32_t evalPathHTree(PointSet point_set, PointPair point_pair);
  int32_t evalPathVTree(PointSet point_set, PointPair point_pair);

  int32_t evalTotalEGRWL();
  int32_t evalNetEGRWL(std::string net_name);
  int32_t evalPathEGRWL(std::string net_name, std::string point_name1, std::string point_name2);
};

}  // namespace ieval