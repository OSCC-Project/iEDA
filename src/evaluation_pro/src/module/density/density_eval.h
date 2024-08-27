#include "density_db.h"

namespace ieval {
class DensityEval
{
 public:
  DensityEval();
  ~DensityEval();

  std::string evalCellDensity();
  std::string evalPinDensity();
  std::string evalNetDensity();
  std::string evalChannelDensity();
  std::string evalWhitespaceDensity();
};
}  // namespace ieval