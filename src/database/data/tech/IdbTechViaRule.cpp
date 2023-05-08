#include "IdbTechViaRule.h"

namespace idb {
  void IdbTechViaRuleList::printViaRule() {
    for (auto &viaRule : _tech_via_rules) {
      std::cout << "viaRuleGenerateName ::" << viaRule->get_name() << std::endl;
    }
  }

  IdbTechViaRule *IdbTechViaRuleList::getTechViaRule(int cutLayerId) {
    for (auto &viaRule : _tech_via_rules) {
      if (viaRule->get_cut_layer_id() == cutLayerId) {
        return viaRule.get();
      }
    }
    return nullptr;
  }
}  // namespace idb
