/**
 * @file SdcCollection.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The collection of sdc obj list for tcl.
 * @version 0.1
 * @date 2021-10-14
 */
#pragma once

#include <variant>
#include <vector>

#include "SdcCommand.hh"
#include "netlist/DesignObject.hh"

namespace ista {

using SdcCollectionObj = std::variant<SdcCommandObj*, DesignObject*>;

/**
 * @brief The sdc command obj list for wild match.
 *
 */
class SdcCollection : public SdcCommandObj {
 public:
  enum class CollectionType {
    kClock,
    kPin,
    kPort,
    kInst,
    kNet,
    kNetlist,
    kAllClocks
  };

  SdcCollection(CollectionType collection_type,
                std::vector<SdcCollectionObj>&& collection_objs);
  ~SdcCollection() override = default;

  unsigned isSdcCollection() override { return 1; }
  unsigned isClockCollection() {
    return _collection_type == CollectionType::kClock;
  }
  unsigned isNetlistCollection() {
    return _collection_type == CollectionType::kNetlist;
  }
  auto get_collection_type() { return _collection_type; }
  auto& get_collection_objs() { return _collection_objs; }

 private:
  CollectionType _collection_type;                 //!< The obj list type.
  std::vector<SdcCollectionObj> _collection_objs;  // The obj list.
};

}  // namespace ista
