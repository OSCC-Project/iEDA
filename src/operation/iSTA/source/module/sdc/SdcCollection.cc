/**
 * @file SdcCollection.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The collection of sdc obj list for tcl.
 * @version 0.1
 * @date 2021-10-14
 */
#include "SdcCollection.hh"

namespace ista {
SdcCollection::SdcCollection(CollectionType collection_type,
                             std::vector<SdcCollectionObj>&& collection_objs)
    : _collection_type(collection_type),
      _collection_objs(std::move(collection_objs)) {}
}  // namespace ista
