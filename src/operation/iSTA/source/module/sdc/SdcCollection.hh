// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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
    kAllClocks,
    kAllInputPorts,
    kAllOutputPorts
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
