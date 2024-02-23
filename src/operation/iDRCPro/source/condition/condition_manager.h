// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include <list>
#include <map>
#include <string>
#include <vector>

#include "condition.h"
#include "condition_record.h"
#include "drc_basic_segment.h"
#include "idrc_util.h"
#include "idrc_violation_manager.h"
#include "tech_rules.h"

namespace idb {
class IdbLayer;
}

namespace idrc {

/**
 * rule conditions are concepts built from tech lef drc rules, it contains a condition matrix to guide condition check orders, the rule
 * matrix index indicates the checking order,
 *
 */
class DrcConditionManager
{
  using ConditionRecordPtr = std::shared_ptr<ConditionRecord>;

 public:
  DrcConditionManager(DrcViolationManager* violation_manager) : _violation_manager(violation_manager) {}
  ~DrcConditionManager() {}

  void recordSpacingForward(idb::IdbLayer* layer, int id, DrcCoordinate* spacing_forward, DrcCoordinate* spacing_backward)
  {
    auto& records = _condition_recording_map[layer][id];
    for (auto& record : records) {
      record->recordSpacingForward(DrcBasicSegment(*spacing_forward, *spacing_backward));
    }
  }

  void recordSpacingBackward(idb::IdbLayer* layer, int id, DrcCoordinate* spacing_forward, DrcCoordinate* spacing_backward)
  {
    auto& records = _condition_recording_map[layer][id];
    for (auto& record : records) {
      record->recordSpacingBackward(DrcBasicSegment(*spacing_forward, *spacing_backward));
    }
  }

  ConditionRecordPtr getEmptyRecord()
  {
    ConditionRecordPtr record = nullptr;
    if (!_record_pool.empty()) {
      record = _record_pool.front();
      _record_pool.pop_front();
    } else {
      record = std::make_shared<ConditionRecord>();
    }
    return record;
  }

#define DEBUG

  void recordEdge(idb::IdbLayer* layer, int id, DrcSegmentType segment_type, DrcBasicPoint* segment_forward,
                  DrcBasicPoint* segment_backward)
  {
    bool is_vertical = segment_forward->get_x() == segment_backward->get_x();
    auto& records = _condition_recording_map[layer][id];

#ifdef DEBUG
    auto gtl_polygon = DrcUtil::getPolygon(segment_forward);
    std::vector<ieda_solver::GtlRect> gtl_rects;
    DrcBasicSegment segment(*segment_forward, *segment_backward);
    auto gtl_points = DrcUtil::getPolygonPoints(segment_forward);
    if (gtl_points.size() > 4 && segment_type == DrcSegmentType::kConvexEndEdge) {
      int a = 0;
    }
#endif

    switch (segment_type.get_type()) {
      case DrcSegmentType::kConvexStartEdge: {
        auto conditions = DrcTechRuleInst->get_condition_with_width(layer, segment_forward - segment_backward);
        if (conditions.has_value()) {
          for (auto* condition : *conditions.value()) {
            auto record = getEmptyRecord();
            record->set_condition(condition);
            record->set_polygon(*segment_forward, *segment_backward);
            record->set_is_wire_vertical(!is_vertical);
            records.emplace_back(record);
          }
        } else {
          auto record = getEmptyRecord();
          record->set_polygon(*segment_forward, *segment_backward);
          record->set_is_wire_vertical(!is_vertical);
          records.emplace_back(record);
        }
        int a = 0;
        break;
      }
      case DrcSegmentType::kConvexEndEdge: {
        for (auto record_it = records.begin(); record_it != records.end();) {
          if ((*record_it)->in(segment_forward, segment_backward)) {
            (*record_it)->endRecording(is_vertical ? segment_forward->get_x() : segment_forward->get_y());
            _record_pool.push_back(*record_it);
            record_it = records.erase(record_it);
          } else {
            ++record_it;
          }
        }
        int a = 0;
        break;
      }
      case DrcSegmentType::kTurnInEdge: {
        for (auto record_it = records.begin(); record_it != records.end();) {
          if ((*record_it)->contains(segment_forward, segment_backward)) {
            auto new_record = getEmptyRecord();
            new_record->copySpacing((*record_it).get());
            new_record->set_condition((*record_it)->get_condition());
            int new_lb_x = (*record_it)->get_wire_region().get_lb().get_x();
            int new_lb_y = (*record_it)->get_wire_region().get_lb().get_y();
            int new_ur_x = (*record_it)->get_wire_region().get_ur().get_x();
            int new_ur_y = (*record_it)->get_wire_region().get_ur().get_y();
            if (is_vertical) {
              if (segment_forward->get_y() == (*record_it)->get_polygon_forward()) {
                new_ur_y = segment_backward->get_y();
              } else {
                new_lb_y = segment_forward->get_y();
              }
            } else {
              if (segment_forward->get_x() == (*record_it)->get_polygon_forward()) {
                new_ur_x = segment_backward->get_x();
              } else {
                new_lb_x = segment_forward->get_x();
              }
            }
            new_record->set_wire_region({new_lb_x, new_lb_y, new_ur_x, new_ur_y});
            records.emplace_back(new_record);
            (*record_it)->endRecording(is_vertical ? segment_forward->get_x() : segment_forward->get_y());
            _record_pool.push_back(*record_it);
            record_it = records.erase(record_it);
          } else {
            ++record_it;
          }
        }
        break;
      }
      case DrcSegmentType::kTurnOutEdge: {
        if (!records.empty()) {
          auto& last_record = records.back();
          auto new_record = getEmptyRecord();
          new_record->set_condition(last_record->get_condition());
          int new_lb_x = is_vertical ? segment_forward->get_x() : last_record->get_polygon_backward();
          int new_lb_y = is_vertical ? last_record->get_polygon_backward() : segment_forward->get_y();
          int new_ur_x = is_vertical ? segment_forward->get_x() : last_record->get_polygon_forward();
          int new_ur_y = is_vertical ? last_record->get_polygon_forward() : segment_forward->get_y();
          if (is_vertical) {
            if (segment_backward->get_y() == last_record->get_polygon_forward()) {
              new_ur_y = segment_forward->get_y();
            } else {
              new_lb_y = segment_backward->get_y();
            }
          } else {
            if (segment_backward->get_x() == last_record->get_polygon_forward()) {
              new_ur_x = segment_forward->get_x();
            } else {
              new_lb_x = segment_backward->get_x();
            }
          }
          records.emplace_back(new_record);
          new_record->set_wire_region({new_lb_x, new_lb_y, new_ur_x, new_ur_y});
          records.emplace_back(new_record);
        }
        break;
      }
      case DrcSegmentType::kConcaveStartEdge: {
        // TODO: 汇聚
        int a = 0;
        break;
      }
      case DrcSegmentType::kConcaveEndEdge: {
        // TODO: 发散
        int a = 0;
        break;
      }
      default:
        break;
    }

    if (records.empty()) {
      _condition_recording_map[layer].erase(id);
    }
  }

  // void recordEdge(idb::IdbLayer* layer, int id, DrcSegmentType segment_type, int segment_forward, int segment_backward, int
  // bucket_coord,
  //                 bool is_vertical)
  // {
  //   auto& records = _condition_recording_map[layer][id];
  //   switch (segment_type.get_type()) {
  //     case DrcSegmentType::kConvexStartEdge: {
  //       auto conditions = DrcTechRuleInst->get_condition_with_width(layer, segment_forward - segment_backward);
  //       if (conditions.has_value()) {
  //         for (auto* condition : *conditions.value()) {
  //           auto record = getEmptyRecord();
  //           record->set_condition(condition);
  //           record->set_polygon(segment_forward, segment_backward, bucket_coord, is_vertical);
  //           records.emplace_back(record);
  //         }
  //       }
  //       break;
  //     }
  //     case DrcSegmentType::kConvexEndEdge: {
  //       for (auto record_it = records.begin(); record_it != records.end();) {
  //         if ((*record_it)->contains(segment_forward, segment_backward)) {
  //           (*record_it)->endRecording(bucket_coord, is_vertical);
  //           _record_pool.push_back(*record_it);
  //           record_it = records.erase(record_it);
  //         } else {
  //           ++record_it;
  //         }
  //       }
  //       break;
  //     }
  //     case DrcSegmentType::kTurnInEdge: {
  //       // TODO: close those contains the segment
  //       // TODO: create smaller wire records and copy record data
  //       for (auto record_it = records.begin(); record_it != records.end();) {
  //         if ((*record_it)->contains(segment_forward, segment_backward)) {
  //           auto new_record = getEmptyRecord();
  //           new_record->copySpacing((*record_it).get());
  //           new_record->set_condition((*record_it)->get_condition());
  //           int new_forward
  //               = segment_forward == (*record_it)->get_polygon_forward() ? segment_backward : (*record_it)->get_polygon_forward();
  //           int new_backward
  //               = segment_backward == (*record_it)->get_polygon_backward() ? segment_forward : (*record_it)->get_polygon_backward();
  //           new_record->set_polygon(new_forward, new_backward, (*record_it)->get_polygon_start(), is_vertical);
  //           records.emplace_back(new_record);
  //           (*record_it)->endRecording(bucket_coord, is_vertical);
  //           _record_pool.push_back(*record_it);
  //           record_it = records.erase(record_it);
  //         } else {
  //           ++record_it;
  //         }
  //       }
  //       break;
  //     }
  //     case DrcSegmentType::kTurnOutEdge: {
  //       // TODO: create bigger wire records based on last record (?)
  //       if (!records.empty()) {
  //         auto& last_record = records.back();
  //         auto new_record = getEmptyRecord();
  //         new_record->set_condition(last_record->get_condition());
  //         new_record->set_polygon(std::max(segment_forward, last_record->get_polygon_forward()),
  //                                 std::min(segment_backward, last_record->get_polygon_backward()), last_record->get_polygon_start(),
  //                                 is_vertical);
  //         // new_record->recordSpacing(last_record->getCurrentSpacingForward(bucket_coord, is_vertical),
  //         //                           last_record->getCurrentSpacingBackward(bucket_coord, is_vertical));
  //         records.emplace_back(new_record);
  //       }
  //       break;
  //     }
  //     case DrcSegmentType::kConcaveStartEdge: {
  //       // TODO: 汇聚
  //       break;
  //     }
  //     case DrcSegmentType::kConcaveEndEdge: {
  //       // TODO: 发散
  //       break;
  //     }
  //     default:
  //       break;
  //   }

  //   if (records.empty()) {
  //     _condition_recording_map[layer].erase(id);
  //   }
  // }

 private:
  DrcViolationManager* _violation_manager;

  std::map<idb::IdbLayer*, std::map<int, std::list<ConditionRecordPtr>>> _condition_recording_map;  // layer -> polygon id -> records
  std::deque<ConditionRecordPtr> _record_pool;
};

}  // namespace idrc