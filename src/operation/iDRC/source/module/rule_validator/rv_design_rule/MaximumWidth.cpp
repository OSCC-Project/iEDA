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
#include <cstdint>
#include "RuleValidator.hpp"

namespace idrc {

void RuleValidator::verifyMaximumWidth(RVBox& rv_box)
{
    std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();

    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_result_poly_set_map;

    for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
        if (!drc_shape->get_is_routing()) {
          continue;
        }
        // 将形状的矩形转换为 GTLRectInt 并加入对应的层和网络的多边形集中
        routing_net_result_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    }

    for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_result_poly_set_map){

        int32_t max_width = routing_layer_list[routing_layer_idx].get_max_width();

        for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map){
            std::vector<GTLPolyInt> gtl_poly_list;// 存储多边形列表
            gtl_poly_set.get_polygons(gtl_poly_list);// 从多边形集中提取所有多边形

            for (GTLPolyInt& gtl_poly : gtl_poly_list){
                std::vector<GTLRectInt> gtl_max_rects;
                gtl::get_max_rectangles(gtl_max_rects, gtl_poly);
                
                GTLRectInt violation_rect;// 存储违反最小宽度的矩形
                int32_t rect_width = 0;// 初始化矩形宽度为一个较大值

                for (const GTLRectInt& max_rect : gtl_max_rects) {
                    //取窄边做宽
                    rect_width = gtl::delta(max_rect, gtl::HORIZONTAL) < gtl::delta(max_rect, gtl::VERTICAL) ? gtl::delta(max_rect, gtl::HORIZONTAL) : gtl::delta(max_rect, gtl::VERTICAL);
                    if(rect_width > max_width){
                        violation_rect = max_rect;
                        Violation violation;
                        violation.set_violation_type(ViolationType::kMaximumWidth);
                        violation.set_is_routing(true);
                        violation.set_violation_net_set({net_idx});
                        violation.set_required_size(max_width);
                        violation.set_layer_idx(routing_layer_idx);
                        violation.set_rect(DRCUTIL.convertToPlanarRect(violation_rect));
                        rv_box.get_violation_list().push_back(violation);
                    }
                }

            }
        }
    }

}

}  // namespace idrc
