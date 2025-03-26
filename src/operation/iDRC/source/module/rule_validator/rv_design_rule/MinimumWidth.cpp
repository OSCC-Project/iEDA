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

#include <boost/polygon/isotropy.hpp>
#include <boost/polygon/rectangle_concept.hpp>
#include <cstdint>
#include "Direction.hpp"
#include "RuleValidator.hpp"
#include <boost/geometry/index/rtree.hpp>
using namespace boost::polygon::operators;

namespace idrc { 

void RuleValidator::verifyMinimumWidth(RVBox& rv_box)
{
    // 获取布线层列表的引用
    std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();

    // 定义两个映射：按层索引和网络索引存储结果多边形集和环境多边形集
    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_result_poly_set_map;
    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_env_poly_set_map;

    // 遍历结果形状列表，填充 routing_net_result_poly_set_map
    for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
        if (!drc_shape->get_is_routing()) {
          continue;
        }
        // 将形状的矩形转换为 GTLRectInt 并加入对应的层和网络的多边形集中
        routing_net_result_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    }

    // 遍历环境形状列表，填充 routing_net_env_poly_set_map
    for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
        if (!drc_shape->get_is_routing()) {
          continue;
        }
        // 将环境形状的矩形转换为 GTLRectInt 并加入对应的层和网络的多边形集中
        routing_net_env_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    }

    // 遍历每层的结果多边形集
    for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_result_poly_set_map){
        // 获取当前层的最小宽度要求
        int32_t min_width = routing_layer_list[routing_layer_idx].get_min_width();
        
        // 遍历每个网络的多边形集
        for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map){
            std::vector<GTLPolyInt> gtl_poly_list;// 存储多边形列表
            gtl_poly_set.get_polygons(gtl_poly_list);// 从多边形集中提取所有多边形

            //包含在env中的不检测
            //与env相交+？？条件不检测；？？还不清楚
            //??:面积大于一半(猜测)

            bool test;// 标记当前多边形是否需要检测
            GTLPolySetInt tmp_set;// 临时存储当前多边形的多边形集
            GTLPolySetInt inter_test_set;// 存储与环境多边形集的交集

            // 遍历每个多边形
            for (GTLPolyInt& gtl_poly : gtl_poly_list){

                test = true;
                tmp_set.insert(gtl_poly);

                // 获取当前层的所有环境多边形集
                std::map<int32_t, GTLPolySetInt> net_env_poly_set_map = routing_net_env_poly_set_map[routing_layer_idx];

                // 检查与环境多边形集的关系
                for (auto& [net_env_idx, env_poly_set] : net_env_poly_set_map){
                    inter_test_set.clear();
                    inter_test_set = tmp_set & env_poly_set;
                    // 如果完全包含在环境多边形集中或交集面积超过一半，则跳过检测
                    if(tmp_set == inter_test_set || gtl::area(inter_test_set) >= gtl::area(tmp_set)/2){
                        test = false;
                        break;
                    }
                }
                tmp_set.clear();

                if(!test){// 如果不需要检测，则跳过此多边形
                    continue;
                }

                //获取最大矩形
                std::vector<GTLRectInt> gtl_max_rects;
                gtl::get_max_rectangles(gtl_max_rects, gtl_poly);
                GTLRectInt exclude_rect;


                //过滤掉get_max_rectangles中的一些特殊冗余矩形，基于与第一个矩形的关系
                for (auto it = gtl_max_rects.begin()+1; it != gtl_max_rects.end(); ) {
                    const auto& exclude_rect = *it;
                    if (gtl::yl(exclude_rect) == gtl::yl(gtl_max_rects[0]) &&
                        gtl::xl(exclude_rect) >  gtl::xl(gtl_max_rects[0]) &&
                        gtl::xh(exclude_rect) == gtl::xh(gtl_max_rects[0]) &&
                        gtl::yh(exclude_rect) >  gtl::yh(gtl_max_rects[0])) {
                        it = gtl_max_rects.erase(it); // 删除并更新迭代器
                    } else if (gtl::xl(exclude_rect) == gtl::xl(gtl_max_rects[0]) &&
                                gtl::yl(exclude_rect) <  gtl::yl(gtl_max_rects[0]) &&
                                gtl::yh(exclude_rect) == gtl::yh(gtl_max_rects[0]) &&
                                gtl::xh(exclude_rect) <  gtl::xh(gtl_max_rects[0])) {
                        it = gtl_max_rects.erase(it);
                    }else if (gtl::xh(exclude_rect) == gtl::xh(gtl_max_rects[0]) &&
                                gtl::yh(exclude_rect) == gtl::yh(gtl_max_rects[0]) &&
                                gtl::xl(exclude_rect) >  gtl::xl(gtl_max_rects[0]) &&
                                gtl::yl(exclude_rect) <  gtl::yl(gtl_max_rects[0])) {
                        it = gtl_max_rects.erase(it);
                    }else if(gtl::xl(exclude_rect) == gtl::xl(gtl_max_rects[0]) &&
                                gtl::yl(exclude_rect) == gtl::yl(gtl_max_rects[0]) &&
                                gtl::xh(exclude_rect) <  gtl::xh(gtl_max_rects[0]) &&
                                gtl::yh(exclude_rect) >  gtl::yh(gtl_max_rects[0])){
                        it = gtl_max_rects.erase(it);
                    }else{
                        ++it; // 检查下一个元素
                    }
                }

                GTLRectInt violation_rect;// 存储违反最小宽度的矩形
                int32_t rect_width = min_width*2;// 初始化矩形宽度为一个较大值
                for (const GTLRectInt& max_rect : gtl_max_rects) {
                    //取窄边做宽
                    rect_width = gtl::delta(max_rect, gtl::HORIZONTAL) < gtl::delta(max_rect, gtl::VERTICAL) ? gtl::delta(max_rect, gtl::HORIZONTAL) : gtl::delta(max_rect, gtl::VERTICAL);
                    if(rect_width < min_width){
                        violation_rect = max_rect;
                        Violation violation;
                        violation.set_violation_type(ViolationType::kMinimumWidth);
                        violation.set_is_routing(true);
                        violation.set_violation_net_set({net_idx});
                        violation.set_required_size(min_width);
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
