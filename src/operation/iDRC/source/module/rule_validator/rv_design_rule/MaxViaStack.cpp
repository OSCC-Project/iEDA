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
#include "RuleValidator.hpp"

namespace idrc {

void RuleValidator::verifyMaxViaStack(RVBox& rv_box)
{
#if 1
struct CmpGTLRectInt
{
  bool operator()(const GTLRectInt& a, const GTLRectInt& b) const
  {
    GTLRectInt aa=a;
    GTLRectInt bb=b;
    PlanarRect a_rect = DRCUTIL.convertToPlanarRect(aa);
    PlanarRect b_rect = DRCUTIL.convertToPlanarRect(bb);
      if (a_rect != b_rect) {
        if (CmpPlanarRectByXASC()(a_rect, b_rect)) {
          return true;
        } else {
          return false; 
        }
      } else {
        return false;
    }
  }
};
#endif

  // rule get
  // only suit for rule of :LEF58_MAXVIASTACK STRING "MAXVIASTACK 4 NOSINGLE RANGE M1 M7
  // it need modify
  int32_t max_via_stack = 4;
  int32_t metal_idx_start = 0;
  int32_t metal_idx_end=6;
  //上面三个应该是需要获取的。下面三个是需要推导的(也是下面代码真正用的数据)。 暂定写死，需要修改

  int32_t cut_idx_start=1;
  int32_t cut_idx_end=6;
  int32_t required_size = 8000;//why 8000? 4*2000?

  std::vector<Violation>& violation_list = rv_box.get_violation_list();

  // 因为GTLRectInt 没有处理hash的功能，所以需要自己处理
  // 两个策略，一个是传入GTLRectInt比较函数，一个是用rtree来处理。
  
  // 第二个维度是存<layeridx,netidx>
  std::map<GTLRectInt, std::vector<std::pair<int32_t,int32_t>>, CmpGTLRectInt>rect_layeridx_map;

  bgi::rtree<std::pair<BGRectInt, std::pair<int32_t,int32_t>>, bgi::quadratic<16>>rtree;

  // auto& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  // auto& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();

  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (rect->get_is_routing()) {
      continue;
    }
    if(rect->get_layer_idx() < cut_idx_start || rect->get_layer_idx() > cut_idx_end) {
      continue;
    }
    auto gtlrect=GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
    rect_layeridx_map[gtlrect].push_back({rect->get_layer_idx(),rect->get_net_idx()});
    BGRectInt rtree_rect(BGPointInt(rect->get_ll_x(), rect->get_ll_y()), BGPointInt(rect->get_ur_x(), rect->get_ur_y()));
    rtree.insert(std::pair<BGRectInt, std::pair<int32_t,int32_t>>{rtree_rect, {rect->get_layer_idx(),rect->get_net_idx()}});
  }
  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (rect->get_is_routing()) {
      continue;
    }
    if(rect->get_layer_idx() < cut_idx_start || rect->get_layer_idx() > cut_idx_end) {
      continue;
    }
    auto gtlrect=GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
    rect_layeridx_map[gtlrect].push_back({rect->get_layer_idx(),rect->get_net_idx()});
    BGRectInt rtree_rect(BGPointInt(rect->get_ll_x(), rect->get_ll_y()), BGPointInt(rect->get_ur_x(), rect->get_ur_y()));
    rtree.insert(std::pair<BGRectInt, std::pair<int32_t,int32_t>>{rtree_rect, {rect->get_layer_idx(),rect->get_net_idx()}});
  }
  
  for(auto& [rect, layer_net_idx_list] : rect_layeridx_map) {
    std::map<int32_t,std::vector<int32_t>>layer_net_idx_map;
    // FIX:如果一个layer 上一个via有多个net产生的cutshort ,代码无法处理
    std::sort(layer_net_idx_list.begin(), layer_net_idx_list.end());
    //去重
    layer_net_idx_list.erase(std::unique(layer_net_idx_list.begin(), layer_net_idx_list.end()), layer_net_idx_list.end());
    // 旧的先加进map
    for(auto&[layer_idx, net_idx]:layer_net_idx_list) {
      layer_net_idx_map[layer_idx].push_back(net_idx);
    }

    //查询歪的rect，加入其中，贴边的不要
    std::vector<std::pair<BGRectInt, std::pair<int32_t,int32_t>>> result;
    BGRectInt rtree_query_rect(BGPointInt(gtl::xl(rect)+1, gtl::yl(rect)+1), BGPointInt(gtl::xh(rect)-1, gtl::yh(rect)-1));
    rtree.query(bgi::intersects(rtree_query_rect), std::back_inserter(result));
    
    //map处理主要逻辑
    int32_t count_add_before=0;
    int32_t count_add=0;
    for(auto& [rect_wai, layer_net_idx] : result) {
      if(layer_net_idx_map.find(layer_net_idx.first) == layer_net_idx_map.end()) {
        layer_net_idx_map[layer_net_idx.first].push_back(layer_net_idx.second);
        count_add++;
      }
      else {
        count_add_before++;
      }
    }
    // 处理歪的rect 不记录成 vio的rect的方式是：判断歪的个数和原来的个数，如果歪的个数大于原来的个数，则不记录成vio的rect

    if(count_add_before>=count_add){
      int32_t count = 0;
      int32_t start_idx = -1;
      int32_t end_idx=-1;
      for (int32_t i = cut_idx_start; i <= cut_idx_end; i++) {
        if (layer_net_idx_map[i].size()>0) {
          count++;
          end_idx=i;
          if(count==1){
            start_idx = i;
          }
        } else {
          if (count > max_via_stack) {
            Violation violation;
            violation.set_violation_type(ViolationType::kMaxViaStack);
            violation.set_is_routing(true);
            violation.set_layer_idx(i-1-1);//i-1是找上一个layer,再-1是记录metal的idx
            auto vio_rect=GTLRectInt(gtl::xl(rect), gtl::yl(rect), gtl::xh(rect), gtl::yh(rect));
            violation.set_rect(DRCUTIL.convertToPlanarRect(vio_rect));
            std::set<int32_t> violation_net_set;
            violation_net_set.insert(layer_net_idx_map[start_idx].front());
            violation.set_violation_net_set(violation_net_set);
            violation.set_required_size(required_size);
            violation_list.push_back(violation);
          }
          count = 0;
        }
      }
      if (count > max_via_stack) {
        Violation violation;
        violation.set_violation_type(ViolationType::kMaxViaStack);
        violation.set_is_routing(true);
        violation.set_layer_idx(end_idx-1);
        auto vio_rect=GTLRectInt(gtl::xl(rect), gtl::yl(rect), gtl::xh(rect), gtl::yh(rect));
        violation.set_rect(DRCUTIL.convertToPlanarRect(vio_rect));
        std::set<int32_t> violation_net_set;
        violation_net_set.insert(layer_net_idx_map[start_idx].front());
        violation.set_violation_net_set(violation_net_set);
        violation.set_required_size(required_size);
        violation_list.push_back(violation);
      }
    }
  }
}

}  // namespace idrc
