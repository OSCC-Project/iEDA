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

void RuleValidator::verifyOutOfDie(RVBox& rv_box)
{
    std::vector<Violation>& violation_list = rv_box.get_violation_list();

    auto& die=DRCDM.getDatabase().get_die();
    GTLPolySetInt die_poly_set;
    die_poly_set+=GTLRectInt(die.get_ll_x(),die.get_ll_y(),die.get_ur_x(),die.get_ur_y());

    int outofdie_count=0;
    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_poly_set;
    for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
        if (rect->get_net_idx() == -1) {  //net_idx为-1 代表环境的shape
            continue; 
        }
        int32_t net_idx = rect->get_net_idx();
        int32_t layer_idx = rect->get_layer_idx();
        //FIXME：when the vio at the corner, does't have any design case to confirm the correctness of getting rect_vio.
        if(rect->get_ll_x()<die.get_ll_x()||rect->get_ll_y()<die.get_ll_y()||rect->get_ur_x()>die.get_ur_x()||rect->get_ur_y()>die.get_ur_y()){
            outofdie_count++;

            std::set<int32_t> net_set;
            net_set.insert(net_idx);

            GTLPolySetInt rect_poly_set;
            rect_poly_set+=GTLRectInt(rect->get_ll_x(),rect->get_ll_y(),rect->get_ur_x(),rect->get_ur_y());

            rect_poly_set-=die_poly_set;

            GTLRectInt rect_vio;
            gtl::extents(rect_vio,rect_poly_set);

            Violation violation;
            violation.set_violation_type(ViolationType::kOutOfDie);
            violation.set_is_routing(true);
            violation.set_violation_net_set(net_set);
            // violation.set_required_size();
            violation.set_layer_idx(layer_idx);
            violation.set_rect(PlanarRect(gtl::xl(rect_vio),gtl::yl(rect_vio),gtl::xh(rect_vio),gtl::yh(rect_vio)));
            violation_list.push_back(violation);

            // DRCLOG.info(Loc::current(),rect->get_ll_x()," ",rect->get_ll_y()," ",rect->get_ur_x()," ",rect->get_ur_y());
            // DRCLOG.info(Loc::current(),die.get_ll_x()," ",die.get_ll_y()," ",die.get_ur_x()," ",die.get_ur_y());
            // DRCLOG.info(Loc::current(),"vio position: ",gtl::xl(rect_vio)," ",gtl::yl(rect_vio)," ",gtl::xh(rect_vio)," ",gtl::yh(rect_vio));
        }
    }
    // DRCLOG.info(Loc::current(),"OutOfDie number:= ",outofdie_count);
}

}  // namespace idrc
