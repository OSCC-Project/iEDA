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




//目标：代码优化，相同部分提炼成函数调用

//目标：解决length相关的问题，有两个方案：计算polygon的各边长度，然后把数据带下去；反查相交矩形，算对应的边长。
//目标：需要排除凹角的边的检测区域（有待商榷）

namespace idrc {
/*
PROPERTY LEF58_SPACING
"SPACING eolSpace
ENDOFLINE eolWidth
[[OPPOSITEWIDTH oppositeWidth]
WITHIN eolWithin [wrongDirWithin]
[SAMEMASK]
[WITHCUT [CUTCLASS cutClass] [ABOVE] withCutSpace
[ENCLOSUREEND enclosureEndWidth
[WITHIN enclosureEndWithin]]]
[ENDTOEND endToEndSpace [oneCutSpace twoCutSpace]
[EXTENSION extension [wrongDirExtension]]
[OTHERENDWIDTH otherEndWidth]]
[MAXLENGTH maxLength | MINLENGTH minLength [TWOSIDES]]
[EQUALRECTWIDTH]
[PARALLELEDGE [SUBTRACTEOLWIDTH] parSpace
WITHIN parWithin [MINLENGTH minLength] [TWOEDGES]
[SAMEMETAL][NONEOLCORNERONLY]]
[ENCLOSECUT [BELOW | ABOVE] encloseDist
CUTSPACING cutToMetalSpace [ALLCUTS]]
| TOCONCAVECORNER [MINLENGTH minLength]
*/
//只写了目前要用到的数据
//SPACING 0.100 ENDOFLINE 0.070 WITHIN 0.025 ENDTOEND 0.080 PARALLELEDGE SUBTRACTEOLWIDTH 0.115 WITHIN 0.070 MINLENGTH 0.050 ENCLOSECUT BELOW 0.050 CUTSPACING 0.145 ALLCUTS ;
//SPACING 0.115 ENDOFLINE 0.055 WITHIN 0.000 PARALLELEDGE 0.060 WITHIN 0.120 MINLENGTH 0.150 TWOEDGES SAMEMETAL ;
class Eol_Layer_Data{
    public:
    enum class CutDirection
    {
        kNone,
        kBelow,
        kAbove,
        kBoth,
    };

    int32_t get_eol_space() const { return _eol_space; }
    int32_t get_eol_width() const { return _eol_width; }
    int32_t get_eol_within() const { return _eol_within; }
    int32_t get_par_space() const { return _par_space; }
    int32_t get_par_within() const { return  _par_within; }
    bool is_ENDTOEND() const { return _is_ENDTOEND; }
    int32_t get_ete_space() const { return _ete_space; }
    bool is_PARALLELEDGE() const { return _is_PARALLELEDGE; }
    bool is_SUBTRACTEOLWIDTH() const { return _is_SUBTRACTEOLWIDTH; }
    int32_t get_min_length() const { return _min_length; }
    bool is_TWOEDGES() const { return _is_TWOEDGES; }
    bool is_SAMEMETAL() const { return _is_SAMEMETAL; }
    bool is_ENCLOSECUT() const { return _is_ENCLOSECUT; }
    CutDirection get_cut_dir() const { return _cut_dir; }
    int32_t get_enclose_dist() const { return _enclose_dist; }
    int32_t get_ctm_space() const { return _ctm_space; }
    bool is_ALLCUTS() const { return _is_ALLCUTS; }

    Eol_Layer_Data() = default;
    Eol_Layer_Data(int32_t eol_space = 0, int32_t eol_width = 0, int32_t eol_within = 0,
                bool ENDTOEND = false, int32_t ete_space = 0,
                bool PARALLELEDGE = false,bool SUBTRACTEOLWIDTH = false,  int32_t par_space = 0,
                int32_t par_within = 0, int32_t min_lenth = 0, bool TWOEDGES = false, bool SAMEMETAL = false,
                bool ENCLOSECUT = false, CutDirection cut_dir = CutDirection::kNone, int32_t enclose_dist = 0,
                int32_t ctm_space = 0, bool ALLCUTS = false
                ):
                _eol_space(eol_space), _eol_width(eol_width), _eol_within(eol_within),
                _is_ENDTOEND(ENDTOEND), _ete_space(ete_space),
                _is_PARALLELEDGE(PARALLELEDGE),_is_SUBTRACTEOLWIDTH(SUBTRACTEOLWIDTH), 
                _par_space(par_space), _par_within(par_within), _min_length(min_lenth),
                _is_TWOEDGES(TWOEDGES), _is_SAMEMETAL(SAMEMETAL),
                _is_ENCLOSECUT(ENCLOSECUT), _enclose_dist(enclose_dist), 
                _cut_dir(cut_dir), _ctm_space(ctm_space), _is_ALLCUTS(ALLCUTS)
                {};
    ~Eol_Layer_Data() = default;
    
    
    private:
    int32_t _eol_space;
    int32_t _eol_width;
    int32_t _eol_within;
    bool _is_ENDTOEND;//用于判断是否启用端到端检测逻辑
    int32_t _ete_space;
    bool _is_PARALLELEDGE;//用于判断是否启用平行边检测逻辑
    bool _is_SUBTRACTEOLWIDTH;
    int32_t _par_space;
    int32_t _par_within;
    int32_t _min_length;
    bool _is_TWOEDGES;
    bool _is_SAMEMETAL;
    bool _is_ENCLOSECUT;//用于判断是否启用cut相关检测逻辑
    CutDirection _cut_dir;
    int32_t _enclose_dist;
    int32_t _ctm_space;
    bool _is_ALLCUTS;

    
};

class Eol_Layer_Rules{
    public:

    //用于画出最大搜索框，找出所有潜在违例矩形。
    int32_t get_rule_list_size(int32_t layer_idx){
        return  layer_rules[layer_idx].size();
    }
    int32_t get_max_eol_space(int32_t layer_idx){
        int32_t max_eol_space = 0;
        for(auto& one_rule : layer_rules[layer_idx]){
            max_eol_space = std::max(max_eol_space, one_rule.get_eol_space());
        }
        return max_eol_space;
    }
    int32_t get_max_eol_width(int32_t layer_idx){
        int32_t max_eol_width = 0;
        for(auto& one_roule : layer_rules[layer_idx]){
            max_eol_width = std::max(max_eol_width, one_roule.get_eol_width());
        }
        return max_eol_width;
    }
    int32_t get_max_eol_within(int32_t layer_idx){
        int32_t max_eol_within = 0;
        for(auto& one_roule : layer_rules[layer_idx]){
            max_eol_within = std::max(max_eol_within, one_roule.get_eol_within());
        }
        return max_eol_within;
    }
    int32_t get_max_par_space(int32_t layer_idx){
        int32_t max_par_space = 0;
        for(auto& one_roule : layer_rules[layer_idx]){
            max_par_space = std::max(max_par_space, one_roule.get_par_space());
        }
        return max_par_space;
    }
    int32_t get_max_par_within(int32_t layer_idx){
        int32_t max_par_within = 0;
        for(auto& one_roule : layer_rules[layer_idx]){
            max_par_within = std::max(max_par_within, one_roule.get_par_within());
        }
        return max_par_within;
    }
    int32_t get_max_sub_par_space(int32_t layer_idx){
        if(layer_idx == 0){
            return 240;
        }else {
            return 230;
        }
    }
    int32_t get_max_nsub_par_space(int32_t layer_idx){
        return 120;
    }

    std::vector<Eol_Layer_Data> get_rules(int32_t layer_idx){
        return layer_rules[layer_idx];
    }

    int32_t get_max_ete_space(int32_t layer_idx){
        int32_t max_ete_space = 0;
        for(auto& one_roule : layer_rules[layer_idx]){
            max_ete_space = std::max(max_ete_space, one_roule.get_ete_space());
        }
        return max_ete_space;
    }

    private:
    //规则从复杂到简单，逆排是为了细化的时候，从复杂到简单进行判断
    std::map<int32_t, std::vector<Eol_Layer_Data>> layer_rules = {
        {0, {Eol_Layer_Data(230, 110, 0, false, 0, true, false, 120, 240, 300, true, true),
             Eol_Layer_Data(140, 140, 50, false, 0, true, true, 240, 140, 100),
             Eol_Layer_Data(120, 140, 50)}},
        {1, {Eol_Layer_Data(230, 110, 0, false, 0, true, false, 120, 240, 300, true, true),
             Eol_Layer_Data(200, 140, 50, true, 160, true, true, 230, 140, 100, false, false, true, Eol_Layer_Data::CutDirection::kBelow, 100, 290, true),
             Eol_Layer_Data(160, 140, 50, true, 160, true, true, 230, 140, 100),
             Eol_Layer_Data(140, 140, 50, true, 160)}},
        {2, {Eol_Layer_Data(230, 110, 0, false, 0, true, false, 120, 240, 300, true, true),
             Eol_Layer_Data(200, 140, 50, true, 160, true, true, 230, 140, 100, false, false, true, Eol_Layer_Data::CutDirection::kBelow, 100, 290, true),
             Eol_Layer_Data(160, 140, 50, true, 160, true, true, 230, 140, 100),
             Eol_Layer_Data(140, 140, 50, true, 160)}},
        {3, {Eol_Layer_Data(230, 110, 0, false, 0, true, false, 120, 240, 300, true, true),
             Eol_Layer_Data(200, 140, 50, true, 160, true, true, 230, 140, 100, false, false, true, Eol_Layer_Data::CutDirection::kBelow, 100, 290, true),
             Eol_Layer_Data(160, 140, 50, true, 160, true, true, 230, 140, 100),
             Eol_Layer_Data(140, 140, 50, true, 160)}},
        {4, {Eol_Layer_Data(230, 110, 0, false, 0, true, false, 120, 240, 300, true, true),
             Eol_Layer_Data(200, 140, 50, true, 160, true, true, 230, 140, 100, false, false, true, Eol_Layer_Data::CutDirection::kBelow, 100, 290, true),
             Eol_Layer_Data(160, 140, 50, true, 160, true, true, 230, 140, 100),
             Eol_Layer_Data(140, 140, 50, true, 160)}},
        {5, {Eol_Layer_Data(230, 110, 0, false, 0, true, false, 120, 240, 300, true, true),
             Eol_Layer_Data(200, 140, 50, true, 160, true, true, 230, 140, 100, false, false, true, Eol_Layer_Data::CutDirection::kBelow, 100, 290, true),
             Eol_Layer_Data(160, 140, 50, true, 160, true, true, 230, 140, 100),
             Eol_Layer_Data(140, 140, 50, true, 160)}},
        {6, {Eol_Layer_Data(230, 110, 0, false, 0, true, false, 120, 240, 300, true, true),
             Eol_Layer_Data(200, 140, 50, true, 160, true, true, 230, 140, 100, false, false, true, Eol_Layer_Data::CutDirection::kBelow, 100, 290, true),
             Eol_Layer_Data(160, 140, 50, true, 160, true, true, 230, 140, 100),
             Eol_Layer_Data(140, 140, 50, true, 160)}}
    };
};


/*
辅助函数：生成违例矩形区域
注意与违例区相交矩形和目标矩形的相对位置关系:
第一个参数是位于左边/上面的矩形，第二个参数是位于右边/下边的矩形
两个参数是相对位置关系，不是对应目标矩形和相交矩形
两个net的矩形相交产生的违例会在这里处理成无效矩形
*/
PlanarRect gen_vio_rect_left_right( PlanarRect& rect_left,  PlanarRect& rect_right) {
    int32_t rect_ll_x, rect_ll_y, rect_ur_x, rect_ur_y;

    // 左右关系
    if (rect_left.get_ur_x() <= rect_right.get_ll_x()) {
        rect_ll_x = rect_left.get_ur_x();
        rect_ur_x = rect_right.get_ll_x();
        if (rect_left.get_ur_y() < rect_right.get_ll_y()) { // rect_right 在上方
            rect_ll_y = rect_left.get_ur_y();
            rect_ur_y = rect_right.get_ll_y();
        } else if (rect_right.get_ur_y() < rect_left.get_ll_y()) { // rect_right 在下方
            rect_ll_y = rect_right.get_ur_y();
            rect_ur_y = rect_left.get_ll_y();
        } else { // y 方向重叠
            rect_ll_y = std::max(rect_left.get_ll_y(), rect_right.get_ll_y());
            rect_ur_y = std::min(rect_left.get_ur_y(), rect_right.get_ur_y());
        }
    }else {
        return PlanarRect(0, 0, 0, 0); // 重叠或无效情况
    }
    if (rect_ll_x > rect_ur_x || rect_ll_y > rect_ur_y) {
        return PlanarRect(0, 0, 0, 0); // 无效矩形
    }

    return PlanarRect(rect_ll_x, rect_ll_y, rect_ur_x, rect_ur_y);
}

PlanarRect gen_vio_rect_down_up( PlanarRect& rect_down,  PlanarRect& rect_up) {
    int32_t rect_ll_x, rect_ll_y, rect_ur_x, rect_ur_y;

    if (rect_down.get_ur_y() <= rect_up.get_ll_y()) {
        rect_ll_y = rect_down.get_ur_y();
        rect_ur_y = rect_up.get_ll_y();
        if (rect_down.get_ur_x() < rect_up.get_ll_x()) { // rect_up 在右侧
            rect_ll_x = rect_down.get_ur_x();
            rect_ur_x = rect_up.get_ll_x();
        } else if (rect_up.get_ur_x() < rect_down.get_ll_x()) { // rect_up 在左侧
            rect_ll_x = rect_up.get_ur_x();
            rect_ur_x = rect_down.get_ll_x();
        } else { // x 方向重叠
            rect_ll_x = std::max(rect_down.get_ll_x(), rect_up.get_ll_x());
            rect_ur_x = std::min(rect_down.get_ur_x(), rect_up.get_ur_x());
        }
    }else {
        return PlanarRect(0, 0, 0, 0); // 重叠或无效情况
    }
    if (rect_ll_x > rect_ur_x || rect_ll_y > rect_ur_y) {
        return PlanarRect(0, 0, 0, 0); // 无效矩形
    }

    return PlanarRect(rect_ll_x, rect_ll_y, rect_ur_x, rect_ur_y);
}

/*辅助函数：生成矩形四个方向两边的未覆盖边长，用于判断单边是否满足min_length*/
void get_rect_len(const GTLPolyInt& poly, const PlanarRect& rect, std::vector<int32_t>& len) {
    int32_t east_up = 0;
    int32_t east_down = 0;
    int32_t west_up = 0;
    int32_t west_down = 0;
    int32_t north_left = 0;
    int32_t north_right = 0;
    int32_t south_left = 0;
    int32_t south_right = 0;

    PlanarCoord ll = rect.get_ll();
    PlanarCoord ur = rect.get_ur();
    PlanarCoord ul = PlanarCoord(ll.get_x(), ur.get_y());
    PlanarCoord lr = PlanarCoord(ur.get_x(), ll.get_y());
    //遍历多边形获得边列表
    std::vector<std::pair<PlanarCoord, PlanarCoord>> edge;
    for(auto it = poly.begin(); it != poly.end(); it++){
        GTLPointInt current = *it;
        auto next_it = it;
        next_it++; // 移动到下一个迭代器
        if (next_it == poly.end()) {
            next_it = poly.begin(); // 如果到达末尾，则回到起点
        }
        GTLPointInt next = *next_it;
        edge.push_back(std::make_pair(PlanarCoord(current.x(), current.y()), PlanarCoord(next.x(), next.y())));
    }
    //根据方向和矩形获取边长

    for(auto& one_edge : edge){
        auto [p1, p2] = one_edge; 
        if(p1.get_y() == p2.get_y()){
            //水平边
            if(p1 == ur || p2 == ur){
                east_up = std::abs(p1.get_x() - p2.get_x());
            }
            if (p1 == lr || p2 == lr) {
                east_down = std::abs(p1.get_x() - p2.get_x());
            }
            if (p1 == ul || p2 == ul) {
                west_up = std::abs(p1.get_x() - p2.get_x());
            }
            if (p1 == ll || p2 == ll) {
                west_down = std::abs(p1.get_x() - p2.get_x());
            }
        }else if(p1.get_x() == p2.get_x()){
            if(p1 == ul || p2 == ul){
                north_left = std::abs(p1.get_y() - p2.get_y());
            }
            if (p1 == ur || p2 == ur) {
                north_right = std::abs(p1.get_y() - p2.get_y());
            }
            if (p1 == ll || p2 == ll) {
                south_left = std::abs(p1.get_y() - p2.get_y());
            }
            if (p1 == lr || p2 == lr) {
                south_right = std::abs(p1.get_y() - p2.get_y());
            }
        }
    }

       
    
    len.push_back(east_up);
    len.push_back(east_down);
    len.push_back(west_up);
    len.push_back(west_down);
    len.push_back(north_left);
    len.push_back(north_right);
    len.push_back(south_left);
    len.push_back(south_right);
}

/*辅助函数：计算向量叉积*/
int32_t cross_product(const GTLPointInt& p1, const GTLPointInt& p2, const GTLPointInt& p3) {
    int32_t x1 = p2.x() - p1.x();
    int32_t y1 = p2.y() - p1.y();
    int32_t x2 = p3.x() - p2.x();
    int32_t y2 = p3.y() - p2.y();
    return x1 * y2 - x2 * y1;
}



/*辅助函数：判断一边在多边形中是否是末端。一边两端都是凸角就是末端*/
/*多边形遍历顺序默认为顺时针：叉积为负是凸角；叉积为正是凹角*/
void get_end_edge(const GTLPolyInt& poly, const PlanarRect& rect, std::vector<bool>& is_end){
    bool up = false;
    bool down = false;
    bool left = false;
    bool right = false;
    PlanarCoord ll = rect.get_ll();
    PlanarCoord ur = rect.get_ur();
    PlanarCoord ul = PlanarCoord(ll.get_x(), ur.get_y());
    PlanarCoord lr = PlanarCoord(ur.get_x(), ll.get_y());
    std::vector<std::pair<PlanarCoord, bool>> coord;//真为凸角，假为凹角
    std::vector<std::tuple<PlanarCoord, PlanarCoord, bool>> edge;//真为末端，假为非末端
    std::vector<GTLPointInt> vertices;
    for (auto it = poly.begin(); it != poly.end(); ++it) {
        vertices.push_back(*it);
    }
    size_t n = vertices.size();
    for (size_t i = 0; i < n; ++i) {
        GTLPointInt prev = vertices[(i - 1 + n) % n];
        GTLPointInt current = vertices[i];
        GTLPointInt next = vertices[(i + 1) % n];
        int32_t cp = cross_product(prev, current, next);
        if (cp < 0) {
            coord.push_back({PlanarCoord(current.x(),current.y()),true});
        }else {
            coord.push_back({PlanarCoord(current.x(),current.y()),false});
        }
    }
    for (size_t i = 0; i < n; ++i) {
        GTLPointInt p1 = vertices[i];
        GTLPointInt p2 = vertices[(i + 1) % n];
        bool is_end_edge = coord[i].second && coord[(i + 1) % n].second;
        edge.push_back({PlanarCoord(p1.x(), p1.y()), PlanarCoord(p2.x(), p2.y()), is_end_edge});
    }
    for (auto& one_edge : edge) {
        auto[p1, p2, is_end_edge] = one_edge;
        if(p1.get_y() == p2.get_y()){
            if (p1.get_y() == ur.get_y() && ul.get_x() >= std::min(p1.get_x(), p2.get_x()) && ur.get_x() <= std::max(p1.get_x(), p2.get_x())) {
                up = is_end_edge;
            }
            if (p1.get_y() == ll.get_y() && ll.get_x() >= std::min(p1.get_x(), p2.get_x()) && lr.get_x() <= std::max(p1.get_x(), p2.get_x())) {
                down = is_end_edge;
            }
        }else if(p1.get_x() == p2.get_x()) {
            if (p1.get_x() == ll.get_x() && ll.get_y() >= std::min(p1.get_y(), p2.get_y()) && ul.get_y() <= std::max(p1.get_y(), p2.get_y())) {
                left = is_end_edge;
            }
            if (p1.get_x() == ur.get_x() && lr.get_y() >= std::min(p1.get_y(), p2.get_y()) && ur.get_y() <= std::max(p1.get_y(), p2.get_y())) {
                right = is_end_edge;
            }
        }
    }


    is_end.push_back(up);
    is_end.push_back(down);
    is_end.push_back(left);
    is_end.push_back(right);
}




void RuleValidator::verifyEndOfLineSpacing(RVBox& rv_box)
{
    std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_layer_net_all_gtl_poly_set_map;
    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_layer_net_result_gtl_poly_set_map;
    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> cut_layer_net_all_gtl_poly_set_map;

    for(DRCShape* drc_shape : rv_box.get_drc_env_shape_list()){
        if(!drc_shape->get_is_routing()){
            cut_layer_net_all_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
        }
        routing_layer_net_all_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    }
    for(DRCShape* drc_shape : rv_box.get_drc_result_shape_list()){
        if(!drc_shape->get_is_routing()){
            cut_layer_net_all_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
        }else {
            routing_layer_net_result_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
            routing_layer_net_all_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
        }
    }

    for(auto& [routing_layer_idx, net_all_gtl_poly_set_map] : routing_layer_net_all_gtl_poly_set_map){

        if(routing_layer_idx > 6){
            break;
        }
        if (routing_layer_idx == 0) {
            continue;
        }
        bgi::rtree<BGRectInt, bgi::quadratic<16>> bg_routing_rect_result_rtree;
        bgi::rtree<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>, bgi::quadratic<16>> bg_routing_rect_net_mental_len_all_rtree;
        bgi::rtree<BGRectInt, bgi::quadratic<16>> bg_cut_rect_result_rtree;

        /*len列表会包含四个值：东上/下，西上/下,北左/右，南左/右。分别对应0.1.2.3.4.5.6.7*/
        std::vector<std::tuple<PlanarRect, std::pair<int32_t, int32_t>, std::pair<std::vector<int32_t>, std::vector<bool>>>> check_p_rect_net_menta_len_list;
        int32_t min_width = routing_layer_list[routing_layer_idx].get_min_width();

        PlanarRect invalid_rect = PlanarRect(0, 0, 0, 0);
        // PlanarRect test_rect1 = PlanarRect(93690,65350,93910,65450); 
        // PlanarRect test_rect2 = PlanarRect(100830,81950,101050,82050);

        Eol_Layer_Rules eol_layer_rules;
        std::vector<Eol_Layer_Data> eol_rules = eol_layer_rules.get_rules(routing_layer_idx);
        int32_t max_eol_space = eol_layer_rules.get_max_eol_space(routing_layer_idx);
        int32_t max_eol_within = eol_layer_rules.get_max_eol_within(routing_layer_idx);
        int32_t max_sub_par_space = eol_layer_rules.get_max_sub_par_space(routing_layer_idx);
        int32_t max_nsub_par_space = eol_layer_rules.get_max_nsub_par_space(routing_layer_idx);
        int32_t max_par_within = eol_layer_rules.get_max_par_within(routing_layer_idx);
        int32_t max_eol_width = eol_layer_rules.get_max_eol_width(routing_layer_idx);
        int32_t max_ete_space = eol_layer_rules.get_max_ete_space(routing_layer_idx);

        int32_t mesewet;//max_eol_space,max_eol_within,max_ete_space中的最大值
        int32_t mpsew;//max_sub_par_space - width，max_nsub_par_space之中的最大值，然后和max_eol_within比较取最大值
        //max_sub_par_space涉及到线宽，mpsew只能拿到矩形后才能算出。

        mesewet = std::max(max_eol_space, max_eol_within);
        mesewet = std::max(mesewet, max_ete_space);
        
        // 获取该层方向
        Direction layer_dir = routing_layer_list[routing_layer_idx].get_prefer_direction();
        //构建绕线层result和all的r树。
        std::vector<std::tuple<BGRectInt,std::pair<int32_t, int32_t>, std::vector<int32_t>>> insert_all_rect;
        int32_t mental_idx = 0;
        for(auto& [net_idx, all_gtl_poly_set] : net_all_gtl_poly_set_map){
            std::vector<GTLPolyInt> all_poly;
            all_gtl_poly_set.get(all_poly);
            for(auto& one_poly : all_poly){
                std::vector<GTLRectInt> one_gtl_rect_list;
                gtl::get_max_rectangles(one_gtl_rect_list, one_poly);
                for(auto& gtl_rect : one_gtl_rect_list){
                    BGRectInt bg_rect = DRCUTIL.convertToBGRectInt(gtl_rect);
                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
                    std::vector<int32_t> len;
                    std::vector<bool> is_end;
                    get_rect_len(one_poly, p_rect, len);
                    get_end_edge(one_poly, p_rect, is_end);
                    insert_all_rect.push_back(std::make_tuple(bg_rect, std::make_pair(net_idx, mental_idx), len));
                    int32_t rect_width = p_rect.getWidth();
                    if(rect_width > max_eol_width || rect_width < min_width || p_rect.getRectDirection() != layer_dir){
                        continue;
                    }
                    check_p_rect_net_menta_len_list.push_back(std::make_tuple(p_rect, std::make_pair(net_idx, mental_idx), std::make_pair(len, is_end)));
                }
                mental_idx++;
            }
        }
        bg_routing_rect_net_mental_len_all_rtree.insert(insert_all_rect.begin(), insert_all_rect.end());
        std::vector<BGRectInt> insert_rect;
        for(auto& [net_idx, result_gtl_poly_set] : routing_layer_net_result_gtl_poly_set_map[routing_layer_idx]){
            std::vector<GTLRectInt> result_gtl_rect_list;
            gtl::get_max_rectangles(result_gtl_rect_list, result_gtl_poly_set);
            for(auto& gtl_rect : result_gtl_rect_list){
                BGRectInt bg_rect = DRCUTIL.convertToBGRectInt(gtl_rect);
                insert_rect.push_back(bg_rect);
            }
        }
        bg_routing_rect_result_rtree.insert(insert_rect.begin(), insert_rect.end());
        insert_rect.clear();
        //构建绕线below cut层的r树。
        for(auto& [net_idx, result_gtl_poly_set] : cut_layer_net_all_gtl_poly_set_map[routing_layer_idx]){
            std::vector<GTLRectInt> result_gtl_rect_list;
            gtl::get_max_rectangles(result_gtl_rect_list, result_gtl_poly_set);
            for(auto& gtl_rect : result_gtl_rect_list){
                BGRectInt bg_rect = DRCUTIL.convertToBGRectInt(gtl_rect);
                insert_rect.push_back(bg_rect);
            }
        }
        for(auto& [net_idx, result_gtl_poly_set] : cut_layer_net_all_gtl_poly_set_map[routing_layer_idx - 1]){
            std::vector<GTLRectInt> result_gtl_rect_list;
            gtl::get_max_rectangles(result_gtl_rect_list, result_gtl_poly_set);
            for(auto& gtl_rect : result_gtl_rect_list){
                BGRectInt bg_rect = DRCUTIL.convertToBGRectInt(gtl_rect);
                insert_rect.push_back(bg_rect);
            }
        }
        bg_cut_rect_result_rtree.insert(insert_rect.begin(), insert_rect.end());

        
        for(auto& one_p_rect_net_mental : check_p_rect_net_menta_len_list){
            auto [one_p_rect, net_mental, len_end] = one_p_rect_net_mental;
            auto [one_net_idx, one_mental_idx] = net_mental;
            auto [one_len, one_is_end] = len_end;
            int32_t max_par_space = std::max(max_sub_par_space - one_p_rect.getWidth(), max_nsub_par_space);
            mpsew = std::max(max_par_space, max_eol_within);
            //四个区域代码行：东：571-1183 西：1191-1796 北：1804-2379 南：2378-2950
            if(routing_layer_list[routing_layer_idx].isPreferH()){
                int32_t one_east_up = one_len[0];
                int32_t one_east_down = one_len[1];
                int32_t one_west_up = one_len[2];
                int32_t one_west_down = one_len[3];
                //水平方向，生成东西两端最大检测区，先改造东西区域
                //东区：
                if (one_is_end[3]) {
                    BGRectInt bg_max_detect_east(
                        BGPointInt(one_p_rect.get_ur_x() - max_par_within + 1, one_p_rect.get_ll_y() - mpsew + 1),
                        BGPointInt(one_p_rect.get_ur_x() + mesewet - 1, one_p_rect.get_ur_y() + mpsew - 1)
                    );
                    std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_hid_vio_east;
                    bg_routing_rect_net_mental_len_all_rtree.query(bgi::intersects(bg_max_detect_east), std::back_inserter(bg_hid_vio_east));
                    //在大树里面找潜在违例矩形，构建小树查具体规则。
                    bgi::rtree<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>, bgi::quadratic<16>> bg_hid_vio_east_rtree;
                    bg_hid_vio_east_rtree.insert(bg_hid_vio_east.begin(), bg_hid_vio_east.end());
                    //针对每一个规则来查找违例
                    for(auto& one_rule : eol_rules){
                        //四种规则，分四种情况考虑
                        //问题；一个区域只记录一种规则的违例，还是多种（如果有），需要后期对比再研究一下。
                        //获取所有参数，有些是默认的参数，在分支里也用不到，获取了也没关系，省得四个分支获取四次。
                        int32_t eol_space = one_rule.get_eol_space();
                        int32_t eol_width = one_rule.get_eol_width();
                        int32_t eol_within = one_rule.get_eol_within();
                        int32_t ete_space = one_rule.get_ete_space();
                        int32_t par_space = one_rule.get_par_space();
                        int32_t par_within = one_rule.get_par_within();
                        int32_t min_length = one_rule.get_min_length();
                        int32_t enclose_dist = one_rule.get_enclose_dist();
                        int32_t ctm_space = one_rule.get_ctm_space();

                        if(one_rule.is_TWOEDGES() && one_rule.is_SAMEMETAL()){
                            //触发两侧平行边检测逻辑
                            //通过线宽，线长，双侧平行边判断是否使用twoedge的eolspace
                            //问题:minlenth是指的单侧边的长度，需要拿出来单独判断
                            if(one_p_rect.getWidth() > eol_width || one_east_up < min_length || one_east_down < min_length){
                                continue;
                            }
                            BGRectInt det_east_up(
                                BGPointInt(one_p_rect.get_ur_x() - par_within + 1, one_p_rect.get_ur_y() + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ur_y() + par_space - 1)
                            );
                            BGRectInt det_east_down(
                                BGPointInt(one_p_rect.get_ur_x() - par_within + 1, one_p_rect.get_ll_y() - par_space + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ll_y() - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_up_list;
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_down_list;
                            bg_hid_vio_east_rtree.query(bgi::intersects(det_east_up), std::back_inserter(east_up_list));
                            bg_hid_vio_east_rtree.query(bgi::intersects(det_east_down), std::back_inserter(east_down_list));
                            if(east_up_list.empty() || east_down_list.empty()){
                                continue;
                            }
                            //samemental长度覆盖par_within的并行边不算数。两个并行边矩形还要相连成同一个金属。
                            bool is_check_2prl =false;//用于判断是否使用两侧并行检测规则：同一金属，两个并行，两个都不豁免
                            for(auto& up_bg_rect : east_up_list){
                                auto [up_rect, net_mental, len] = up_bg_rect;
                                auto [up_net, up_mental] = net_mental;
                                PlanarRect up_p_rect = DRCUTIL.convertToPlanarRect(up_rect);
                                //排除有y重叠的。排除和被检矩形（one_p_rect）同net的
                                if(one_p_rect.get_ur_y() >= up_p_rect.get_ll_y() || one_net_idx == up_net){
                                    continue;
                                }
                                //覆盖par_within的边获得豁免。
                                if(up_p_rect.get_ll_x() <= one_p_rect.get_ur_x() - par_within && up_p_rect.get_ur_x() >= one_p_rect.get_ur_x()){
                                    continue;
                                }
                                for(auto& down_bg_rect : east_down_list){
                                    auto [down_rect, net_mental, len] = down_bg_rect;
                                    auto [down_net, down_mental] = net_mental;
                                    PlanarRect down_p_rect = DRCUTIL.convertToPlanarRect(down_rect);
                                    if(one_p_rect.get_ll_y() <= down_p_rect.get_ur_y() || one_net_idx == down_net){
                                        continue;
                                    }
                                    if(down_p_rect.get_ll_x() <= one_p_rect.get_ur_x() - par_within && down_p_rect.get_ur_x() >= one_p_rect.get_ur_x()){
                                        continue;
                                    }
                                    if (up_mental != down_mental) {
                                        continue;
                                    }
                                    is_check_2prl = true;
                                    break;
                                }
                                if(is_check_2prl){
                                    break;
                                }
                            }
                            if(!is_check_2prl){
                                continue;
                                //双边检测条件不满足，继续检测其他规则。
                            }
                            //满足双侧平行检测条件，开始检测末端违例
                            BGRectInt detect_east(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_space - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_list;
                            bg_hid_vio_east_rtree.query(bgi::intersects(detect_east), std::back_inserter(east_list));
                            if(east_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            bool have_vio = false;
                            for(auto& other_rect_net_mental : east_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                PlanarRect vio_rect = gen_vio_rect_left_right(one_p_rect, other_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_east;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_east;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_east));
                                bg_hid_vio_east_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_east));
                                if(bg_result_east.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_east) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                                break;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有双侧平行违例产生，就继续判断下一个规则。有就跳过下面的规则。
                            }
                            break;
                            //如果检测到了双边平行违例，该区域就不检测其他违例。
                            //双侧平行边没有端到端
                            //双侧检查完成，问题：short导致的误检，不管；不知道为什么innov不检测的问题；
                        }else if (one_rule.is_ENCLOSECUT()) {
                            //触发cut相关检测逻辑
                            if(one_p_rect.getWidth() > eol_width || one_p_rect.getLength() < min_length){
                                continue;
                            }
                            if(one_rule.is_SUBTRACTEOLWIDTH()){
                                par_space -=one_p_rect.getWidth();
                            }
                            BGRectInt det_east_up(
                                BGPointInt(one_p_rect.get_ur_x() - par_within + 1, one_p_rect.get_ur_y() + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ur_y() + par_space - 1)
                            );
                            BGRectInt det_east_down(
                                BGPointInt(one_p_rect.get_ur_x() - par_within + 1, one_p_rect.get_ll_y() - par_space + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ll_y() - 1)
                            );
                            bool is_check_prl = false;
                            if (one_east_up >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_up_list;
                                bg_hid_vio_east_rtree.query(bgi::intersects(det_east_up), std::back_inserter(east_up_list));
                                if(!east_up_list.empty()){
                                    for(auto& up_bg_rect : east_up_list){
                                        auto [up_rect, net_mental, len] = up_bg_rect;
                                        auto [up_net, up_mental] = net_mental;
                                        PlanarRect up_p_rect = DRCUTIL.convertToPlanarRect(up_rect);
                                        if(one_p_rect.get_ur_y() >= up_p_rect.get_ll_y() || one_net_idx == up_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (one_east_down >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_down_list;
                                bg_hid_vio_east_rtree.query(bgi::intersects(det_east_down), std::back_inserter(east_down_list));
                                if(!east_down_list.empty()){
                                    for(auto& down_bg_rect : east_down_list){
                                        auto [down_rect, net_mental, len] = down_bg_rect;
                                        auto [down_net, down_mental] = net_mental;
                                        PlanarRect down_p_rect = DRCUTIL.convertToPlanarRect(down_rect);
                                        if(one_p_rect.get_ll_y() <= down_p_rect.get_ur_y() || one_net_idx == down_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if(!is_check_prl){
                                continue;
                            }
                            BGRectInt detect_east(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_space - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_list;
                            bg_hid_vio_east_rtree.query(bgi::intersects(detect_east), std::back_inserter(east_list));
                            if(east_list.empty()){
                                continue;
                                //范围内没有违例矩形，继续查下一个规则。
                            }

                            // if (one_p_rect == test_rect1 ) {
                            //     for(auto& r_nm_l : east_list){
                            //         auto [rect, nm, len] = r_nm_l;
                            //         PlanarRect rectp = DRCUTIL.convertToPlanarRect(rect);
                            //         std::cout<<"test hid_vio: ("<<rectp.get_ll_x()<<","<<rectp.get_ll_y()<<")-("
                            //         <<rectp.get_ur_x()<<","<<rectp.get_ur_y()<<")"<<std::endl;
                            //     }
                            // }

                            //先把可能的末端违例矩形找出来，再用cut条件来筛选，那些保留，那些跳过
                            BGRectInt find_cut_rect = DRCUTIL.convertToBGRectInt(one_p_rect);
                            std::vector<BGRectInt> all_cut;
                            bg_cut_rect_result_rtree.query(bgi::intersects(find_cut_rect), std::back_inserter(all_cut));
                            //排除掉没有包含在矩形内的cut。
                            for (auto it = all_cut.begin(); it != all_cut.end(); ) {
                                PlanarRect one_p_cut = DRCUTIL.convertToPlanarRect(*it);
                                if (!DRCUTIL.isInside(one_p_rect, one_p_cut)) {
                                    it = all_cut.erase(it); 
                                } else {
                                    ++it; 
                                }
                            }
                            if(all_cut.empty()){
                                continue;
                                //如果不包含cut，就可以直接检测下一个规则了
                            }
                            bool have_vio = false;
                            std::vector<std::tuple<PlanarRect, int32_t, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : east_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                if(one_p_rect.get_ur_x() > other_p_rect.get_ll_x()){
                                    continue;
                                    //本来可以在生成违例区的时候再排除
                                    //这里提前排除是为了不影响后续cut的判断
                                }
                                //针对每一个末端矩形，结合cut来看是否产生违例。
                                bool is_vio_cutprl;
                                for(auto& one_cut : all_cut){
                                    PlanarRect one_p_cut = DRCUTIL.convertToPlanarRect(one_cut);
                                    if(one_rule.is_ALLCUTS()){
                                        //如果有一个cut不满足要求就不检测
                                        is_vio_cutprl = true;
                                        if(one_p_rect.get_ur_x() - one_p_cut.get_ur_x() >= enclose_dist || other_p_rect.get_ll_x() - one_p_cut.get_ur_x() >= ctm_space){
                                            is_vio_cutprl = false;
                                            break;
                                        }
                                    }
                                    else {
                                        //只要有一个cut满足规则就检测。
                                        is_vio_cutprl = false;
                                        if(one_p_rect.get_ur_x() - one_p_cut.get_ur_x() < enclose_dist && other_p_rect.get_ll_x() - one_p_cut.get_ur_x() < ctm_space){
                                            is_vio_cutprl = true;
                                            break;
                                        }
                                    }
                                }
                                if(!is_vio_cutprl){
                                    continue;
                                }
                                //画出违例区域
                                PlanarRect vio_rect = gen_vio_rect_left_right(one_p_rect, other_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_east;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_east;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_east));
                                bg_hid_vio_east_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_east));
                                if(bg_result_east.empty()){
                                    continue;
                                    //排除env和env产生的违例
                                }
                                bool is_overlap = false;
                                //排除违例区和env，res相交的违例
                                for (auto& all_rect : bg_all_east) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                int32_t eol_rs_space = eol_space;
                                if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                //是端到端就改变rs,去掉距离大于ete_space的违例。
                                    if (other_p_rect.getRectDirection() == Direction::kHorizontal && (other_len[2] != 0 || other_len[3] != 0)) {
                                        eol_rs_space = ete_space;
                                    }
                                }
                                if(vio_rect.getXSpan() >= eol_rs_space){
                                    continue;
                                }
                                violation_list.push_back(std::make_tuple(vio_rect, eol_rs_space, other_net_idx));
                            }
                            for (auto i = violation_list.begin(); i != violation_list.end();) {
                                auto [rect1, rs1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, rs2, net2] = *j;
                                    if (rs1 <= rs2) {
                                        // 保留 i（rs 较小或相等），删除 j
                                        if (rs1 == rs2 && rect1.getXSpan() <= rect2.getXSpan()){
                                            j = violation_list.erase(j);
                                        }
                                        else if (rs1 == rs2 && rect1.getXSpan() > rect2.getXSpan()) {
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; 
                                        }else if (rs1 < rs2) {
                                            j = violation_list.erase(j);
                                        }
                                    } else {
                                        // 保留 j（rs 较小），删除 i
                                        i = violation_list.erase(i);
                                        erased = true;
                                        break; // 退出内层循环，重新开始外层循环
                                    }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, eol_rs_space, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_rs_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有平行,cut违例产生，就继续判断下一个规则。
                                //有平行,cut违例产生，就跳过下面的规则。
                            }
                            break;
                        }else if (one_rule.is_PARALLELEDGE() && one_rule.is_SUBTRACTEOLWIDTH() && !one_rule.is_ENCLOSECUT()) {
                            //触发单侧平行边检测逻辑
                            if(one_p_rect.getWidth() > eol_width || one_p_rect.getLength() < min_length){
                                continue;
                            }
                            if(one_rule.is_SUBTRACTEOLWIDTH()){
                                par_space -=one_p_rect.getWidth();
                            }
                            BGRectInt det_east_up(
                                BGPointInt(one_p_rect.get_ur_x() - par_within + 1, one_p_rect.get_ur_y() + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ur_y() + par_space - 1)
                            );
                            BGRectInt det_east_down(
                                BGPointInt(one_p_rect.get_ur_x() - par_within + 1, one_p_rect.get_ll_y() - par_space + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ll_y() - 1)
                            );
                            bool is_check_prl = false;
                            if (one_east_up >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_up_list;
                                bg_hid_vio_east_rtree.query(bgi::intersects(det_east_up), std::back_inserter(east_up_list));
                                if(!east_up_list.empty()){
                                    for(auto& up_bg_rect : east_up_list){
                                        auto [up_rect, net_mental, len] = up_bg_rect;
                                        auto [up_net, up_mental] = net_mental;
                                        PlanarRect up_p_rect = DRCUTIL.convertToPlanarRect(up_rect);
                                        if(one_p_rect.get_ur_y() >= up_p_rect.get_ll_y() || one_net_idx == up_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (one_east_down >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_down_list;
                                bg_hid_vio_east_rtree.query(bgi::intersects(det_east_down), std::back_inserter(east_down_list));
                                if(!east_down_list.empty()){
                                    for(auto& down_bg_rect : east_down_list){
                                        auto [down_rect, net_mental, len] = down_bg_rect;
                                        auto [down_net, down_mental] = net_mental;
                                        PlanarRect down_p_rect = DRCUTIL.convertToPlanarRect(down_rect);
                                        if(one_p_rect.get_ll_y() <= down_p_rect.get_ur_y() || one_net_idx == down_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if(!is_check_prl){
                                continue;
                            }
                            BGRectInt detect_east(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_space - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_list;
                            bg_hid_vio_east_rtree.query(bgi::intersects(detect_east), std::back_inserter(east_list));
                            if(east_list.empty()){
                                continue;
                                //范围内没有违例矩形，继续查下一个规则。
                            }
                            bool have_vio = false;
                            std::vector<std::pair<PlanarRect, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : east_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                
                                PlanarRect vio_rect = gen_vio_rect_left_right(one_p_rect, other_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_east;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_east;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_east));
                                bg_hid_vio_east_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_east));
                                if(bg_result_east.empty()){
                                    continue;
                                    //排除env和env产生的违例
                                }
                                bool is_overlap = false;
                                //排除违例区和env，res相交的违例
                                for (auto& all_rect : bg_all_east) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                violation_list.push_back({vio_rect, other_net_idx});
                            }
                            for (auto i = violation_list.begin(); i != violation_list.end();) {
                                auto [rect1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, net2] = *j;
                                        if (rect1.getXSpan() <= rect2.getXSpan()) {
                                            // 保留 i（rs 较小或相等），删除 j
                                            j = violation_list.erase(j);
                                        } else {
                                            // 保留 j（rs 较小），删除 i
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; // 退出内层循环，重新开始外层循环
                                        }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有平行,就继续判断下一个规则。
                            }
                            break;
                        }else {
                            //触发最简eol检测逻辑
                            int32_t max_spacing = eol_space;
                            if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                max_spacing = std::max(eol_space, ete_space);
                            }
                            BGRectInt detect_east(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + max_spacing - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> east_list;
                            bg_hid_vio_east_rtree.query(bgi::intersects(detect_east), std::back_inserter(east_list));
                            if(east_list.empty()){
                                continue;
                                //范围内没有违例矩形，继续查下一个规则。
                            }
                            std::vector<std::tuple<PlanarRect, int32_t, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : east_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                PlanarRect vio_rect = gen_vio_rect_left_right(one_p_rect, other_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_east;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_east;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_east));
                                bg_hid_vio_east_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_east));
                                if(bg_result_east.empty()){
                                    continue;
                                    //排除env和env产生的违例
                                }
                                bool is_overlap = false;
                                //排除违例区和env，res相交的违例
                                for (auto& all_rect : bg_all_east) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                int32_t eol_rs_space = eol_space;
                                if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                //是端到端就改变rs,去掉距离大于ete_space的违例。
                                    if (other_p_rect.getRectDirection() == Direction::kHorizontal &&  (other_len[2] != 0 && other_len[3] != 0) && other_p_rect.getWidth() < eol_width) {
                                        eol_rs_space = ete_space;
                                    }
                                }
                                if(vio_rect.getXSpan() >= eol_rs_space){
                                    continue;
                                }
                                violation_list.push_back(std::make_tuple(vio_rect, eol_rs_space, other_net_idx));
                            }
                            for(auto i = violation_list.begin(); i != violation_list.end();){
                                auto [rect1, rs1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, rs2, net2] = *j;
                                    if (rect1.getXSpan() <= rect2.getXSpan()) {
                                        // 保留 i（rs 较小或相等），删除 j
                                        if (rect1.getXSpan() == rect2.getXSpan() && rs1 <= rs2){
                                            j = violation_list.erase(j);
                                        }
                                        else if (rect1.getXSpan() == rect2.getXSpan() && rs1 > rs2) {
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; 
                                        }else if (rect1.getXSpan() < rect2.getXSpan()) {
                                            j = violation_list.erase(j);
                                        }
                                    } else {
                                        // 保留 j（rs 较小），删除 i
                                        i = violation_list.erase(i);
                                        erased = true;
                                        break; // 退出内层循环，重新开始外层循环
                                    }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, eol_rs_space, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_rs_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                            }
                        }
                        //每个最大违例区暂时只检测一种违例规则，后续如果一个区域包含多种违例，再调试。
                    }
                }
                


                // if(one_p_rect == test_rect2){
                //     int32_t x = 1;
                // }

                //西区：
                if (one_is_end[2]) {
                    BGRectInt bg_max_detect_west(
                        BGPointInt(one_p_rect.get_ll_x() - mesewet + 1, one_p_rect.get_ll_y() - mpsew + 1),
                        BGPointInt(one_p_rect.get_ll_x() + max_par_within -1, one_p_rect.get_ur_y() + mpsew -1)
                    );
                    std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_hid_vio_west;
                    bg_routing_rect_net_mental_len_all_rtree.query(bgi::intersects(bg_max_detect_west), std::back_inserter(bg_hid_vio_west));
                    bgi::rtree<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>, bgi::quadratic<16>> bg_hid_vio_west_rtree;
                    bg_hid_vio_west_rtree.insert(bg_hid_vio_west.begin(), bg_hid_vio_west.end());
                    //针对每一个规则来查找违例
                    for(auto& one_rule : eol_rules){
                        //四种规则，分四种情况考虑
                        //问题；一个区域只记录一种规则的违例，还是多种（如果有），需要后期对比再研究一下。
                        //获取所有参数，有些是默认的参数，在分支里也用不到，获取了也没关系，省得四个分支获取四次。
                        int32_t eol_space = one_rule.get_eol_space();
                        int32_t eol_width = one_rule.get_eol_width();
                        int32_t eol_within = one_rule.get_eol_within();
                        int32_t ete_space = one_rule.get_ete_space();
                        int32_t par_space = one_rule.get_par_space();
                        int32_t par_within = one_rule.get_par_within();
                        int32_t min_length = one_rule.get_min_length();
                        int32_t enclose_dist = one_rule.get_enclose_dist();
                        int32_t ctm_space = one_rule.get_ctm_space();

                        if(one_rule.is_TWOEDGES() && one_rule.is_SAMEMETAL()){
                            //触发两侧平行边检测逻辑
                            //通过线宽，线长，双侧平行边判断是否使用twoedge的eolspace
                            if(one_p_rect.getWidth() > eol_width || one_west_up < min_length || one_west_down < min_length){
                                continue;
                            }
                            BGRectInt det_west_up(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() + 1),
                                BGPointInt(one_p_rect.get_ll_x() + par_within - 1, one_p_rect.get_ur_y() + par_space - 1)
                            );
                            BGRectInt det_west_down(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() - par_space + 1),
                                BGPointInt(one_p_rect.get_ll_x() + par_within - 1, one_p_rect.get_ll_y() -1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_up_list;
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_down_list;
                            bg_hid_vio_west_rtree.query(bgi::intersects(det_west_up), std::back_inserter(west_up_list));
                            bg_hid_vio_west_rtree.query(bgi::intersects(det_west_down), std::back_inserter(west_down_list));
                            if(west_up_list.empty() || west_down_list.empty()){
                                continue;
                            }
                            //samemental长度覆盖par_within的并行边不算数。两个并行边矩形还要相连成同一个金属。
                            bool is_check_2prl =false;//用于判断是否使用两侧并行检测规则：同一金属，两个并行，两个都不豁免
                            for(auto& up_bg_rect : west_up_list){
                                auto [up_rect, net_mental, len] = up_bg_rect;
                                auto [up_net, up_mental] = net_mental;
                                PlanarRect up_p_rect = DRCUTIL.convertToPlanarRect(up_rect);
                                //排除有y重叠的。排除和被检矩形（one_p_rect）同net的
                                if(one_p_rect.get_ur_y() >= up_p_rect.get_ll_y() || one_net_idx == up_net){
                                    continue;
                                }
                                //覆盖par_within的边获得豁免。
                                if(up_p_rect.get_ll_x() <= one_p_rect.get_ll_x() && up_p_rect.get_ur_x() >= one_p_rect.get_ll_x() + par_within){
                                    continue;
                                }
                                for(auto& down_bg_rect : west_down_list){
                                    auto [down_rect, net_mental, len] = down_bg_rect;
                                    auto [down_net, down_mental] = net_mental;
                                    PlanarRect down_p_rect = DRCUTIL.convertToPlanarRect(down_rect);
                                    if(one_p_rect.get_ll_y() <= down_p_rect.get_ur_y() || one_net_idx == down_net){
                                        continue;
                                    }
                                    if(down_p_rect.get_ll_x() <= one_p_rect.get_ll_x() && down_p_rect.get_ur_x() >= one_p_rect.get_ll_x() + par_within){
                                        continue;
                                    }
                                    if (up_mental != down_mental) {
                                        continue;
                                    }
                                    is_check_2prl = true;
                                    break;
                                }
                                if(is_check_2prl){
                                    break;
                                }
                            }
                            if(!is_check_2prl){
                                continue;
                                //双边检测条件不满足，继续检测其他规则。
                            }
                            //满足双侧平行检测条件，开始检测末端违例
                            BGRectInt detect_west(
                                BGPointInt(one_p_rect.get_ll_x() - eol_space + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_list;
                            bg_hid_vio_west_rtree.query(bgi::intersects(detect_west), std::back_inserter(west_list));
                            if(west_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            bool have_vio = false;
                            for(auto& other_rect_net_mental : west_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                PlanarRect vio_rect = gen_vio_rect_left_right(other_p_rect, one_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_west;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_west;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_west));
                                bg_hid_vio_west_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_west));
                                if(bg_result_west.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_west) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                                break;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有双侧平行违例产生，就继续判断下一个规则。有就跳过下面的规则。
                            }
                            break;
                        }else if (one_rule.is_ENCLOSECUT()) {
                            //触发cut相关检测逻辑
                            if (one_p_rect.getWidth() > eol_width || one_p_rect.getLength() < min_length) {
                                continue;
                            }
                            if(one_rule.is_SUBTRACTEOLWIDTH()){
                                par_space -=one_p_rect.getWidth();
                            }
                            BGRectInt det_west_up(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() + 1),
                                BGPointInt(one_p_rect.get_ll_x() + par_within - 1, one_p_rect.get_ur_y() + par_space - 1)
                            );
                            BGRectInt det_west_down(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() - par_space + 1),
                                BGPointInt(one_p_rect.get_ll_x() + par_within - 1, one_p_rect.get_ll_y() -1)
                            );
                            bool is_check_prl = false;
                            if (one_west_up >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_up_list;
                                bg_hid_vio_west_rtree.query(bgi::intersects(det_west_up), std::back_inserter(west_up_list));
                                if(!west_up_list.empty()){
                                    for(auto& up_bg_rect : west_up_list){
                                        auto [up_rect, net_mental, len] = up_bg_rect;
                                        auto [up_net, up_mental] = net_mental;
                                        PlanarRect up_p_rect = DRCUTIL.convertToPlanarRect(up_rect);
                                        if(one_p_rect.get_ur_y() >= up_p_rect.get_ll_y() || one_net_idx == up_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (one_west_down >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_down_list;
                                bg_hid_vio_west_rtree.query(bgi::intersects(det_west_down), std::back_inserter(west_down_list));
                                if(!west_down_list.empty()){
                                    for(auto& down_bg_rect : west_down_list){
                                        auto [down_rect, net_mental, len] = down_bg_rect;
                                        auto [down_net, down_mental] = net_mental;
                                        PlanarRect down_p_rect = DRCUTIL.convertToPlanarRect(down_rect);
                                        if(one_p_rect.get_ll_y() <= down_p_rect.get_ur_y() || one_net_idx == down_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (!is_check_prl) {
                                continue;
                            }
                            BGRectInt detect_west(
                                BGPointInt(one_p_rect.get_ll_x() - eol_space + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_list;
                            bg_hid_vio_west_rtree.query(bgi::intersects(detect_west), std::back_inserter(west_list));
                            if(west_list.empty()){
                                continue;
                                //范围内没有违例矩形，继续查下一个规则。
                            }

                            BGRectInt find_cut_rect = DRCUTIL.convertToBGRectInt(one_p_rect);
                            std::vector<BGRectInt> all_cut;
                            bg_cut_rect_result_rtree.query(bgi::intersects(find_cut_rect), std::back_inserter(all_cut));
                            for (auto it = all_cut.begin(); it != all_cut.end(); ) {
                                PlanarRect one_p_cut = DRCUTIL.convertToPlanarRect(*it);
                                if (!DRCUTIL.isInside(one_p_rect, one_p_cut)) {
                                    it = all_cut.erase(it); 
                                } else {
                                    ++it; 
                                }
                            }
                            if(all_cut.empty()){
                                continue;
                                //如果不包含cut，就可以直接检测下一个规则了
                            }
                            bool have_vio = false;
                            std::vector<std::tuple<PlanarRect, int32_t, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : west_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                if(other_p_rect.get_ur_x() > one_p_rect.get_ll_x()){
                                    continue;
                                    //本来可以在生成违例区的时候再排除
                                    //这里提前排除是为了不影响后续cut的判断
                                }
                                //针对每一个末端矩形，结合cut来看是否产生违例。
                                bool is_vio_cutprl;
                                for(auto& one_cut : all_cut){
                                    PlanarRect one_p_cut = DRCUTIL.convertToPlanarRect(one_cut);
                                    if(one_rule.is_ALLCUTS()){
                                        //如果有一个cut不满足要求就不检测
                                        is_vio_cutprl = true;
                                        if(one_p_cut.get_ll_x() - one_p_rect.get_ll_x() >= enclose_dist || one_p_cut.get_ll_x() - other_p_rect.get_ur_x() >= ctm_space){
                                            is_vio_cutprl = false;
                                            break;
                                        }
                                    }
                                    else {
                                        //只要有一个cut满足规则就检测。
                                        is_vio_cutprl = false;
                                        if(one_p_cut.get_ll_x() - one_p_rect.get_ll_x() < enclose_dist && one_p_cut.get_ll_x() - other_p_rect.get_ur_x() < ctm_space){
                                            is_vio_cutprl = true;
                                            break;
                                        }
                                    }
                                }
                                if(!is_vio_cutprl){
                                    continue;
                                }
                                //画出违例区域
                                PlanarRect vio_rect = gen_vio_rect_left_right(other_p_rect, one_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_west;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_west;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_west));
                                bg_hid_vio_west_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_west));
                                if(bg_result_west.empty()){
                                    continue;
                                    //排除env和env产生的违例
                                }
                                bool is_overlap = false;
                                //排除违例区和env，res相交的违例
                                for (auto& all_rect : bg_all_west) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                int32_t eol_rs_space = eol_space;
                                if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                //是端到端就改变rs,去掉距离大于ete_space的违例。
                                    if (other_p_rect.getRectDirection() == Direction::kHorizontal && (other_len[0] != 0 || other_len[1] != 0)) {
                                        eol_rs_space = ete_space;
                                    }
                                }
                                if(vio_rect.getXSpan() >= eol_rs_space){
                                    continue;
                                }
                                violation_list.push_back(std::make_tuple(vio_rect, eol_rs_space, other_net_idx));
                            }
                            for(auto i = violation_list.begin(); i != violation_list.end();){
                                auto [rect1, rs1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, rs2, net2] = *j;
                                    if (rs1 <= rs2) {
                                        // 保留 i（rs 较小或相等），删除 j
                                        if (rs1 == rs2 && rect1.getXSpan() <= rect2.getXSpan()){
                                            j = violation_list.erase(j);
                                        }
                                        else if (rs1 == rs2 && rect1.getXSpan() > rect2.getXSpan()) {
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; 
                                        }else if (rs1 < rs2) {
                                            j = violation_list.erase(j);
                                        }
                                    } else {
                                        // 保留 j（rs 较小），删除 i
                                        i = violation_list.erase(i);
                                        erased = true;
                                        break; // 退出内层循环，重新开始外层循环
                                    }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, eol_rs_space, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_rs_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有平行,cut违例产生，就继续判断下一个规则。
                                //有平行,cut违例产生，就跳过下面的规则。
                            }
                            break;
                        }else if (one_rule.is_PARALLELEDGE() && one_rule.is_SUBTRACTEOLWIDTH() && !one_rule.is_ENCLOSECUT()) {
                            //触发单侧平行边检测逻辑
                            if (one_p_rect.getWidth() > eol_width || one_p_rect.getLength() < min_length) {
                                continue;
                            }
                            if(one_rule.is_SUBTRACTEOLWIDTH()){
                                par_space -=one_p_rect.getWidth();
                            }
                            BGRectInt det_west_up(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() + 1),
                                BGPointInt(one_p_rect.get_ll_x() + par_within - 1, one_p_rect.get_ur_y() + par_space - 1)
                            );
                            BGRectInt det_west_down(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() - par_space + 1),
                                BGPointInt(one_p_rect.get_ll_x() + par_within - 1, one_p_rect.get_ll_y() -1)
                            );
                            bool is_check_prl = false;
                            if (one_west_up >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_up_list;
                                bg_hid_vio_west_rtree.query(bgi::intersects(det_west_up), std::back_inserter(west_up_list));
                                if(!west_up_list.empty()){
                                    for(auto& up_bg_rect : west_up_list){
                                        auto [up_rect, net_mental, len] = up_bg_rect;
                                        auto [up_net, up_mental] = net_mental;
                                        PlanarRect up_p_rect = DRCUTIL.convertToPlanarRect(up_rect);
                                        if(one_p_rect.get_ur_y() >= up_p_rect.get_ll_y() || one_net_idx == up_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (one_west_down >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_down_list;
                                bg_hid_vio_west_rtree.query(bgi::intersects(det_west_down), std::back_inserter(west_down_list));
                                if(!west_down_list.empty()){
                                    for(auto& down_bg_rect : west_down_list){
                                        auto [down_rect, net_mental, len] = down_bg_rect;
                                        auto [down_net, down_mental] = net_mental;
                                        PlanarRect down_p_rect = DRCUTIL.convertToPlanarRect(down_rect);
                                        if(one_p_rect.get_ll_y() <= down_p_rect.get_ur_y() || one_net_idx == down_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (!is_check_prl) {
                                continue;
                            }
                            BGRectInt detect_west(
                                BGPointInt(one_p_rect.get_ll_x() - eol_space + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_list;
                            bg_hid_vio_west_rtree.query(bgi::intersects(detect_west), std::back_inserter(west_list));
                            if(west_list.empty()){
                                continue;
                                //范围内没有违例矩形，继续查下一个规则。
                            }


                            // if (one_p_rect == test_rect2 ) {
                            //     for(auto& r_nm_l : west_list){
                            //         auto [rect, nm, len] = r_nm_l;
                            //         PlanarRect rectp = DRCUTIL.convertToPlanarRect(rect);
                            //         std::cout<<"test2 hid_vio: ("<<rectp.get_ll_x()<<","<<rectp.get_ll_y()<<")-("
                            //         <<rectp.get_ur_x()<<","<<rectp.get_ur_y()<<")"<<std::endl;
                            //     }
                            // }
                            bool have_vio = false;
                            std::vector<std::pair<PlanarRect, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : west_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                PlanarRect vio_rect = gen_vio_rect_left_right(other_p_rect, one_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_west;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_west;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_west));
                                bg_hid_vio_west_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_west));
                                if(bg_result_west.empty()){
                                    continue;
                                    //排除env和env产生的违例
                                }
                                bool is_overlap = false;
                                //排除违例区和env，res相交的违例
                                for (auto& all_rect : bg_all_west) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                violation_list.push_back({vio_rect, other_net_idx});
                            }
                            for (auto i = violation_list.begin(); i != violation_list.end();) {
                                auto [rect1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, net2] = *j;
                                    if (rect1.getXSpan() <= rect2.getXSpan()) {
                                        // 保留 i（rs 较小或相等），删除 j
                                        j = violation_list.erase(j);
                                    } else {
                                        // 保留 j（rs 较小），删除 i
                                        i = violation_list.erase(i);
                                        erased = true;
                                        break; // 退出内层循环，重新开始外层循环
                                    }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有平行,就继续判断下一个规则。
                            }
                            break;
                        }else {
                            //触发最简eol检测逻辑
                            int32_t max_spacing = eol_space;
                            if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                max_spacing = std::max(eol_space, ete_space);
                            }
                            BGRectInt detect_west(
                                BGPointInt(one_p_rect.get_ll_x() - max_spacing + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> west_list;
                            bg_hid_vio_west_rtree.query(bgi::intersects(detect_west), std::back_inserter(west_list));
                            if(west_list.empty()){
                                continue;
                                //范围内没有违例矩形，继续查下一个规则。
                            }
                            std::vector<std::tuple<PlanarRect, int32_t, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : west_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                PlanarRect vio_rect = gen_vio_rect_left_right(other_p_rect, one_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_west;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_west;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_west));
                                bg_hid_vio_west_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_west));
                                if(bg_result_west.empty()){
                                    continue;
                                    //排除env和env产生的违例
                                }
                                bool is_overlap = false;
                                //排除违例区和env，res相交的违例
                                for (auto& all_rect : bg_all_west) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                int32_t eol_rs_space = eol_space;
                                if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                //是端到端就改变rs,去掉距离大于ete_space的违例。
                                    if (other_p_rect.getRectDirection() == Direction::kHorizontal && (other_len[0] != 0 && other_len[1] != 0) && other_p_rect.getWidth() < eol_width) {
                                        eol_rs_space = ete_space;
                                    }
                                }
                                if(vio_rect.getXSpan() >= eol_rs_space){
                                    continue;
                                }
                                violation_list.push_back(std::make_tuple(vio_rect, eol_rs_space, other_net_idx));
                                
                            }
                            for(auto i = violation_list.begin(); i != violation_list.end();){
                                auto [rect1, rs1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, rs2, net2] = *j;
                                    if (rect1.getXSpan() <= rect2.getXSpan()) {
                                        // 保留 i（rs 较小或相等），删除 j
                                        if (rect1.getXSpan() == rect2.getXSpan() && rs1 <= rs2){
                                            j = violation_list.erase(j);
                                        }
                                        else if (rect1.getXSpan() == rect2.getXSpan() && rs1 > rs2) {
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; 
                                        }else if (rect1.getXSpan() < rect2.getXSpan()) {
                                            j = violation_list.erase(j);
                                        }
                                    } else {
                                        // 保留 j（rs 较小），删除 i
                                        i = violation_list.erase(i);
                                        erased = true;
                                        break; // 退出内层循环，重新开始外层循环
                                    }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, eol_rs_space, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_rs_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                            }
                        }                    
                    }          
                }          
            }
            else {
                //垂直方向，使用maxspacing生成南北两端检测区，检测所有潜在违例
                int32_t one_north_left = one_len[4];
                int32_t one_north_rihgt = one_len[5];
                int32_t one_south_left = one_len[6];
                int32_t one_south_right = one_len[7];
                //北区：
                if (one_is_end[0]) {
                    BGRectInt bg_max_detect_north(
                        BGPointInt(one_p_rect.get_ll_x() - mpsew + 1, one_p_rect.get_ur_y() - max_par_within + 1),
                        BGPointInt(one_p_rect.get_ur_x() + mpsew - 1, one_p_rect.get_ur_y() + mesewet - 1)
                    );
                    std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_hid_vio_north;
                    bg_routing_rect_net_mental_len_all_rtree.query(bgi::intersects(bg_max_detect_north), std::back_inserter(bg_hid_vio_north));
                    bgi::rtree<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>, bgi::quadratic<16>> bg_hid_vio_north_rtree;
                    bg_hid_vio_north_rtree.insert(bg_hid_vio_north.begin(), bg_hid_vio_north.end());
                    for(auto& one_rule : eol_rules){
                        int32_t eol_space = one_rule.get_eol_space();
                        int32_t eol_width = one_rule.get_eol_width();
                        int32_t eol_within = one_rule.get_eol_within();
                        int32_t ete_space = one_rule.get_ete_space();
                        int32_t par_space = one_rule.get_par_space();
                        int32_t par_within = one_rule.get_par_within();
                        int32_t min_length = one_rule.get_min_length();
                        int32_t enclose_dist = one_rule.get_enclose_dist();
                        int32_t ctm_space = one_rule.get_ctm_space();
                        if(one_rule.is_TWOEDGES() && one_rule.is_SAMEMETAL()){
                            if(one_p_rect.getWidth() > eol_width || one_north_left < min_length || one_north_rihgt < min_length){
                                continue;
                            }
                            BGRectInt det_north_left(
                                BGPointInt(one_p_rect.get_ll_x() - par_space + 1, one_p_rect.get_ur_y() - par_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            BGRectInt det_north_right(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ur_y() - par_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + par_space - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_left_list;
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_right_list;
                            bg_hid_vio_north_rtree.query(bgi::intersects(det_north_left), std::back_inserter(north_left_list));
                            bg_hid_vio_north_rtree.query(bgi::intersects(det_north_right), std::back_inserter(north_right_list));
                            if(north_left_list.empty() || north_right_list.empty()){
                                continue;
                            }
                            bool is_check_2prl =false;//用于判断是否使用两侧并行检测规则：同一金属，两个并行，两个都不豁免
                            for(auto& left_bg_rect : north_left_list){
                                auto [left_rect, net_mental, len] = left_bg_rect;
                                auto [left_net, left_mental] = net_mental;
                                PlanarRect left_p_rect = DRCUTIL.convertToPlanarRect(left_rect);
                                if(one_p_rect.get_ll_x() <= left_p_rect.get_ur_x() || one_net_idx == left_net){
                                    continue;
                                }
                                //覆盖par_within的边获得豁免。
                                if(left_p_rect.get_ll_y() <= one_p_rect.get_ur_y() - par_within && left_p_rect.get_ur_y() >= one_p_rect.get_ur_y()){
                                    continue;
                                }
                                for(auto& right_bg_rect : north_right_list){
                                    auto [right_rect, net_mental, len] = right_bg_rect;
                                    auto [right_net, right_mental] = net_mental;
                                    PlanarRect right_p_rect = DRCUTIL.convertToPlanarRect(right_rect);
                                    if(one_p_rect.get_ur_x() >= right_p_rect.get_ll_x() || one_net_idx == right_net){
                                        continue;
                                    }
                                    if(right_p_rect.get_ll_y() <= one_p_rect.get_ur_y() - par_within && right_p_rect.get_ur_y() >= one_p_rect.get_ur_y()){
                                        continue;
                                    }
                                    if (left_mental != right_mental) {
                                        continue;
                                    }
                                    is_check_2prl = true;
                                    break;
                                }
                                if(is_check_2prl){
                                    break;
                                }
                            }
                            if(!is_check_2prl){
                                continue;
                                //双边检测条件不满足，继续检测其他规则。
                            }
                            BGRectInt detect_north(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ur_y() + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ur_y() + eol_space - 1) 
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_list;
                            bg_hid_vio_north_rtree.query(bgi::intersects(detect_north), std::back_inserter(north_list));
                            if(north_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            bool have_vio = false;
                            for(auto& other_rect_net_mental : north_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                PlanarRect vio_rect = gen_vio_rect_down_up(one_p_rect, other_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_north;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_north;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_north));
                                bg_hid_vio_north_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_north));
                                if(bg_result_north.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_north) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                                break;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有双侧平行违例产生，就继续判断下一个规则。有就跳过下面的规则。
                            }
                            break;
                        }else if (one_rule.is_ENCLOSECUT()) {
                            if(one_p_rect.getWidth() > eol_width || one_p_rect.getLength() < min_length){
                                continue;
                            }
                            if(one_rule.is_SUBTRACTEOLWIDTH()){
                                par_space -=one_p_rect.getWidth();
                            }
                            BGRectInt det_north_left(
                                BGPointInt(one_p_rect.get_ll_x() - par_space + 1, one_p_rect.get_ur_y() - par_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            BGRectInt det_north_right(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ur_y() - par_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + par_space - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            bool is_check_prl = false;
                            if (one_north_left >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_left_list;
                                bg_hid_vio_north_rtree.query(bgi::intersects(det_north_left), std::back_inserter(north_left_list));
                                if(!north_left_list.empty()){
                                    for(auto& left_bg_rect : north_left_list){
                                        auto [left_rect, net_mental, len] = left_bg_rect;
                                        auto [left_net, left_mental] = net_mental;
                                        PlanarRect left_p_rect = DRCUTIL.convertToPlanarRect(left_rect);
                                        if(one_p_rect.get_ll_x() <= left_p_rect.get_ur_x() || one_net_idx == left_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (one_north_rihgt >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_right_list;
                                bg_hid_vio_north_rtree.query(bgi::intersects(det_north_right), std::back_inserter(north_right_list));
                                if(!north_right_list.empty()){
                                    for(auto& right_bg_rect : north_right_list){
                                        auto [right_rect, net_mental, len] = right_bg_rect;
                                        auto [right_net, right_mental] = net_mental;
                                        PlanarRect right_p_rect = DRCUTIL.convertToPlanarRect(right_rect);
                                        if(one_p_rect.get_ur_x() >= right_p_rect.get_ll_x() || one_net_idx == right_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if(!is_check_prl){
                                continue;
                            }
                            BGRectInt detect_north(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ur_y() + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ur_y() + eol_space - 1) 
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_list;
                            bg_hid_vio_north_rtree.query(bgi::intersects(detect_north), std::back_inserter(north_list));
                            if(north_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            BGRectInt find_cut_rect = DRCUTIL.convertToBGRectInt(one_p_rect);
                            std::vector<BGRectInt> all_cut;
                            bg_cut_rect_result_rtree.query(bgi::intersects(find_cut_rect), std::back_inserter(all_cut));
                            //排除掉没有包含在矩形内的cut。
                            for (auto it = all_cut.begin(); it != all_cut.end(); ) {
                                PlanarRect one_p_cut = DRCUTIL.convertToPlanarRect(*it);
                                if (!DRCUTIL.isInside(one_p_rect, one_p_cut)) {
                                    it = all_cut.erase(it); 
                                } else {
                                    ++it; 
                                }
                            }
                            if(all_cut.empty()){
                                continue;
                                //如果不包含cut，就可以直接检测下一个规则了
                            }
                            bool have_vio = false;
                            std::vector<std::tuple<PlanarRect, int32_t, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : north_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                if(one_p_rect.get_ur_y() > other_p_rect.get_ll_y()){
                                    continue;
                                    //本来可以在生成违例区的时候再排除
                                    //这里提前排除是为了不影响后续cut的判断
                                }
                                //针对每一个末端矩形，结合cut来看是否产生违例。
                                bool is_vio_cutprl;
                                for(auto& one_cut : all_cut){
                                    PlanarRect one_p_cut = DRCUTIL.convertToPlanarRect(one_cut);
                                    if(one_rule.is_ALLCUTS()){
                                        //如果有一个cut不满足要求就不检测
                                        is_vio_cutprl = true;
                                        if(one_p_rect.get_ur_y() - one_p_cut.get_ur_y() >= enclose_dist || other_p_rect.get_ll_y() - one_p_cut.get_ur_y() >= ctm_space){
                                            is_vio_cutprl = false;
                                            break;
                                        }
                                    }
                                    else {
                                        //只要有一个cut满足规则就检测。
                                        is_vio_cutprl = false;
                                        if(one_p_rect.get_ur_y() - one_p_cut.get_ur_y() < enclose_dist && other_p_rect.get_ll_y() - one_p_cut.get_ur_y() < ctm_space){
                                            is_vio_cutprl = true;
                                            break;
                                        }
                                    }
                                }
                                if(!is_vio_cutprl){
                                    continue;
                                }
                                //画出违例区域
                                PlanarRect vio_rect = gen_vio_rect_down_up(one_p_rect, other_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_north;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_north;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_north));
                                bg_hid_vio_north_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_north));
                                if(bg_result_north.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_north) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                int32_t eol_rs_space = eol_space;
                                if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                //是端到端就改变rs,去掉距离大于ete_space的违例。
                                    if (other_p_rect.getRectDirection() == Direction::kVertical && (other_len[6] != 0 || other_len[7] != 0)) {
                                        eol_rs_space = ete_space;
                                    }
                                }
                                if(vio_rect.getYSpan() >= eol_rs_space){
                                    continue;
                                }
                                violation_list.push_back(std::make_tuple(vio_rect, eol_rs_space, other_net_idx));
                            }
                            for (auto i = violation_list.begin(); i != violation_list.end();) {
                                auto [rect1, rs1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, rs2, net2] = *j;
                                    if (rs1 <= rs2) {
                                        // 保留 i（rs 较小或相等），删除 j
                                        if (rs1 == rs2 && rect1.getYSpan() <= rect2.getYSpan()){
                                            j = violation_list.erase(j);
                                        }
                                        else if (rs1 == rs2 && rect1.getYSpan() > rect2.getYSpan()) {
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; 
                                        }else if (rs1 < rs2) {
                                            j = violation_list.erase(j);
                                        }
                                    } else {
                                        // 保留 j（rs 较小），删除 i
                                        i = violation_list.erase(i);
                                        erased = true;
                                        break; // 退出内层循环，重新开始外层循环
                                    }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, eol_rs_space, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_rs_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有平行,cut违例产生，就继续判断下一个规则。
                                //有平行,cut违例产生，就跳过下面的规则。
                            }
                            break;
                        }else if (one_rule.is_PARALLELEDGE() && one_rule.is_SUBTRACTEOLWIDTH() && !one_rule.is_ENCLOSECUT()) {
                            if(one_p_rect.getWidth() > eol_width || one_p_rect.getLength() < min_length){
                                continue;
                            }
                            if(one_rule.is_SUBTRACTEOLWIDTH()){
                                par_space -= one_p_rect.getWidth();
                            }
                            BGRectInt det_north_left(
                                BGPointInt(one_p_rect.get_ll_x() - par_space + 1, one_p_rect.get_ur_y() - par_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            BGRectInt det_north_right(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ur_y() - par_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + par_space - 1, one_p_rect.get_ur_y() + eol_within - 1)
                            );
                            bool is_check_prl = false;
                            if (one_north_left >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_left_list;
                                bg_hid_vio_north_rtree.query(bgi::intersects(det_north_left), std::back_inserter(north_left_list));
                                if(!north_left_list.empty()){
                                    for(auto& left_bg_rect : north_left_list){
                                        auto [left_rect, net_mental, len] = left_bg_rect;
                                        auto [left_net, left_mental] = net_mental;
                                        PlanarRect left_p_rect = DRCUTIL.convertToPlanarRect(left_rect);
                                        if(one_p_rect.get_ll_x() <= left_p_rect.get_ur_x() || one_net_idx == left_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (one_north_rihgt >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_right_list;
                                bg_hid_vio_north_rtree.query(bgi::intersects(det_north_right), std::back_inserter(north_right_list));
                                if(!north_right_list.empty()){
                                    for(auto& right_bg_rect : north_right_list){
                                        auto [right_rect, net_mental, len] = right_bg_rect;
                                        auto [right_net, right_mental] = net_mental;
                                        PlanarRect right_p_rect = DRCUTIL.convertToPlanarRect(right_rect);
                                        if(one_p_rect.get_ur_x() >= right_p_rect.get_ll_x() || one_net_idx == right_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if(!is_check_prl){
                                continue;
                            }
                            BGRectInt detect_north(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ur_y() + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ur_y() + eol_space - 1) 
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_list;
                            bg_hid_vio_north_rtree.query(bgi::intersects(detect_north), std::back_inserter(north_list));
                            if(north_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            bool have_vio = false;
                            std::vector<std::pair<PlanarRect, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : north_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                
                                PlanarRect vio_rect = gen_vio_rect_down_up(one_p_rect, other_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_north;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_north;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_north));
                                bg_hid_vio_north_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_north));
                                if(bg_result_north.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_north) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                violation_list.push_back({vio_rect, other_net_idx});
                            }
                            for (auto i = violation_list.begin(); i != violation_list.end();) {
                                auto [rect1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, net2] = *j;
                                        if (rect1.getYSpan() <= rect2.getYSpan()) {
                                            // 保留 i（rs 较小或相等），删除 j
                                            j = violation_list.erase(j);
                                        } else {
                                            // 保留 j（rs 较小），删除 i
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; // 退出内层循环，重新开始外层循环
                                        }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有平行,就继续判断下一个规则。
                            }
                            break;
                        }else {
                            int32_t max_spacing = eol_space;
                            if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                max_spacing = std::max(eol_space, ete_space);
                            }
                            BGRectInt detect_north(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ur_y() + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ur_y() + max_spacing - 1) 
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> north_list;
                            bg_hid_vio_north_rtree.query(bgi::intersects(detect_north), std::back_inserter(north_list));
                            if(north_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            std::vector<std::tuple<PlanarRect, int32_t, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : north_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                
                                PlanarRect vio_rect = gen_vio_rect_down_up(one_p_rect, other_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_north;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_north;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_north));
                                bg_hid_vio_north_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_north));
                                if(bg_result_north.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_north) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                int32_t eol_rs_space = eol_space;
                                if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                //是端到端就改变rs,去掉距离大于ete_space的违例。
                                    if (other_p_rect.getRectDirection() == Direction::kVertical && (other_len[6] != 0 || other_len[7] != 0)) {
                                        eol_rs_space = ete_space;
                                    }
                                }
                                if(vio_rect.getYSpan() >= eol_rs_space){
                                    continue;
                                }
                                violation_list.push_back(std::make_tuple(vio_rect, eol_rs_space, other_net_idx));
                            }
                            for(auto i = violation_list.begin(); i != violation_list.end();){
                                auto [rect1, rs1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, rs2, net2] = *j;
                                    if (rect1.getYSpan() <= rect2.getYSpan()) {
                                        // 保留 i（rs 较小或相等），删除 j
                                        if (rect1.getYSpan() == rect2.getYSpan() && rs1 <= rs2){
                                            j = violation_list.erase(j);
                                        }
                                        else if (rect1.getYSpan() == rect2.getYSpan() && rs1 > rs2) {
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; 
                                        }else if (rect1.getYSpan() < rect2.getYSpan()) {
                                            j = violation_list.erase(j);
                                        }
                                    } else {
                                        // 保留 j（rs 较小），删除 i
                                        i = violation_list.erase(i);
                                        erased = true;
                                        break; // 退出内层循环，重新开始外层循环
                                    }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, eol_rs_space, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_rs_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                            }
                        }
                    }
                }

                //南区：
                if (one_is_end[1]) {
                    BGRectInt bg_max_detect_south(
                        BGPointInt(one_p_rect.get_ll_x() - mpsew + 1, one_p_rect.get_ll_y() - mesewet + 1),
                        BGPointInt(one_p_rect.get_ur_x() + mpsew - 1, one_p_rect.get_ll_y() + max_par_within - 1)
                    );
                    std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_hid_vio_south;
                    bg_routing_rect_net_mental_len_all_rtree.query(bgi::intersects(bg_max_detect_south), std::back_inserter(bg_hid_vio_south));
                    bgi::rtree<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>, bgi::quadratic<16>> bg_hid_vio_south_rtree;
                    bg_hid_vio_south_rtree.insert(bg_hid_vio_south.begin(), bg_hid_vio_south.end());
                    for(auto& one_rule : eol_rules){
                        int32_t eol_space = one_rule.get_eol_space();
                        int32_t eol_width = one_rule.get_eol_width();
                        int32_t eol_within = one_rule.get_eol_within();
                        int32_t ete_space = one_rule.get_ete_space();
                        int32_t par_space = one_rule.get_par_space();
                        int32_t par_within = one_rule.get_par_within();
                        int32_t min_length = one_rule.get_min_length();
                        int32_t enclose_dist = one_rule.get_enclose_dist();
                        int32_t ctm_space = one_rule.get_ctm_space();
                        if(one_rule.is_TWOEDGES() && one_rule.is_SAMEMETAL()){
                            if(one_p_rect.getWidth() > eol_width || one_south_left < min_length || one_south_right < min_length){
                                continue;
                            }
                            BGRectInt det_south_left(
                                BGPointInt(one_p_rect.get_ll_x() - par_space + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ll_y() + par_within - 1)
                            );
                            BGRectInt det_south_right(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + par_space - 1, one_p_rect.get_ll_y() + par_within - 1)
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_left_list;
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_right_list;
                            bg_hid_vio_south_rtree.query(bgi::intersects(det_south_left), std::back_inserter(south_left_list));
                            bg_hid_vio_south_rtree.query(bgi::intersects(det_south_right), std::back_inserter(south_right_list));
                            if(south_left_list.empty() || south_right_list.empty()){
                                continue;
                            }
                            bool is_check_2prl =false;//用于判断是否使用两侧并行检测规则：同一金属，两个并行，两个都不豁免
                            for(auto& left_bg_rect : south_left_list){
                                auto [left_rect, net_mental, len] = left_bg_rect;
                                auto [left_net, left_mental] = net_mental;
                                PlanarRect left_p_rect = DRCUTIL.convertToPlanarRect(left_rect);
                                if(one_p_rect.get_ll_x() <= left_p_rect.get_ur_x() || one_net_idx == left_net){
                                    continue;
                                }
                                //覆盖par_within的边获得豁免。
                                if(left_p_rect.get_ll_y() <= one_p_rect.get_ll_y() && left_p_rect.get_ur_y() >= one_p_rect.get_ll_y() + par_within){
                                    continue;
                                }
                                for(auto& right_bg_rect : south_right_list){
                                    auto [right_rect, net_mental, len] = right_bg_rect;
                                    auto [right_net, right_mental] = net_mental;
                                    PlanarRect right_p_rect = DRCUTIL.convertToPlanarRect(right_rect);
                                    if(one_p_rect.get_ur_x() >= right_p_rect.get_ll_x() || one_net_idx == right_net){
                                        continue;
                                    }
                                    if(right_p_rect.get_ll_y() <= one_p_rect.get_ll_y() && right_p_rect.get_ur_y() >= one_p_rect.get_ll_y() + par_within){
                                        continue;
                                    }
                                    if (left_mental != right_mental) {
                                        continue;
                                    }
                                    is_check_2prl = true;
                                    break;
                                }
                                if(is_check_2prl){
                                    break;
                                }
                            }
                            if(!is_check_2prl){
                                continue;
                                //双边检测条件不满足，继续检测其他规则。
                            }
                            BGRectInt detect_south(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() - eol_space + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ll_y() - 1) 
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_list;
                            bg_hid_vio_south_rtree.query(bgi::intersects(detect_south), std::back_inserter(south_list));
                            if(south_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            bool have_vio = false;
                            for(auto& other_rect_net_mental : south_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                PlanarRect vio_rect = gen_vio_rect_down_up(other_p_rect, one_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_south;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_south;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_south));
                                bg_hid_vio_south_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_south));
                                if(bg_result_south.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_south) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                                break;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有双侧平行违例产生，就继续判断下一个规则。有就跳过下面的规则。
                            }
                            break;
                        }else if (one_rule.is_ENCLOSECUT()) {
                            if(one_p_rect.getWidth() > eol_width || one_p_rect.getLength() < min_length){
                                continue;
                            }
                            if(one_rule.is_SUBTRACTEOLWIDTH()){
                                par_space -=one_p_rect.getWidth();
                            }
                            BGRectInt det_south_left(
                                BGPointInt(one_p_rect.get_ll_x() - par_space + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ll_y() + par_within - 1)
                            );
                            BGRectInt det_south_right(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + par_space - 1, one_p_rect.get_ll_y() + par_within - 1)
                            );
                            bool is_check_prl = false;
                            if (one_south_left >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_left_list;
                                bg_hid_vio_south_rtree.query(bgi::intersects(det_south_left), std::back_inserter(south_left_list));
                                if(!south_left_list.empty()){
                                    for(auto& left_bg_rect : south_left_list){
                                        auto [left_rect, net_mental, len] = left_bg_rect;
                                        auto [left_net, left_mental] = net_mental;
                                        PlanarRect left_p_rect = DRCUTIL.convertToPlanarRect(left_rect);
                                        if(one_p_rect.get_ll_x() <= left_p_rect.get_ur_x() || one_net_idx == left_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (one_south_right >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_right_list;
                                bg_hid_vio_south_rtree.query(bgi::intersects(det_south_right), std::back_inserter(south_right_list));
                                if(!south_right_list.empty()){
                                    for(auto& right_bg_rect : south_right_list){
                                        auto [right_rect, net_mental, len] = right_bg_rect;
                                        auto [right_net, right_mental] = net_mental;
                                        PlanarRect right_p_rect = DRCUTIL.convertToPlanarRect(right_rect);
                                        if(one_p_rect.get_ur_x() >= right_p_rect.get_ll_x() || one_net_idx == right_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if(!is_check_prl){
                                continue;
                            }
                            BGRectInt detect_south(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() - eol_space + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ll_y() - 1) 
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_list;
                            bg_hid_vio_south_rtree.query(bgi::intersects(detect_south), std::back_inserter(south_list));
                            if(south_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            BGRectInt find_cut_rect = DRCUTIL.convertToBGRectInt(one_p_rect);
                            std::vector<BGRectInt> all_cut;
                            bg_cut_rect_result_rtree.query(bgi::intersects(find_cut_rect), std::back_inserter(all_cut));
                            //排除掉没有包含在矩形内的cut。
                            for (auto it = all_cut.begin(); it != all_cut.end(); ) {
                                PlanarRect one_p_cut = DRCUTIL.convertToPlanarRect(*it);
                                if (!DRCUTIL.isInside(one_p_rect, one_p_cut)) {
                                    it = all_cut.erase(it); 
                                } else {
                                    ++it; 
                                }
                            }
                            if(all_cut.empty()){
                                continue;
                                //如果不包含cut，就可以直接检测下一个规则了
                            }
                            bool have_vio = false;
                            std::vector<std::tuple<PlanarRect, int32_t, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : south_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                if(one_p_rect.get_ll_y() < other_p_rect.get_ur_y()){
                                    continue;
                                    //本来可以在生成违例区的时候再排除
                                    //这里提前排除是为了不影响后续cut的判断
                                }
                                //针对每一个末端矩形，结合cut来看是否产生违例。
                                bool is_vio_cutprl;
                                for(auto& one_cut : all_cut){
                                    PlanarRect one_p_cut = DRCUTIL.convertToPlanarRect(one_cut);
                                    if(one_rule.is_ALLCUTS()){
                                        //如果有一个cut不满足要求就不检测
                                        is_vio_cutprl = true;
                                        if(one_p_cut.get_ll_y() - one_p_rect.get_ll_y() >= enclose_dist || one_p_cut.get_ll_y() - other_p_rect.get_ur_y() >= ctm_space){
                                            is_vio_cutprl = false;
                                            break;
                                        }
                                    }
                                    else {
                                        //只要有一个cut满足规则就检测。
                                        is_vio_cutprl = false;
                                        if(one_p_cut.get_ll_y() - one_p_rect.get_ll_y() < enclose_dist && one_p_cut.get_ll_y() - other_p_rect.get_ur_y() < ctm_space){
                                            is_vio_cutprl = true;
                                            break;
                                        }
                                    }
                                }
                                if(!is_vio_cutprl){
                                    continue;
                                }
                                //画出违例区域
                                PlanarRect vio_rect = gen_vio_rect_down_up(other_p_rect, one_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_south;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_south;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_south));
                                bg_hid_vio_south_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_south));
                                if(bg_result_south.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_south) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                int32_t eol_rs_space = eol_space;
                                if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                //是端到端就改变rs,去掉距离大于ete_space的违例。
                                    if (other_p_rect.getRectDirection() == Direction::kVertical && (other_len[4] != 0 || other_len[5] != 0)) {
                                        eol_rs_space = ete_space;
                                    }
                                }
                                if(vio_rect.getYSpan() >= eol_rs_space){
                                    continue;
                                }
                                violation_list.push_back(std::make_tuple(vio_rect, eol_rs_space, other_net_idx));
                            }
                            for (auto i = violation_list.begin(); i != violation_list.end();) {
                                auto [rect1, rs1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, rs2, net2] = *j;
                                    if (rs1 <= rs2) {
                                        // 保留 i（rs 较小或相等），删除 j
                                        if (rs1 == rs2 && rect1.getYSpan() <= rect2.getYSpan()){
                                            j = violation_list.erase(j);
                                        }
                                        else if (rs1 == rs2 && rect1.getYSpan() > rect2.getYSpan()) {
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; 
                                        }else if (rs1 < rs2) {
                                            j = violation_list.erase(j);
                                        }
                                    } else {
                                        // 保留 j（rs 较小），删除 i
                                        i = violation_list.erase(i);
                                        erased = true;
                                        break; // 退出内层循环，重新开始外层循环
                                    }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, eol_rs_space, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_rs_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有平行,cut违例产生，就继续判断下一个规则。
                                //有平行,cut违例产生，就跳过下面的规则。
                            }
                            break;
                        }else if (one_rule.is_PARALLELEDGE() && one_rule.is_SUBTRACTEOLWIDTH() && !one_rule.is_ENCLOSECUT()){
                            if(one_p_rect.getWidth() > eol_width || one_p_rect.getLength() < min_length){
                                continue;
                            }
                            if(one_rule.is_SUBTRACTEOLWIDTH()){
                                par_space -= one_p_rect.getWidth();
                            }
                            BGRectInt det_south_left(
                                BGPointInt(one_p_rect.get_ll_x() - par_space + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ll_x() - 1, one_p_rect.get_ll_y() + par_within - 1)
                            );
                            BGRectInt det_south_right(
                                BGPointInt(one_p_rect.get_ur_x() + 1, one_p_rect.get_ll_y() - eol_within + 1),
                                BGPointInt(one_p_rect.get_ur_x() + par_space - 1, one_p_rect.get_ll_y() + par_within - 1)
                            );
                            bool is_check_prl = false;
                            if (one_south_left >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_left_list;
                                bg_hid_vio_south_rtree.query(bgi::intersects(det_south_left), std::back_inserter(south_left_list));
                                if(!south_left_list.empty()){
                                    for(auto& left_bg_rect : south_left_list){
                                        auto [left_rect, net_mental, len] = left_bg_rect;
                                        auto [left_net, left_mental] = net_mental;
                                        PlanarRect left_p_rect = DRCUTIL.convertToPlanarRect(left_rect);
                                        if(one_p_rect.get_ll_x() <= left_p_rect.get_ur_x() || one_net_idx == left_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if (one_south_right >= min_length) {
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_right_list;
                                bg_hid_vio_south_rtree.query(bgi::intersects(det_south_right), std::back_inserter(south_right_list));
                                if(!south_right_list.empty()){
                                    for(auto& right_bg_rect : south_right_list){
                                        auto [right_rect, net_mental, len] = right_bg_rect;
                                        auto [right_net, right_mental] = net_mental;
                                        PlanarRect right_p_rect = DRCUTIL.convertToPlanarRect(right_rect);
                                        if(one_p_rect.get_ur_x() >= right_p_rect.get_ll_x() || one_net_idx == right_net){
                                            continue;
                                        }
                                        is_check_prl = true;
                                    }
                                }
                            }
                            if(!is_check_prl){
                                continue;
                            }
                            BGRectInt detect_south(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() - eol_space + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ll_y() - 1) 
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_list;
                            bg_hid_vio_south_rtree.query(bgi::intersects(detect_south), std::back_inserter(south_list));
                            if(south_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            bool have_vio = false;
                            std::vector<std::pair<PlanarRect, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : south_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                
                                PlanarRect vio_rect = gen_vio_rect_down_up(other_p_rect, one_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_south;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_south;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_south));
                                bg_hid_vio_south_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_south));
                                if(bg_result_south.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_south) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                violation_list.push_back({vio_rect, other_net_idx});
                            }
                            for (auto i = violation_list.begin(); i != violation_list.end();) {
                                auto [rect1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, net2] = *j;
                                        if (rect1.getYSpan() <= rect2.getYSpan()) {
                                            // 保留 i（rs 较小或相等），删除 j
                                            j = violation_list.erase(j);
                                        } else {
                                            // 保留 j（rs 较小），删除 i
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; // 退出内层循环，重新开始外层循环
                                        }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                                have_vio =true;
                            }
                            if(!have_vio){
                                continue;
                                //如果该区域没有平行,就继续判断下一个规则。
                            }
                            break;
                        }else {
                            int32_t max_spacing = eol_space;
                            if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                max_spacing = std::max(eol_space, ete_space);
                            }
                            BGRectInt detect_south(
                                BGPointInt(one_p_rect.get_ll_x() - eol_within + 1, one_p_rect.get_ll_y() - max_spacing + 1),
                                BGPointInt(one_p_rect.get_ur_x() + eol_within - 1, one_p_rect.get_ll_y() - 1) 
                            );
                            std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> south_list;
                            bg_hid_vio_south_rtree.query(bgi::intersects(detect_south), std::back_inserter(south_list));
                            if(south_list.empty()){
                                continue;
                                //双侧查不到，下一级规则还有可能查到，继续查下一个规则。
                            }
                            std::vector<std::tuple<PlanarRect, int32_t, int32_t>> violation_list;
                            for(auto& other_rect_net_mental : south_list){
                                auto [other_bg_rect, other_net_mental, other_len] = other_rect_net_mental;
                                auto [other_net_idx, other_mental_idx] = other_net_mental;
                                PlanarRect other_p_rect = DRCUTIL.convertToPlanarRect(other_bg_rect);
                                if (other_p_rect.getWidth() < min_width) {
                                    continue;
                                }
                                if(DRCUTIL.isClosedOverlap(one_p_rect, other_p_rect)){
                                    continue;
                                }
                                
                                PlanarRect vio_rect = gen_vio_rect_down_up(other_p_rect, one_p_rect);
                                if(vio_rect == invalid_rect){
                                    continue;
                                }
                                BGRectInt check_vio_rect = DRCUTIL.convertToBGRectInt(vio_rect);
                                std::vector<BGRectInt> bg_result_south;
                                std::vector<std::tuple<BGRectInt, std::pair<int32_t, int32_t>, std::vector<int32_t>>> bg_all_south;
                                bg_routing_rect_result_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_result_south));
                                bg_hid_vio_south_rtree.query(bgi::intersects(check_vio_rect), std::back_inserter(bg_all_south));
                                if(bg_result_south.empty()){
                                    continue;
                                }
                                bool is_overlap = false;
                                for (auto& all_rect : bg_all_south) {
                                    auto [rect, net_mental, len] = all_rect;
                                    PlanarRect p_rect = DRCUTIL.convertToPlanarRect(rect);
                                    if (DRCUTIL.isOpenOverlap(p_rect, vio_rect)) {
                                        is_overlap = true;
                                        break;
                                    }
                                }
                                if (is_overlap) {
                                    continue; 
                                }
                                int32_t eol_rs_space = eol_space;
                                if(one_rule.is_ENDTOEND()){
                                //如果包含端到端关键字，单独处理。判断是否是端到端。
                                //是端到端就改变rs,去掉距离大于ete_space的违例。
                                    if (other_p_rect.getRectDirection() == Direction::kVertical && (other_len[4] != 0 || other_len[5] != 0)) {
                                        eol_rs_space = ete_space;
                                    }
                                }
                                if(vio_rect.getYSpan() >= eol_rs_space){
                                    continue;
                                }
                                violation_list.push_back(std::make_tuple(vio_rect, eol_rs_space, other_net_idx));
                            }
                            for(auto i = violation_list.begin(); i != violation_list.end();){
                                auto [rect1, rs1, net1] = *i;
                                bool erased = false;
                                for (auto j = std::next(i); j != violation_list.end();) {
                                    auto [rect2, rs2, net2] = *j;
                                    if (rect1.getYSpan() <= rect2.getYSpan()) {
                                        // 保留 i（rs 较小或相等），删除 j
                                        if (rect1.getYSpan() == rect2.getYSpan() && rs1 <= rs2){
                                            j = violation_list.erase(j);
                                        }
                                        else if (rect1.getYSpan() == rect2.getYSpan() && rs1 > rs2) {
                                            i = violation_list.erase(i);
                                            erased = true;
                                            break; 
                                        }else if (rect1.getYSpan() < rect2.getYSpan()) {
                                            j = violation_list.erase(j);
                                        }
                                    } else {
                                        // 保留 j（rs 较小），删除 i
                                        i = violation_list.erase(i);
                                        erased = true;
                                        break; // 退出内层循环，重新开始外层循环
                                    }
                                }
                                if (!erased) {
                                    ++i;
                                }
                            }
                            for(auto& rect_rs_net : violation_list){
                                auto [vio_rect, eol_rs_space, other_net_idx] = rect_rs_net;
                                Violation violation;
                                violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                                violation.set_is_routing(true);
                                violation.set_violation_net_set({one_net_idx, other_net_idx});
                                violation.set_required_size(eol_rs_space);
                                violation.set_layer_idx(routing_layer_idx);
                                violation.set_rect(vio_rect);
                                rv_box.get_violation_list().push_back(violation);
                            }
                        }
                    }
                }
            }
        }
    }
}



}  // namespace idrc
