#include "CutEolSpacingCheck.hpp"

namespace idrc {

bool CutEolSpacingCheck::check(DrcRect* target_rect)
{
  _check_result = true;
  checkCutEolSpacing(target_rect);
  return _check_result;
}

void CutEolSpacingCheck::init(DrcConfig* config, Tech* tech, RegionQuery* region_query)
{
  _config = config;
  _tech = tech;
  _region_query = region_query;
}

void CutEolSpacingCheck::checkCutEolSpacing(DrcNet* target_net)
{
  for (auto& [layerId, cut_rect_list] : target_net->get_layer_to_cut_rects_map()) {
    for (auto target_rect : cut_rect_list) {
      checkCutEolSpacing(target_rect);
    }
  }
}

bool CutEolSpacingCheck::isSpanLengthMet(DrcRect* target_rect)
{
  //
  // int spanlength = _lef58_cut_eol_spacing->get_span_length();
  return true;
}

bool CutEolSpacingCheck::isEolWidthMet(DrcRect* target_rect)
{
  return true;
}

void CutEolSpacingCheck::queryInExtBoxes()
{
  if (_above_metal_rect_list.empty()) {
    return;
  }
  int layer_id = _above_metal_rect_list[0].first->get_layer_id();
  for (auto [query_box, corner_dir, edge_dir] : _ext_query_box_list) {
    std::vector<std::pair<RTreeSegment, DrcEdge*>> query_result;
    _region_query->queryEdgeInRoutingLayer(layer_id, query_box, query_result);
    for (auto [rtree_seg, drc_edge] : query_result) {
      //排除掉方向不对的和自身poly的；
      if (edge_dir == EdgeDirection::kEast || edge_dir == EdgeDirection::kWest) {
        if (drc_edge->isHorizontal() && drc_edge->get_owner_polygon() != _cut_above_poly) {
          switch (corner_dir) {
            case CornerDirEnum::kNE:
              _cut_edges_need_to_check.insert(EdgeDirection::kNorth);
              _cut_edges_need_to_check.insert(EdgeDirection::kEast);
              break;
            case CornerDirEnum::kNW:
              _cut_edges_need_to_check.insert(EdgeDirection::kNorth);
              _cut_edges_need_to_check.insert(EdgeDirection::kWest);
              break;
            case CornerDirEnum::kSE:
              _cut_edges_need_to_check.insert(EdgeDirection::kSouth);
              _cut_edges_need_to_check.insert(EdgeDirection::kEast);
              break;
            case CornerDirEnum::kSW:
              _cut_edges_need_to_check.insert(EdgeDirection::kSouth);
              _cut_edges_need_to_check.insert(EdgeDirection::kWest);
              break;
            default:
              std::cout << "[DRC CutEolSpacingCheck Warning]:corner dir is default, Skip check!" << std::endl;
          }
        }
      }
      if (edge_dir == EdgeDirection::kNorth || edge_dir == EdgeDirection::kSouth) {
        if (drc_edge->isVertical() && drc_edge->get_owner_polygon() != _cut_above_poly) {
          switch (corner_dir) {
            case CornerDirEnum::kNE:
              _cut_edges_need_to_check.insert(EdgeDirection::kNorth);
              _cut_edges_need_to_check.insert(EdgeDirection::kEast);
              break;
            case CornerDirEnum::kNW:
              _cut_edges_need_to_check.insert(EdgeDirection::kNorth);
              _cut_edges_need_to_check.insert(EdgeDirection::kWest);
              break;
            case CornerDirEnum::kSE:
              _cut_edges_need_to_check.insert(EdgeDirection::kSouth);
              _cut_edges_need_to_check.insert(EdgeDirection::kEast);
              break;
            case CornerDirEnum::kSW:
              _cut_edges_need_to_check.insert(EdgeDirection::kSouth);
              _cut_edges_need_to_check.insert(EdgeDirection::kWest);
              break;
            default:
              std::cout << "[DRC CutEolSpacingCheck Warning]:corner dir is default, Skip check!" << std::endl;
          }
        }
      }
    }
  }
}

void CutEolSpacingCheck::checkNeighborWireMet()
{
  getExtBoxes();
  queryInExtBoxes();
}

void CutEolSpacingCheck::getExtBoxes()
{
  int backward_ext = _lef58_cut_eol_spacing->get_backward_ext();
  int side_ext = _lef58_cut_eol_spacing->get_side_ext();
  for (auto [drc_rect, dir] : _above_metal_rect_list) {
    if (dir == EdgeDirection::kEast && _is_right_trigger_edge) {
      RTreeBox ext_query_up_box;
      bg::set<bg::min_corner, 0>(ext_query_up_box, drc_rect->get_right() - backward_ext);
      bg::set<bg::min_corner, 1>(ext_query_up_box, drc_rect->get_top());
      bg::set<bg::max_corner, 0>(ext_query_up_box, drc_rect->get_right());
      bg::set<bg::max_corner, 1>(ext_query_up_box, drc_rect->get_top() + side_ext);
      _ext_query_box_list.push_back(std::make_tuple(ext_query_up_box, CornerDirEnum::kSW, EdgeDirection::kEast));
      RTreeBox ext_query_down_box;
      bg::set<bg::min_corner, 0>(ext_query_down_box, drc_rect->get_right() - backward_ext);
      bg::set<bg::min_corner, 1>(ext_query_down_box, drc_rect->get_bottom() - side_ext);
      bg::set<bg::max_corner, 0>(ext_query_down_box, drc_rect->get_right());
      bg::set<bg::max_corner, 1>(ext_query_down_box, drc_rect->get_bottom());
      _ext_query_box_list.push_back(std::make_tuple(ext_query_down_box, CornerDirEnum::kNW, EdgeDirection::kEast));
    }
    if (dir == EdgeDirection::kWest && _is_left_trigger_edge) {
      RTreeBox ext_query_up_box;
      bg::set<bg::min_corner, 0>(ext_query_up_box, drc_rect->get_left());
      bg::set<bg::min_corner, 1>(ext_query_up_box, drc_rect->get_top());
      bg::set<bg::max_corner, 0>(ext_query_up_box, drc_rect->get_left() + backward_ext);
      bg::set<bg::max_corner, 1>(ext_query_up_box, drc_rect->get_top() + side_ext);
      _ext_query_box_list.push_back(std::make_tuple(ext_query_up_box, CornerDirEnum::kSE, EdgeDirection::kWest));
      RTreeBox ext_query_down_box;
      bg::set<bg::min_corner, 0>(ext_query_down_box, drc_rect->get_left());
      bg::set<bg::min_corner, 1>(ext_query_down_box, drc_rect->get_bottom() - side_ext);
      bg::set<bg::max_corner, 0>(ext_query_down_box, drc_rect->get_left() + backward_ext);
      bg::set<bg::max_corner, 1>(ext_query_down_box, drc_rect->get_bottom());
      _ext_query_box_list.push_back(std::make_tuple(ext_query_down_box, CornerDirEnum::kNE, EdgeDirection::kWest));
    }
    if (dir == EdgeDirection::kNorth && _is_top_trigger_edge) {
      RTreeBox ext_query_left_box;
      bg::set<bg::min_corner, 0>(ext_query_left_box, drc_rect->get_left() - side_ext);
      bg::set<bg::min_corner, 1>(ext_query_left_box, drc_rect->get_top() - backward_ext);
      bg::set<bg::max_corner, 0>(ext_query_left_box, drc_rect->get_left());
      bg::set<bg::max_corner, 1>(ext_query_left_box, drc_rect->get_top());
      _ext_query_box_list.push_back(std::make_tuple(ext_query_left_box, CornerDirEnum::kSE, EdgeDirection::kNorth));
      RTreeBox ext_query_right_box;
      bg::set<bg::min_corner, 0>(ext_query_right_box, drc_rect->get_right());
      bg::set<bg::min_corner, 1>(ext_query_right_box, drc_rect->get_top() - backward_ext);
      bg::set<bg::max_corner, 0>(ext_query_right_box, drc_rect->get_right() + side_ext);
      bg::set<bg::max_corner, 1>(ext_query_right_box, drc_rect->get_top());
      _ext_query_box_list.push_back(std::make_tuple(ext_query_right_box, CornerDirEnum::kSW, EdgeDirection::kNorth));
    }
    if (dir == EdgeDirection::kSouth && _is_bottom_trigger_edge) {
      RTreeBox ext_query_left_box;
      bg::set<bg::min_corner, 0>(ext_query_left_box, drc_rect->get_left() - side_ext);
      bg::set<bg::min_corner, 1>(ext_query_left_box, drc_rect->get_bottom());
      bg::set<bg::max_corner, 0>(ext_query_left_box, drc_rect->get_left());
      bg::set<bg::max_corner, 1>(ext_query_left_box, drc_rect->get_bottom() + backward_ext);
      _ext_query_box_list.push_back(std::make_tuple(ext_query_left_box, CornerDirEnum::kNE, EdgeDirection::kSouth));
      RTreeBox ext_query_right_box;
      bg::set<bg::min_corner, 0>(ext_query_right_box, drc_rect->get_right());
      bg::set<bg::min_corner, 1>(ext_query_right_box, drc_rect->get_bottom());
      bg::set<bg::max_corner, 0>(ext_query_right_box, drc_rect->get_right() + side_ext);
      bg::set<bg::max_corner, 1>(ext_query_right_box, drc_rect->get_bottom() + backward_ext);
      _ext_query_box_list.push_back(std::make_tuple(ext_query_right_box, CornerDirEnum::kNW, EdgeDirection::kSouth));
    }
  }
}

void CutEolSpacingCheck::refresh()
{
  _ext_query_box_list.clear();
  _above_metal_rect_list.clear();
  _cut_edges_need_to_check.clear();
  _is_left_trigger_edge = false;
  _is_right_trigger_edge = false;
  _is_top_trigger_edge = false;
  _is_bottom_trigger_edge = false;
  _cut_above_poly = nullptr;
}

void CutEolSpacingCheck::checkCutEolSpacing(DrcRect* target_rect)
{
  int layer_id = target_rect->get_layer_id();
  _lef58_cut_class_list = _tech->get_drc_cut_layer_list()[layer_id]->get_lef58_cut_class_list();
  _lef58_cut_eol_spacing = _tech->get_drc_cut_layer_list()[layer_id]->get_lef58_cut_eol_spacing();
  if (!_lef58_cut_eol_spacing) {
    return;
  } else {
    refresh();
    // 首先要得到cut之上的poly
    // enclosure是否满足overhang要求
    if (!isOverhangMet(target_rect)) {
      return;
    }
    //是否满足EOL+Ext
    if (isSpanLengthMet(target_rect) || isEolWidthMet(target_rect)) {
      //检查是否
      checkNeighborWireMet();

      if (!_cut_edges_need_to_check.empty()) {
        for (auto edge_dir : _cut_edges_need_to_check) {
          if (_lef58_cut_eol_spacing->get_prl() < 0) {
            checkSpacing2_PrlNeg(target_rect, edge_dir);
            // checkSpacing1_PrlNeg(target_rect, edge_dir);
          }
        }
      }
    }
  }
}

void CutEolSpacingCheck::getSpacing1QueryBox_PrlNeg(RTreeBox& query_box, DrcRect* drc_rect, EdgeDirection edge_dir)
{
  int spacing1 = _lef58_cut_eol_spacing->get_to_classes()[0].get_cut_spacing1();
  // int spacing1 = _lef58_cut_eol_spacing->get_cut_spacing1();

  if (edge_dir == EdgeDirection::kNorth) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() - spacing1);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_top() - spacing1);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() + spacing1);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() + spacing1);
  } else if (edge_dir == EdgeDirection::kSouth) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() - spacing1);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() - spacing1);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() + spacing1);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_bottom() + spacing1);
  } else if (edge_dir == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() - spacing1);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() - spacing1);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_left() + spacing1);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() + spacing1);
  } else if (edge_dir == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_right() - spacing1);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() - spacing1);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() + spacing1);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() + spacing1);
  }
}

void CutEolSpacingCheck::checkCornerSpacing(DrcRect* result_rect, DrcRect* target_rect, EdgeDirection edge_dir, int required_spacing)
{
  int distanceX = 0, distanceY = 0;
  if (edge_dir == EdgeDirection::kEast) {
    if (result_rect->get_bottom() > target_rect->get_top()) {
      distanceY = result_rect->get_bottom() - target_rect->get_top();
    } else if (result_rect->get_top() < target_rect->get_bottom()) {
      distanceY = target_rect->get_bottom() - result_rect->get_top();
    }
    if (target_rect->get_right() >= result_rect->get_left() && target_rect->get_right() <= result_rect->get_right()) {
      distanceX = 0;
    } else if (target_rect->get_right() < result_rect->get_left()) {
      distanceX = result_rect->get_left() - target_rect->get_right();
    } else {
      distanceX = target_rect->get_right() - result_rect->get_right();
    }
    if (required_spacing * required_spacing > distanceX * distanceX + distanceY * distanceY) {
      // add_spot();
      _check_result = false;
      // std::cout << "spacing1 vio" << std::endl;
      // _region_query->addViolation(ViolationType::kCutEOLSpacing);
      if (_region_query->addCutEOLSpacingViolation(target_rect, result_rect)) {
        addSpot(target_rect, result_rect);
      }

      return;
    }
  }
  if (edge_dir == EdgeDirection::kWest) {
    int target_rect_left = target_rect->get_left();
    if (result_rect->get_bottom() > target_rect->get_top()) {
      distanceY = result_rect->get_bottom() - target_rect->get_top();
    } else if (result_rect->get_top() < target_rect->get_bottom()) {
      distanceY = target_rect->get_bottom() - result_rect->get_top();
    }
    if (target_rect_left >= result_rect->get_left() && target_rect_left <= result_rect->get_right()) {
      distanceX = 0;
    } else if (target_rect_left < result_rect->get_left()) {
      distanceX = result_rect->get_left() - target_rect_left;
    } else {
      distanceX = target_rect_left - result_rect->get_right();
    }
    if (required_spacing * required_spacing > distanceX * distanceX + distanceY * distanceY) {
      // add_spot();
      _check_result = false;
      // std::cout << "spacing1 vio" << std::endl;
      // _region_query->addViolation(ViolationType::kCutEOLSpacing);
      if (_region_query->addCutEOLSpacingViolation(target_rect, result_rect)) {
        addSpot(target_rect, result_rect);
      }

      return;
    }
  }
  if (edge_dir == EdgeDirection::kNorth) {
    int target_rect_top = target_rect->get_top();
    if (result_rect->get_bottom() > target_rect->get_top()) {
      distanceY = result_rect->get_bottom() - target_rect->get_top();
    } else if (result_rect->get_top() < target_rect->get_bottom()) {
      distanceY = target_rect->get_bottom() - result_rect->get_top();
    }
    if (target_rect_top >= result_rect->get_left() && target_rect_top <= result_rect->get_right()) {
      distanceX = 0;
    } else if (target_rect_top < result_rect->get_left()) {
      distanceX = result_rect->get_left() - target_rect_top;
    } else {
      distanceX = target_rect_top - result_rect->get_right();
    }
    if (required_spacing * required_spacing > distanceX * distanceX + distanceY * distanceY) {
      // add_spot();
      _check_result = false;
      // std::cout << "spacing1 vio" << std::endl;
      // _region_query->addViolation(ViolationType::kCutEOLSpacing);
      if (_region_query->addCutEOLSpacingViolation(target_rect, result_rect)) {
        addSpot(target_rect, result_rect);
      }

      return;
    }
  }
  if (edge_dir == EdgeDirection::kSouth) {
    int target_rect_bottom = target_rect->get_bottom();
    if (result_rect->get_bottom() > target_rect->get_top()) {
      distanceY = result_rect->get_bottom() - target_rect->get_top();
    } else if (result_rect->get_top() < target_rect->get_bottom()) {
      distanceY = target_rect->get_bottom() - result_rect->get_top();
    }
    if (target_rect_bottom >= result_rect->get_left() && target_rect_bottom <= result_rect->get_right()) {
      distanceX = 0;
    } else if (target_rect_bottom < result_rect->get_left()) {
      distanceX = result_rect->get_left() - target_rect_bottom;
    } else {
      distanceX = target_rect_bottom - result_rect->get_right();
    }
    if (required_spacing * required_spacing > distanceX * distanceX + distanceY * distanceY) {
      // add_spot();
      _check_result = false;
      // std::cout << "spacing1 vio" << std::endl;
      // _region_query->addViolation(ViolationType::kCutEOLSpacing);
      if (_region_query->addCutEOLSpacingViolation(target_rect, result_rect)) {
        addSpot(target_rect, result_rect);
      }

      return;
    }
  }
}

std::string CutEolSpacingCheck::getCutClassName(DrcRect* cut_rect)
{
  int size = _lef58_cut_class_list.size();
  for (int index = 0; index < size; index++) {
    if (cut_rect->getWidth() == _lef58_cut_class_list[index]->get_via_width()
        && cut_rect->getLength() == _lef58_cut_class_list[index]->get_via_length()) {
      return _lef58_cut_class_list[index]->get_class_name();
    }
  }
  std::cout << "[DRC::CutEolSpacingCheck Warning] : Unkown Cut Class!" << std::endl;
  return std::string("");
}

int CutEolSpacingCheck::getRequiredSpacing1(DrcRect* result_rect, DrcRect* target_rect)
{
  // int layer_id = result_rect->get_layer_id();
  auto result_cut_class_name = getCutClassName(result_rect);
  auto target_cut_class_name = getCutClassName(target_rect);
  auto class1_name = _lef58_cut_eol_spacing->get_class_name1();
  if (target_cut_class_name.compare(class1_name) == 0) {
    auto to_class_list = _lef58_cut_eol_spacing->get_to_classes();
    for (auto to_class : to_class_list) {
      if (result_cut_class_name.compare(to_class.get_class_name()) == 0) {
        return to_class.get_cut_spacing1();
      }
    }
  }
  return _lef58_cut_eol_spacing->get_cut_spacing1();
}

bool CutEolSpacingCheck::isPrlOverlap(DrcRect* target_rect, DrcRect* result_rect, EdgeDirection edge_dir)
{
  if (edge_dir == EdgeDirection::kEast || edge_dir == EdgeDirection::kWest) {
    if (target_rect->get_bottom() > result_rect->get_top() || target_rect->get_top() < result_rect->get_bottom()) {
      return false;
    } else {
      return true;
    }
  }
  if (edge_dir == EdgeDirection::kNorth || edge_dir == EdgeDirection::kSouth) {
    if (target_rect->get_left() > result_rect->get_right() || target_rect->get_right() < result_rect->get_left()) {
      return false;
    } else {
      return true;
    }
  }
  return false;
}

void CutEolSpacingCheck::checkEdgeSpacing(DrcRect* target_rect, DrcRect* result_rect, EdgeDirection edge_dir, int required_spacing)
{
  if (edge_dir == EdgeDirection::kEast) {
    int spacing1 = std::abs(target_rect->get_right() - result_rect->get_left());
    int spacing2 = std::abs(target_rect->get_right() - result_rect->get_right());
    if (std::min(spacing1, spacing2) < required_spacing) {
      _check_result = false;
      // std::cout << "spacing1 vio" << std::endl;
      // _region_query->addViolation(ViolationType::kCutEOLSpacing);
      if (_region_query->addCutEOLSpacingViolation(target_rect, result_rect)) {
        addSpot(target_rect, result_rect);
      }

      return;
    }
  }
  if (edge_dir == EdgeDirection::kWest) {
    int spacing1 = std::abs(target_rect->get_left() - result_rect->get_left());
    int spacing2 = std::abs(target_rect->get_left() - result_rect->get_right());
    if (std::min(spacing1, spacing2) < required_spacing) {
      _check_result = false;
      // std::cout << "spacing1 vio" << std::endl;
      // _region_query->addViolation(ViolationType::kCutEOLSpacing);
      if (_region_query->addCutEOLSpacingViolation(target_rect, result_rect)) {
        addSpot(target_rect, result_rect);
      }

      return;
    }
  }
  if (edge_dir == EdgeDirection::kNorth) {
    int spacing1 = std::abs(target_rect->get_top() - result_rect->get_left());
    int spacing2 = std::abs(target_rect->get_top() - result_rect->get_right());
    if (std::min(spacing1, spacing2) < required_spacing) {
      _check_result = false;
      // std::cout << "spacing1 vio" << std::endl;
      // _region_query->addViolation(ViolationType::kCutEOLSpacing);
      if (_region_query->addCutEOLSpacingViolation(target_rect, result_rect)) {
        addSpot(target_rect, result_rect);
      }

      return;
    }
  }
  if (edge_dir == EdgeDirection::kSouth) {
    int spacing1 = std::abs(target_rect->get_bottom() - result_rect->get_left());
    int spacing2 = std::abs(target_rect->get_bottom() - result_rect->get_right());
    if (std::min(spacing1, spacing2) < required_spacing) {
      _check_result = false;
      // std::cout << "spacing1 vio" << std::endl;
      // _region_query->addViolation(ViolationType::kCutEOLSpacing);
      if (_region_query->addCutEOLSpacingViolation(target_rect, result_rect)) {
        addSpot(target_rect, result_rect);
      }

      return;
    }
  }
}

void CutEolSpacingCheck::addSpot(DrcRect* target_rect, DrcRect* result_rect)
{
  auto box = DRCUtil::getSpanBoxBetweenTwoRects(target_rect, result_rect);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = target_rect->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getCutLayerNameById(layer_id));
  spot->set_net_id(target_rect->get_net_id());
  spot->set_vio_type(ViolationType::kCutEOLSpacing);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_cut_eol_spacing_spot_list.emplace_back(spot);
}

void CutEolSpacingCheck::checkSpacing1_TwoRect_PrlNeg(DrcRect* target_rect, DrcRect* result_rect, EdgeDirection edge_dir)
{
  int required_spacing = getRequiredSpacing1(result_rect, target_rect);
  if (isPrlOverlap(result_rect, target_rect, edge_dir)) {
    checkEdgeSpacing(result_rect, target_rect, edge_dir, required_spacing);
  } else {
    checkCornerSpacing(result_rect, target_rect, edge_dir, required_spacing);
  }
}

void CutEolSpacingCheck::checkSpacing1_PrlNeg(DrcRect* target_rect, EdgeDirection edge_dir)
{
  RTreeBox query_box;
  getSpacing1QueryBox_PrlNeg(query_box, target_rect, edge_dir);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  _region_query->queryInCutLayer(target_rect->get_layer_id(), query_box, query_result);
  for (auto& [boost_rect, cut_rect] : query_result) {
    checkSpacing1_TwoRect_PrlNeg(cut_rect, target_rect, edge_dir);
  }
}

void CutEolSpacingCheck::getSpacing2QueryBox_PrlNeg(RTreeBox& query_box, DrcRect* drc_rect, EdgeDirection edge_dir)
{
  int spacing2 = _lef58_cut_eol_spacing->get_cut_spacing2();
  int prl = _lef58_cut_eol_spacing->get_prl();

  if (edge_dir == EdgeDirection::kNorth) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() + prl);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_top());
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() - prl);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() + spacing2);
  } else if (edge_dir == EdgeDirection::kSouth) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() + prl);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() - spacing2);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() - prl);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_bottom());
  } else if (edge_dir == EdgeDirection::kWest) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_left() - spacing2);
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() + prl);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_left());
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() - prl);
  } else if (edge_dir == EdgeDirection::kEast) {
    bg::set<bg::min_corner, 0>(query_box, drc_rect->get_right());
    bg::set<bg::min_corner, 1>(query_box, drc_rect->get_bottom() + prl);
    bg::set<bg::max_corner, 0>(query_box, drc_rect->get_right() + spacing2);
    bg::set<bg::max_corner, 1>(query_box, drc_rect->get_top() - prl);
  }
}

void CutEolSpacingCheck::checkSpacing2_PrlNeg(DrcRect* target_rect, EdgeDirection edge_dir)
{
  RTreeBox query_box;
  getSpacing2QueryBox_PrlNeg(query_box, target_rect, edge_dir);
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;

  _region_query->queryInCutLayer(target_rect->get_layer_id(), query_box, query_result);
  if (!query_result.empty()) {
    _check_result = false;
    // _region_query->addViolation(ViolationType::kCutEOLSpacing);
    for (auto& [rt_box, result_rect] : query_result) {
      if (_region_query->addCutEOLSpacingViolation(target_rect, result_rect)) {
        addSpot(target_rect, result_rect);
      }
    }
    // std::cout << "spacing2 vio!!!" << std::endl;
  }
}

bool CutEolSpacingCheck::checkEnclosure(std::vector<std::pair<RTreeBox, DrcRect*>>& above_metal_rect_list, DrcRect* target_cut_rect)
{
  int top_overhang = 0, bottom_overhang = 0, left_overhang = 0, right_overhang = 0;
  int cut_left = target_cut_rect->get_left();
  int cut_right = target_cut_rect->get_right();
  int cut_top = target_cut_rect->get_top();
  int cut_bottom = target_cut_rect->get_bottom();
  for (auto [rtree_box, above_metal_rect] : above_metal_rect_list) {
    int metal_left = above_metal_rect->get_left();
    int metal_right = above_metal_rect->get_right();
    int metal_bottom = above_metal_rect->get_bottom();
    int metal_top = above_metal_rect->get_top();

    if (cut_left - metal_left > left_overhang) {
      left_overhang = cut_left - metal_left;
      _above_metal_rect_list.push_back(std::make_pair(above_metal_rect, EdgeDirection::kWest));
    }
    if (metal_right - cut_right > right_overhang) {
      right_overhang = metal_right - cut_right;
      _above_metal_rect_list.push_back(std::make_pair(above_metal_rect, EdgeDirection::kEast));
    }
    if (cut_bottom - metal_bottom > bottom_overhang) {
      bottom_overhang = cut_bottom - metal_bottom;
      _above_metal_rect_list.push_back(std::make_pair(above_metal_rect, EdgeDirection::kSouth));
    }
    if (metal_top - cut_top > top_overhang) {
      top_overhang = metal_top - cut_top;
      _above_metal_rect_list.push_back(std::make_pair(above_metal_rect, EdgeDirection::kNorth));
    }
  }
  int equal_overhang = _lef58_cut_eol_spacing->get_equal_overhang();
  int smaller_overhang = _lef58_cut_eol_spacing->get_smaller_overhang();
  if (left_overhang == equal_overhang && right_overhang == equal_overhang) {
    if (bottom_overhang < smaller_overhang) {
      _is_bottom_trigger_edge = true;
    }
    if (top_overhang < smaller_overhang) {
      _is_top_trigger_edge = true;
    }
    if (_is_bottom_trigger_edge || _is_top_trigger_edge) {
      return true;
    }
  }
  if (bottom_overhang == equal_overhang && top_overhang == equal_overhang) {
    if (left_overhang < smaller_overhang) {
      _is_left_trigger_edge = true;
    }
    if (right_overhang < smaller_overhang) {
      _is_right_trigger_edge = true;
    }
    if (_is_left_trigger_edge || _is_right_trigger_edge) {
      return true;
    }
  }
  return false;
}

bool CutEolSpacingCheck::isOverhangMet(DrcRect* target_cut_rect)
{
  std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
  getAboveMetalRectList(target_cut_rect, query_result);
  if (!query_result.empty()) {
    std::vector<std::pair<RTreeSegment, DrcEdge*>> edges_query_result;
    int layer_id = target_cut_rect->get_layer_id();
    _region_query->queryEdgeInRoutingLayer(layer_id, query_result[0].first, edges_query_result);
    if (!edges_query_result.empty()) {
      _cut_above_poly = edges_query_result[0].second->get_owner_polygon();
    } else {
      std::cout << "[DRC CutEolSpacingCheck Warning]: Get cut above poly failed!" << std::endl;
    }
  }
  // if (!above_metal_rect) {
  //   std::cout << "[DRC CutEolSpacingCheck Warning]:cut has no above metal" << std::endl;
  // }
  return checkEnclosure(query_result, target_cut_rect);
}

void CutEolSpacingCheck::getAboveMetalRectList(DrcRect* target_cut_rect, std::vector<std::pair<RTreeBox, DrcRect*>>& query_result)
{
  int layer_id = target_cut_rect->get_layer_id();
  // cut上层金属的layer_id与其自身layer_id相同
  _region_query->queryEnclosureInRoutingLayer(layer_id, DRCUtil::getRTreeBox(target_cut_rect), query_result);
}

}  // namespace idrc
