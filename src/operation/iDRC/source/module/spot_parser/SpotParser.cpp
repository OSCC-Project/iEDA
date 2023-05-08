#include "SpotParser.h"

#include "CutSpacingCheck.hpp"
#include "DrcConfig.h"
#include "EOLSpacingCheck.hpp"
#include "EnclosedAreaCheck.h"
#include "EnclosureCheck.hpp"
#include "RegionQuery.h"
#include "RoutingAreaCheck.h"
#include "RoutingSpacingCheck.h"
#include "RoutingWidthCheck.h"
#include "Tech.h"
namespace idrc {

void SpotParser::init(DrcConfig* config, Tech* tech)
{
  _config = config;
  _tech = tech;
}

void SpotParser::reportEnd2EndSpacingViolation(EOLSpacingCheck* check)
{
  std::ofstream spot_file = get_spot_file("End2EndSpacingViolation");

  spot_file << "Total End To End spacing violation num ::" << check->get_e2e_violation_num() << std::endl;
  for (auto& [layerId, spot_list] : check->get_routing_layer_to_e2e_spacing_spots_list()) {
    spot_file << "Metal layer " << layerId << " End To End spacing violation num ::" << spot_list.size() << std::endl;
  }
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;

  int cnt = 0;

  for (auto& [layerId, spot_list] : check->get_routing_layer_to_e2e_spacing_spots_list()) {
    spot_file << "**********************************************" << std::endl;
    spot_file << "Report End To End Spacing violation on Routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    cnt = 0;

    for (auto& spot : spot_list) {
      ++cnt;
      if (cnt <= 50) {
        reportEnd2EndSpot(layerId, spot, spot_file);
      }
    }

    spot_file << "**********************************************" << std::endl;
    spot_file << "End End To End spacing violation report on routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
  }

  spot_file.close();
}

void SpotParser::reportEOLSpacingViolation(EOLSpacingCheck* check)
{
  std::ofstream spot_file = get_spot_file("EOLSpacingViolation");

  spot_file << "Total EOL spacing violation num ::" << check->get_eol_violation_num() << std::endl;
  for (auto& [layerId, spot_list] : check->get_routing_layer_to_eol_spacing_spots_list()) {
    spot_file << "Metal layer " << layerId << " EOL spacing violation num ::" << spot_list.size() << std::endl;
  }
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;

  int cnt = 0;

  for (auto& [layerId, spot_list] : check->get_routing_layer_to_eol_spacing_spots_list()) {
    spot_file << "**********************************************" << std::endl;
    spot_file << "Report EOL Spacing violation on Routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    cnt = 0;

    for (auto& spot : spot_list) {
      ++cnt;
      if (cnt <= 50) {
        reportEOLSpot(layerId, spot, spot_file);
      }
    }

    spot_file << "**********************************************" << std::endl;
    spot_file << "End EOL spacing violation report on routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
  }

  spot_file.close();
}

/**
 * @brief Output Enclosure violation information to a violation report file
 *
 * @param check
 */
void SpotParser::reportEnclosureViolation(EnclosureCheck* check)
{
  std::ofstream spot_file = get_spot_file("EnclosureViolation");

  spot_file << "Total Enclosure violation num ::" << check->get_enclosure_violation_num() << std::endl;
  for (auto& [layerId, spot_list] : check->get_cut_layer_to_enclosure_spots_list()) {
    spot_file << "cut layer " << layerId << " Enclosure violation num ::" << spot_list.size() << std::endl;
  }
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;

  int cnt = 0;

  for (auto& [layerId, spot_list] : check->get_cut_layer_to_enclosure_spots_list()) {
    spot_file << "**********************************************" << std::endl;
    spot_file << "Report Enclosure violation on Cut layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    cnt = 0;

    for (auto& spot : spot_list) {
      ++cnt;
      if (cnt <= 50) {
        reportSpotToFile(layerId, spot, spot_file);
      }
    }

    spot_file << "**********************************************" << std::endl;
    spot_file << "End Enclosure violation report on Cut layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
  }

  spot_file.close();
}

/**
 * @brief Output Cut spacing violation information to a violation report file
 *
 * @param check
 */
void SpotParser::reportCutSpacingViolation(CutSpacingCheck* check)
{
  std::ofstream spot_file = get_spot_file("CutSpacingViolation");

  spot_file << "Total spacing violation num ::" << check->get_spacing_violation_num() << std::endl;
  for (auto& [layerId, spot_list] : check->get_cut_layer_to_spacing_spots_list()) {
    spot_file << "Cut layer " << layerId << " Spacing violation num ::" << spot_list.size() << std::endl;
  }
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;

  int cnt = 0;

  for (auto& [layerId, spot_list] : check->get_cut_layer_to_spacing_spots_list()) {
    spot_file << "**********************************************" << std::endl;
    spot_file << "Report Spacing violation on Cut layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    cnt = 0;

    for (auto& spot : spot_list) {
      ++cnt;
      if (cnt <= 50) {
        reportSpotToFile(layerId, spot, spot_file);
      }
    }

    spot_file << "**********************************************" << std::endl;
    spot_file << "End Spacing violation report on Cut layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
  }

  spot_file.close();
}

/**
 * @brief 将间距违规信息输出到违规报告文件中（如果违规数目超过50个只报告前50个违规信息）
 *
 * @param check 间距违规检查模块指针
 */
void SpotParser::reportSpacingViolation(RoutingSpacingCheck* check)
{
  std::ofstream spot_file = get_spot_file("SpacingViolation");

  spot_file << "Total spacing violation num ::" << check->get_spacing_violation_num() << std::endl;
  for (auto& [layerId, spot_list] : check->get_routing_layer_to_spacing_spots_list()) {
    spot_file << "Metal layer " << layerId << " spacing violation num ::" << spot_list.size() << std::endl;
  }
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;

  int cnt = 0;

  for (auto& [layerId, spot_list] : check->get_routing_layer_to_spacing_spots_list()) {
    spot_file << "**********************************************" << std::endl;
    spot_file << "Report Spacing violation on Routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    cnt = 0;

    for (auto& spot : spot_list) {
      ++cnt;
      if (cnt <= 50) {
        reportSpotToFile(layerId, spot, spot_file);
      }
    }

    spot_file << "**********************************************" << std::endl;
    spot_file << "End spacing violation report on routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
  }

  spot_file.close();
}

/**
 * @brief 将短路违规信息输出到违规报告文件中（如果违规数目超过50个只报告前50个违规信息）
 *
 * @param check 间距违规检查模块指针
 */
void SpotParser::reportShortViolation(RoutingSpacingCheck* check)
{
  std::ofstream spot_file = get_spot_file("ShortViolation");

  spot_file << "Total short violation num ::" << check->get_short_violation_num() << std::endl;
  for (auto& [layerId, spot_list] : check->get_routing_layer_to_short_spots_list()) {
    spot_file << "Metal layer " << layerId << " short violation num ::" << spot_list.size() << std::endl;
  }
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;

  int cnt = 0;

  for (auto& [layerId, spot_list] : check->get_routing_layer_to_short_spots_list()) {
    spot_file << "**********************************************" << std::endl;
    spot_file << "Report short violation on Routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    cnt = 0;

    for (auto& spot : spot_list) {
      ++cnt;
      if (cnt <= 50) {
        reportSpotToFile(layerId, spot, spot_file);
      }
    }

    spot_file << "**********************************************" << std::endl;
    spot_file << "End short violation num on routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
  }

  spot_file.close();
}

/**
 * @brief 将宽度违规信息输出到违规报告文件中（如果违规数目超过50个只报告前50个违规信息）
 *
 * @param check 宽度违规检查模块指针
 */
void SpotParser::reportWidthViolation(RoutingWidthCheck* check)
{
  std::ofstream spot_file = get_spot_file("WidthViolation");

  spot_file << "Total width violation num ::" << check->get_width_violation_num() << std::endl;
  for (auto& [layerId, spot_list] : check->get_routing_layer_to_spots_map()) {
    spot_file << "Metal layer " << layerId << " width violation num ::" << spot_list.size() << std::endl;
  }
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;

  int cnt = 0;

  for (auto& [layerId, spot_list] : check->get_routing_layer_to_spots_map()) {
    spot_file << "**********************************************" << std::endl;
    spot_file << "Report width violation on Routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    cnt = 0;

    for (auto& spot : spot_list) {
      ++cnt;
      if (cnt <= 50) {
        reportSpotToFile(layerId, spot, spot_file);
      }
    }

    spot_file << "**********************************************" << std::endl;
    spot_file << "End width violation report on routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
  }

  spot_file.close();
}

/**
 * @brief 将面积违规信息输出到违规报告文件中（如果违规数目超过50个只报告前50个违规信息）
 *
 * @param check 面积违规检查模块指针
 */
void SpotParser::reportAreaViolation(RoutingAreaCheck* check)
{
  std::ofstream spot_file = get_spot_file("AreaViolation");

  spot_file << "Total area violation num ::" << check->get_area_violation_num() << std::endl;
  for (auto& [layerId, spot_list] : check->get_routing_layer_to_spots_map()) {
    spot_file << "Metal layer " << layerId << " area violation num ::" << spot_list.size() << std::endl;
  }
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;

  int cnt = 0;

  for (auto& [layerId, spot_list] : check->get_routing_layer_to_spots_map()) {
    spot_file << "**********************************************" << std::endl;
    spot_file << "Report area violation on Routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    cnt = 0;

    for (auto& spot : spot_list) {
      ++cnt;
      if (cnt <= 50) {
        reportSpotToFile(layerId, spot, spot_file);
      }
    }

    spot_file << "**********************************************" << std::endl;
    spot_file << "End area violation report on routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
  }

  spot_file.close();
}

/**
 * @brief 将孔洞面积违规信息输出到违规报告文件中（如果违规数目超过50个只报告前50个违规信息）
 *
 * @param check 包围面积违规检查模块指针
 */
void SpotParser::reportEnclosedAreaViolation(EnclosedAreaCheck* check)
{
  std::ofstream spot_file = get_spot_file("EnclosedAreaViolation");

  spot_file << "Total enclosed area violation num ::" << check->get_enclosed_area_violation_num() << std::endl;
  for (auto& [layerId, spot_list] : check->get_routing_layer_to_spots_map()) {
    spot_file << "Metal layer " << layerId << " enclosed area violation num ::" << spot_list.size() << std::endl;
  }
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;

  int cnt = 0;

  for (auto& [layerId, spot_list] : check->get_routing_layer_to_spots_map()) {
    spot_file << "**********************************************" << std::endl;
    spot_file << "Report enclosed area violation on Routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    cnt = 0;

    for (auto& spot : spot_list) {
      ++cnt;
      if (cnt <= 50) {
        reportSpotToFile(layerId, spot, spot_file);
      }
    }

    spot_file << "**********************************************" << std::endl;
    spot_file << "End enclosed area violation report on routing layer " << layerId << std::endl;
    spot_file << "**********************************************" << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
    spot_file << std::endl;
  }

  spot_file.close();
}

/**
 * @brief 通过输入文件名生成一个文件流
 *
 * @param file_name 文件名
 * @return std::ofstream 文件流
 */
std::ofstream SpotParser::get_spot_file(std::string file_name)
{
  std::ostringstream oss;
  oss << file_name << ".txt";
  std::string spot_file_name = oss.str();
  oss.str("");
  std::ofstream spot_file(spot_file_name);
  assert(spot_file.is_open());
  return spot_file;
}

/**
 * @brief 将对应金属层的Spot数据结构中所存储的违规信息输出到文件中
 *
 * @param layerId 金属层Id
 * @param spot 封装的违规数据
 * @param spot_file 文件流
 */
void SpotParser::reportSpotToFile(int layerId, DrcSpot& spot, std::ofstream& spot_file)
{
  spot_file << "-----------------------------------------" << std::endl;
  if (spot.get_violation_type() == ViolationType::kShort) {
    spot_file << "Short Violation" << std::endl;
  } else if (spot.get_violation_type() == ViolationType::kRoutingWidth) {
    spot_file << "Width Violation" << std::endl;
  } else if (spot.get_violation_type() == ViolationType::kRoutingSpacing) {
    spot_file << "Spacing Violation" << std::endl;
  } else if (spot.get_violation_type() == ViolationType::kRoutingArea) {
    spot_file << "Area Violation" << std::endl;
  } else if (spot.get_violation_type() == ViolationType::kEnclosedArea) {
    spot_file << "Enclosed Area Violation" << std::endl;
  } else if (spot.get_violation_type() == ViolationType::kCutSpacing) {
    spot_file << "Cut Spacing Violation" << std::endl;
  } else if (spot.get_violation_type() == ViolationType::kEnclosure) {
    spot_file << "Enclosure Violation" << std::endl;
  }
  std::vector<DrcRect*> spot_rect_list = spot.get_spot_drc_rect_list();
  for (auto rect : spot_rect_list) {
    spot_file << std::endl;
    reportRectToFile(layerId, rect, spot_file);
    spot_file << std::endl;
  }

  spot_file << "-------------------next------------------" << std::endl;
  spot_file << std::endl;
}

void SpotParser::reportEnd2EndSpacingSpot(int layerId, DrcSpot& spot, std::ofstream& spot_file)
{
  spot_file << "Metal layer ::" << layerId << std::endl;
  for (auto edge : spot.get_spot_drc_edge_list()) {
    auto bg_x = static_cast<double>(edge->get_begin_x());
    auto bg_y = static_cast<double>(edge->get_begin_y());
    auto ed_x = static_cast<double>(edge->get_end_x());
    auto ed_y = static_cast<double>(edge->get_end_y());

    spot_file << "Edge" << std::endl;
    spot_file << "BeginPoint ::"
              << "(" << bg_x / 1000 << "," << bg_y / 1000 << ")"
              << " "
              << "EndPoint ::"
              << "(" << ed_x / 1000 << "," << ed_y / 1000 << ")" << std::endl;
  }
}

/**
 * @brief 将对应金属层spot中记录的违规标记区域或违规矩形输出到目标文件中
 *
 * @param layerId 金属层Id
 * @param rect 违规矩形区域或违规矩形本身
 * @param spot_file 文件流
 */
void SpotParser::reportRectToFile(int layerId, DrcRect* rect, std::ofstream& spot_file)
{
  // spot_file << "DrcRect Pointer ::" << rect << std::endl;
  spot_file << "Metal layer ::" << layerId << std::endl;
  if (!rect->isSpotMark()) {
    spot_file << "Net Id ::" << rect->get_net_id() << std::endl;
  }
  auto left = static_cast<double>(rect->get_left());
  auto bottom = static_cast<double>(rect->get_bottom());
  auto right = static_cast<double>(rect->get_right());
  auto top = static_cast<double>(rect->get_top());
  if (rect->isSegmentRect()) {
    spot_file << "SegMent Rect" << std::endl;
  } else if (rect->isViaMetalRect()) {
    spot_file << "Via Metal Rect" << std::endl;
  } else if (rect->isPinRect()) {
    spot_file << "Pin Rect" << std::endl;
  } else if (rect->isBlockRect()) {
    spot_file << "Block Rect" << std::endl;
  } else {
    spot_file << "Violation Box" << std::endl;
  }
  spot_file << "Rect LeftBottom ::"
            << "(" << left / 1000 << "," << bottom / 1000 << ")"
            << " "
            << "Rect TopRight ::"
            << "(" << right / 1000 << "," << top / 1000 << ")" << std::endl;
}

void SpotParser::reportEnd2EndSpot(int layerId, DrcSpot spot, std::ofstream& spot_file)
{
  spot_file << "Metal layer ::" << layerId << std::endl;
  std::cout << spot.get_spot_drc_edge_list().size() << std::endl;
  for (auto edge : spot.get_spot_drc_edge_list()) {
    auto bg_x = static_cast<double>(edge->get_begin_x());
    auto bg_y = static_cast<double>(edge->get_begin_y());
    auto ed_x = static_cast<double>(edge->get_end_x());
    auto ed_y = static_cast<double>(edge->get_end_y());

    spot_file << "Edge" << std::endl;
    spot_file << "BeginPoint ::"
              << "(" << bg_x / 1000 << "," << bg_y / 1000 << ")"
              << " "
              << "EndPoint ::"
              << "(" << ed_x / 1000 << "," << ed_y / 1000 << ")" << std::endl;
  }
  spot_file << std::endl;

  spot_file << "-----------------next---------------------------------" << std::endl;
}

void SpotParser::reportEOLSpot(int layerId, DrcSpot spot, std::ofstream& spot_file)
{
  spot_file << "Metal layer ::" << layerId << std::endl;
  DrcRect* rect = spot.get_spot_eol_vio().second;
  DrcEdge* edge = spot.get_spot_eol_vio().first;

  auto bg_x = static_cast<double>(edge->get_begin_x());
  auto bg_y = static_cast<double>(edge->get_begin_y());
  auto ed_x = static_cast<double>(edge->get_end_x());
  auto ed_y = static_cast<double>(edge->get_end_y());

  auto left = static_cast<double>(rect->get_left());
  auto bottom = static_cast<double>(rect->get_bottom());
  auto right = static_cast<double>(rect->get_right());
  auto top = static_cast<double>(rect->get_top());

  if (rect->isSegmentRect()) {
    spot_file << "SegMent Rect" << std::endl;
  } else if (rect->isViaMetalRect()) {
    spot_file << "Via Metal Rect" << std::endl;
  } else if (rect->isPinRect()) {
    spot_file << "Pin Rect" << std::endl;
  } else if (rect->isBlockRect()) {
    spot_file << "Block Rect" << std::endl;
  } else {
    spot_file << "Violation Box" << std::endl;
  }
  spot_file << "Net Id ::" << rect->get_net_id() << std::endl;
  spot_file << "Rect LeftBottom ::"
            << "(" << left / 1000 << "," << bottom / 1000 << ")"
            << " "
            << "Rect TopRight ::"
            << "(" << right / 1000 << "," << top / 1000 << ")" << std::endl;
  spot_file << "Edge" << std::endl;
  spot_file << "BeginPoint ::"
            << "(" << bg_x / 1000 << "," << bg_y / 1000 << ")"
            << " "
            << "EndPoint ::"
            << "(" << ed_x / 1000 << "," << ed_y / 1000 << ")" << std::endl;
  spot_file << std::endl;

  spot_file << "-----------------next---------------------------------" << std::endl;
}

}  // namespace idrc