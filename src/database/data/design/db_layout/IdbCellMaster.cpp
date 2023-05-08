/**
 * @project		iDB
 * @file		IdbCellMaster.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe macros information,.
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbCellMaster.h"

#include <algorithm>

#include "IdbObs.h"
#include "IdbTerm.h"

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbCellMaster::IdbCellMaster()
{
  _type = CellMasterType::kNone;
  _name = "";
  _symmetry_x = false;
  _symmetry_y = false;
  _symmetry_R90 = false;
  _origin_x = 0;
  _origin_y = 0;
  _width = -1;
  _height = -1;
  _core_filler = false;

  // _term_num = 0;
}

IdbCellMaster::~IdbCellMaster()
{
  for (IdbTerm* term : _term_list) {
    if (term) {
      delete term;
      term = nullptr;
    }
  }

  // _term_num =0;

  for (IdbObs* obs : _obs_list) {
    if (obs) {
      delete obs;
      obs = nullptr;
    }
  }
}

IdbLayer* IdbCellMaster::get_top_layer()
{
  IdbLayer* layer = nullptr;
  for (IdbTerm* term : _term_list) {
    IdbLayer* top_layer = term->get_top_layer();
    if (layer == nullptr) {
      layer = top_layer;
    } else {
      if (layer->get_order() < top_layer->get_order()) {
        layer = top_layer;
      }
    }
  }

  // for (auto obs : _obs_list) {
  // }

  return layer;
}

bool IdbCellMaster::is_ring()
{
  if (_type == CellMasterType::kRing)
    return true;
  else
    return false;
}

bool IdbCellMaster::is_cover()
{
  if (_type >= CellMasterType::kCover && _type <= CellMasterType::kCoverBump)
    return true;
  else
    return false;
}

bool IdbCellMaster::is_block()
{
  if (_type >= CellMasterType::kBlock && _type <= CellMasterType::kBLockSoft)
    return true;
  else
    return false;
}

bool IdbCellMaster::is_core()
{
  if (_type >= CellMasterType::kCore && _type <= CellMasterType::kEndcapBottomRight)
    return true;
  else
    return false;
}

bool IdbCellMaster::is_pad()
{
  if (_type >= CellMasterType::kPad && _type <= CellMasterType::kPadAreaIO)
    return true;
  else
    return false;
}

bool IdbCellMaster::is_endcap()
{
  if (_type >= CellMasterType::kEndcap && _type <= CellMasterType::kEndcapBottomRight)
    return true;
  else
    return false;
}

bool IdbCellMaster::is_core_filler()
{
  return _type == CellMasterType::kCoreSpacer || _core_filler == true ? true : false;
}

bool IdbCellMaster::set_type_core_filler()
{
  if (_type == CellMasterType::kCoreSpacer) {
    _core_filler = true;
    return true;
  } else if (_type < CellMasterType::kCore || _type > CellMasterType::kCoreWelltap) {
    _core_filler = false;
    return false;
  } else {
    /// A filler can only have Power and Ground pins
    for (IdbTerm* term : _term_list) {
      if (term->get_type() != IdbConnectType::kPower && term->get_type() != IdbConnectType::kGround) {
        _core_filler = false;
        return false;
      }
    }

    _core_filler = true;
    return true;
  }
}

bool IdbCellMaster::is_pad_filler()
{
  return _type == CellMasterType::kPadSpacer ? true : false;
}

bool IdbCellMaster::is_logic()
{
  for (auto term : _term_list) {
    if (!term->is_pdn()) {
      return true;
    }
  }

  return false;
}

void IdbCellMaster::set_type(string type_name)
{
  CellMasterType type = IdbEnum::GetInstance()->get_cell_property()->get_type(type_name);

  set_type(type);
}

IdbTerm* IdbCellMaster::add_term(IdbTerm* term)
{
  if (term == nullptr) {
    term = new IdbTerm();
    term->set_cell_master(this);
  }
  _term_list.emplace_back(term);
  //_term_num++;

  return term;
}

IdbTerm* IdbCellMaster::add_term(string name)
{
  IdbTerm* term = new IdbTerm();

  term->set_name(name);
  term->set_cell_master(this);
  _term_list.emplace_back(term);
  //  _term_num++;

  return term;
}
IdbObs* IdbCellMaster::add_obs(IdbObs* obs)
{
  if (obs == nullptr) {
    obs = new IdbObs();
  }
  _obs_list.emplace_back(obs);

  return obs;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbCellMasterList::IdbCellMasterList()
{
  _number = 0;
}

IdbCellMasterList::~IdbCellMasterList()
{
}

void IdbCellMasterList::reset_cell_master()
{
  for (auto& cell_master : _master_List) {
    if (cell_master != nullptr) {
      delete cell_master;
      cell_master = nullptr;
    }
  }

  _master_List.clear();
  _master_map.clear();

  _number = 0;
}

IdbCellMaster* IdbCellMasterList::set_cell_master(string name)
{
  IdbCellMaster* cell_master = find_cell_master(name);
  if (!cell_master) {
    cell_master = new IdbCellMaster();
    cell_master->set_name(name);
    _number++;
    _master_List.emplace_back(cell_master);
    _master_map[name] = cell_master;
  }

  return cell_master;
}

IdbCellMaster* IdbCellMasterList::find_cell_master(const string& src_name)
{
  auto cellMasterIt = _master_map.find(src_name);
  if (cellMasterIt != _master_map.end()) {
    return cellMasterIt->second;
  }
  return nullptr;
}
IdbCellMaster* IdbCellMasterList::find_cell_master(IdbCellMaster* src_master)
{
  return find_cell_master(src_master->get_name());
}

void IdbCellMasterList::initFillerList(vector<string> name_list)
{
  for (string name : name_list) {
    IdbCellMaster* cell_master = find_cell_master(name);
    if (cell_master != nullptr && cell_master->is_core_filler()) {
      cell_master->set_type_core_filler();
    } else {
      std::cout << "Error : %s is not a filler, please check it in lef." << std::endl;
    }
  }
}

void IdbCellMasterList::print()
{
  // for (IdbCellMaster* cell_master : _master_List) {
  //   cell_master->print();
  // }
}
}  // namespace idb
