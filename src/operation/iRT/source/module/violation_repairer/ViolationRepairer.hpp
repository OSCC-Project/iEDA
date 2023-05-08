#pragma once

#include "Config.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "VRDataManager.hpp"

namespace irt {

#define VR_INST (irt::ViolationRepairer::getInst())

class ViolationRepairer
{
 public:
  static void initInst(Config& config, Database& database);
  static ViolationRepairer& getInst();
  static void destroyInst();
  // function
  void repair(std::vector<Net>& net_list);

 private:
  // self
  static ViolationRepairer* _vr_instance;
  // config & database
  VRDataManager _vr_data_manager;

  ViolationRepairer(Config& config, Database& database) { init(config, database); }
  ViolationRepairer(const ViolationRepairer& other) = delete;
  ViolationRepairer(ViolationRepairer&& other) = delete;
  ~ViolationRepairer() = default;
  ViolationRepairer& operator=(const ViolationRepairer& other) = delete;
  ViolationRepairer& operator=(ViolationRepairer&& other) = delete;
  // function
  void init(Config& config, Database& database);
  void repairVRNetList(std::vector<VRNet>& vr_net_list);
  void buildVRResultTree(std::vector<VRNet>& vr_net_list);
  void buildKeyCoordPinMap(VRNet& vr_net);
  void buildCoordTree(VRNet& vr_net);
  void buildPHYNodeResult(VRNet& vr_net);
  void updateConnectionList(TNode<LayerCoord>* coord_node, VRNet& vr_net, std::vector<TNode<PHYNode>*>& pre_connection_list,
                            std::vector<TNode<PHYNode>*>& post_connection_list);
  TNode<PHYNode>* makeWirePHYNode(VRNet& vr_net, LayerCoord first_coord, LayerCoord second_coord);
  TNode<PHYNode>* makeViaPHYNode(VRNet& vr_net, irt_int below_layer_idx, PlanarCoord coord);
  TNode<PHYNode>* makePinPHYNode(VRNet& vr_net, irt_int pin_idx, LayerCoord coord);
  void repairVRResultTree(std::vector<VRNet>& vr_net_list);
  void updateOriginVRResultTree(std::vector<VRNet>& vr_net_list);
};

}  // namespace irt
