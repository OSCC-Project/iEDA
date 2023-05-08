/**
 * @file CmdCurrentDesign.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-07
 */

#include "Cmd.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

CmdCurrentDesign::CmdCurrentDesign(const char* cmd_name) : TclCmd(cmd_name) {}

unsigned CmdCurrentDesign::check() { return 1; }

/**
 * @brief execute the current_design cmd.
 *
 * @return unsigned
 */
unsigned CmdCurrentDesign::exec() {
  Sta* ista = Sta::getOrCreateSta();
  auto* nl = ista->get_netlist();
  SdcCollectionObj collection_obj(nl);

  auto* nl_collection = new SdcCollection(
      SdcCollection::CollectionType::kNetlist, {collection_obj});

  char* result = TclEncodeResult::encode(nl_collection);
  ScriptEngine::getOrCreateInstance()->setResult(result);

  return 1;
}

}  // namespace ista
