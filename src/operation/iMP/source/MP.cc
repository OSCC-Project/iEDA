#include "MP.hh"

#include "Cluster.hh"
#include "DataManager.hh"
#include "HyperGraph.hh"
#include "SA.hh"
namespace imp {

MacroPlacer::MacroPlacer(DataManager* dm, Option* opt)
{
}

MacroPlacer::MacroPlacer(const std::string& idb_json, const std::string& opt_json)
    : MacroPlacer()
{
  setDataManager(idb_json);
}
MacroPlacer::MacroPlacer() : _dm(new DataManager())
{
}
void MacroPlacer::setDataManager(DataManager* dm)
{
  if (_dm != nullptr)
    delete _dm;
  _dm = dm;
}
MacroPlacer::~MacroPlacer()
{
  delete _dm;
}
void MacroPlacer::setDataManager(const std::string& idb_json)
{
  _dm->readFormLefDef(idb_json);
}
}  // namespace imp
