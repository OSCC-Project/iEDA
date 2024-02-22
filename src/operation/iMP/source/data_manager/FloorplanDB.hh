#ifndef IMP_FLOORPLAN_DB
#define IMP_FLOORPLAN_DB

#include "Block.hh"
#include "ParserEngine.hh"
namespace imp {
class FloorplanDB
{
 public:
  FloorplanDB() = default;
  FloorplanDB(ParserEngine* parser) : _parser(parser) { _root = _parser->get_design_ptr(); }
  ~FloorplanDB() = default;
  Block& root() { return *_root; }
  const Block& root() const { return *_root; }
  std::shared_ptr<Block> root_ptr() { return _root; }

 private:
  std::shared_ptr<Block> _root;
  std::unique_ptr<ParserEngine> _parser;
};

}  // namespace imp
#endif