
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <memory>
#include <random>
#include <sstream>

#include "AccessPoint.hpp"
#include "AccessPointType.hpp"
#include "EXTPlanarCoord.hpp"
#include "PlanarRect.hpp"
#include "WireNode.hpp"
#include "rt_serialize.hpp"
#include "serialize.hpp"

using iarchive = boost::archive::binary_iarchive;
using oarchive = boost::archive::binary_oarchive;

static std::random_device gRd;
static std::mt19937 gRnd(gRd());
// Generate a random number in the range of [0, max)
static int rnd(int max = 1000)
{
  return gRnd() % max;
}
static double drnd(double max = 1000.0)
{
  static std::uniform_real_distribution<double> dis(0.0, max);
  return dis(gRnd);
}

template <typename T, typename MemberFunc>
void check_helper(T& a, T& b, MemberFunc func)
{
  EXPECT_EQ((a.*func)(), (b.*func)());
}
template <typename T, typename MemberFunc>
void check_helper(T* a, T* b, MemberFunc func)
{
  EXPECT_EQ((a->*func)(), (b->*func)());
}

template <typename T, typename... MemberFuncs>
void check_members(T& a, T& b, MemberFuncs... funcs)
{
  (check_helper(a, b, funcs), ...);
}

// A test fixture class for serialization tests
class SerializeTest : public ::testing::Test
{
 protected:
  // Override the SetUp() method to perform initialization before each test case
  void SetUp() override { ss.clear(); };

  // Hold the serialized data
  std::stringstream ss{""};
  // Create an output archive (oarchive) that will serialize data to the stringstream
  oarchive oar{ss};
  iarchive iar{ss};
};

// Test the serialization and deserialization of PlanarCoord objects
TEST_F(SerializeTest, PlanarCoord)
{
  // Generate a random PlanarCoord object with random x and y coordinates
  irt::PlanarCoord origin(rnd(), rnd());
  oar << origin;

  irt::PlanarCoord deserialized;
  iar >> deserialized;

  EXPECT_EQ(origin, deserialized);
}

// Test case for PlanarRect
TEST_F(SerializeTest, PlanarRect)
{
  irt::PlanarRect origin(rnd(), rnd(), rnd(), rnd());
  oar << origin;
  irt::PlanarRect deserialized;
  iar >> deserialized;
  EXPECT_EQ(origin, deserialized);
}

// Test case for AccessPoint
TEST_F(SerializeTest, AccessPoint)
{
  irt::AccessPoint origin(rnd(), rnd(), rnd(), irt::AccessPointType::kOnShape);
  oar << origin;
  irt::AccessPoint deserialized;
  iar >> deserialized;
  EXPECT_EQ(origin.get_type(), deserialized.get_type());
  EXPECT_EQ(origin.get_grid_coord(), deserialized.get_grid_coord());
  EXPECT_EQ(origin.get_real_coord(), deserialized.get_real_coord());
}

TEST_F(SerializeTest, BoundingBox)
{
  irt::BoundingBox box;
  box.set_grid_rect(irt::PlanarRect(rnd(), rnd(), rnd(), rnd()));
  box.set_real_rect(irt::PlanarRect(rnd(), rnd(), rnd(), rnd()));

  oar << box;

  irt::BoundingBox deserialized;
  iar >> deserialized;

  EXPECT_EQ(box.get_grid_rect(), deserialized.get_grid_rect());
  EXPECT_EQ(box.get_real_rect(), deserialized.get_real_rect());

  auto str = [](irt::BoundingBox& box) -> std::string {
    std::stringstream ss;
    ss << "\ngrid:{" << box.get_grid_lb_x() << "," << box.get_grid_lb_y() << "," << box.get_grid_rt_x() << "," << box.get_grid_rt_y()
       << "}";
    ss << "\nreal:{" << box.get_real_lb_x() << "," << box.get_real_lb_y() << "," << box.get_real_rt_x() << "," << box.get_real_rt_y()
       << "}";
    return ss.str();
  };
  DLOG(INFO) << "Origin:      " << str(box);
  DLOG(INFO) << "Deserialized:" << str(deserialized);
}

TEST_F(SerializeTest, GridMap_double)
{
  int xsize = rnd();
  int ysize = rnd();
  irt::GridMap<double> origin(xsize, ysize);

  DLOG(INFO) << "generate random gridmap with x=" << xsize << " and y=" << ysize;

  for (int x = 0; x < xsize; ++x) {
    for (int y = 0; y < ysize; ++y) {
      origin[x][y] = drnd();
    }
  }
  oar << origin;

  irt::GridMap<double> deserialized;
  iar >> deserialized;

  EXPECT_EQ(xsize, deserialized.get_x_size());
  EXPECT_EQ(ysize, deserialized.get_y_size());
  for (int x = 0; x < xsize; ++x) {
    for (int y = 0; y < ysize; ++y) {
      EXPECT_DOUBLE_EQ(origin[x][y], deserialized[x][y]);
      // DLOG(INFO) << "checked " << origin[x][y] << "==" << deserialized[x][y];
    }
  }
}

TEST_F(SerializeTest, TNode)
{
  auto child2 = irt::TNode<int>(3);
  auto child1 = irt::TNode<int>(2);
  child1.addChild(&child2);
  auto root = irt::TNode<int>(1);
  root.addChild(&child1);

  oar << &root;

  irt::TNode<int>* ptr;
  iar >> ptr;
  int cnt = 0;
  auto dfs = [&cnt](auto& dfs, irt::TNode<int>* t1, irt::TNode<int>* t2) -> bool {
    if (t1 == nullptr || t2 == nullptr) {
      return t1 == nullptr && t2 == nullptr;
    }

    if (t1->getChildrenNum() != t2->getChildrenNum()) {
      return false;
    }
    for (int i = 0; i < t1->getChildrenNum(); ++i) {
      if (not dfs(dfs, t1->get_child_list()[i], t2->get_child_list()[i])) {
        return false;
      }
    }
    ++cnt;
    return true;
  };
  EXPECT_TRUE(dfs(dfs, ptr, &root));

  DLOG(INFO) << "Checked " << cnt << " nodes for TNode";

  auto free = [](auto& freet, irt::TNode<int>* root) -> void {
    for (irt::TNode<int>* next : root->get_child_list()) {
      freet(freet, next);
    }
    delete (root);
  };
  free(free, ptr);
}

TEST_F(SerializeTest, LayerRect)
{
  irt::LayerRect rect(rnd(), rnd(), rnd(), rnd(), rnd());
  oar << rect;

  irt::LayerRect deserialized;
  iar >> deserialized;
  DLOG(INFO) << "Generated Random LayerRect layer:" << rect.get_layer_idx();
  DLOG(INFO) << "Deserialized LayerRect layer:" << deserialized.get_layer_idx();
  EXPECT_EQ(rect.get_layer_idx(), deserialized.get_layer_idx());
  EXPECT_EQ(rect.get_lb(), deserialized.get_lb());
}

TEST_F(SerializeTest, Guide)
{
  irt::Guide guide(irt::PlanarRect{rnd(), rnd(), rnd(), rnd()}, rnd());

  oar << guide;

  irt::Guide deserialized;
  iar >> deserialized;

  EXPECT_EQ(guide.get_grid_coord(), deserialized.get_grid_coord());
  EXPECT_EQ(guide.get_layer_idx(), deserialized.get_layer_idx());
  EXPECT_EQ(guide.get_rect(), deserialized.get_rect());
}

TEST_F(SerializeTest, SegmentGuide)
{
  irt::Guide guide1(irt::PlanarRect{rnd(), rnd(), rnd(), rnd()}, rnd());
  irt::Guide guide2(irt::PlanarRect{rnd(), rnd(), rnd(), rnd()}, rnd());
  irt::Segment segment(guide1, guide2);

  oar << segment;

  irt::Segment<irt::Guide> deserialized;
  iar >> deserialized;

  EXPECT_EQ(segment.get_first(), deserialized.get_first());
  EXPECT_EQ(segment.get_second(), deserialized.get_second());
  DLOG(INFO) << "Random segment first layer idx is " << deserialized.get_first().get_layer_idx();
}

TEST_F(SerializeTest, LayerCoord)
{
  irt::LayerCoord coord(rnd(), rnd(), rnd());
  oar << coord;

  irt::LayerCoord deserialized;
  iar >> deserialized;

  EXPECT_EQ(coord.get_layer_idx(), deserialized.get_layer_idx());
  EXPECT_EQ(coord.get_planar_coord(), deserialized.get_planar_coord());
}

TEST_F(SerializeTest, RTNode)
{
  irt::RTNode rtnode;
  irt::Guide guide1(irt::PlanarRect{rnd(), rnd(), rnd(), rnd()}, rnd());
  irt::Guide guide2(irt::PlanarRect{rnd(), rnd(), rnd(), rnd()}, rnd());
  rtnode.set_first(guide1);
  rtnode.set_second(guide2);
  oar << rtnode;

  irt::RTNode deserialized;

  iar >> deserialized;

  EXPECT_EQ(rtnode.get_first(), deserialized.get_first());
  EXPECT_EQ(rtnode.get_second(), deserialized.get_second());

  // check if operator== is ok
  guide2.set_lb_x(guide1.get_lb_x() + 1);
  EXPECT_NE(guide1, guide2);
}

TEST_F(SerializeTest, PinNode)
{
  auto node = std::make_unique<irt::PinNode>();
  node->set_net_idx(rnd());
  node->set_pin_idx(rnd());
  node->set_layer_idx(rnd());
  node->set_coord(irt::PlanarCoord(rnd(), rnd()));

  oar << node.get();

  // irt::PinNode deserialized;
  irt::PinNode* p;
  // Deserialize to a pointer will allocate memory and replace the original pointer
  iar >> p;

#if 0
  EXPECT_EQ(node->get_net_idx(), p->get_net_idx());
  EXPECT_EQ(node->get_pin_idx(), p->get_pin_idx());
  EXPECT_EQ(node->get_layer_idx(), p->get_layer_idx());
  EXPECT_EQ(node->get_grid_x(), p->get_grid_x());
  EXPECT_EQ(node->get_grid_y(), p->get_grid_y());
  EXPECT_EQ(node->get_real_x(), p->get_real_x());
  EXPECT_EQ(node->get_real_y(), p->get_real_y());
#endif

  check_members(*node, *p, &irt::PinNode::get_net_idx,
                //
                &irt::PinNode::get_pin_idx,    //
                &irt::PinNode::get_layer_idx,  //
                &irt::PinNode::get_x,          //
                &irt::PinNode::get_y           //

  );
  delete p;
}

TEST_F(SerializeTest, WireNode)
{
  auto node = irt::WireNode();
  node.set_net_idx(rnd());
  node.set_layer_idx(rnd());
  node.set_wire_width(rnd());

  node.get_first().set_coord(irt::PlanarCoord{rnd(), rnd()});

  node.get_second().set_coord(irt::PlanarCoord{rnd(), rnd()});

  oar << node;
  irt::WireNode deserialized;
  iar >> deserialized;

  check_members(node, deserialized,             //
                &irt::WireNode::get_layer_idx,  //
                &irt::WireNode::get_net_idx,    //
                &irt::WireNode::get_wire_width  //
  );
  check_members(node.get_first(), deserialized.get_first(),  //
                &irt::PlanarCoord::get_x,                    //
                &irt::PlanarCoord::get_y                     //
  );
  check_members(node.get_second(), deserialized.get_second(),  //
                &irt::PlanarCoord::get_x,                      //
                &irt::PlanarCoord::get_y                       //
  );
}

TEST_F(SerializeTest, ViaNode)
{
  irt::ViaNode node;
  node.set_net_idx(rnd());
  node.set_via_idx({rnd(), rnd()});
  node.set_coord(irt::PlanarCoord{rnd(), rnd()});

  oar << node;

  irt::ViaNode deserialized;
  iar >> deserialized;
  check_members(node, deserialized,          //
                &irt::ViaNode::get_net_idx,  //
                &irt::ViaNode::get_via_idx   //
  );
  EXPECT_EQ(node.get_x(), deserialized.get_x());
  EXPECT_EQ(node.get_y(), deserialized.get_y());
}

TEST_F(SerializeTest, Net)
{
  irt::Net net;
  oar << net;

  irt::Net deserialized;
  iar >> deserialized;
}

TEST_F(SerializeTest, NetList)
{
  std::vector<irt::Net> net_list(10);
  net_list[0].set_net_idx(100);
  oar << net_list;

  std::vector<irt::Net> deserialized(10);
  iar >> deserialized;

  EXPECT_EQ(deserialized[0].get_net_idx(), 100);
}