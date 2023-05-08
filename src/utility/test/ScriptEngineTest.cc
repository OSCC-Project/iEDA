
#include "ScriptEngine.hh"
#include "gmock/gmock.h"
#include "gtest/gtest-death-test.h"
#include "gtest/gtest.h"

using ieda::ScriptEngine;

namespace {

TEST(TclTest, ctor) {
  ScriptEngine* my_engine = ScriptEngine::getOrCreateInstance();
  ASSERT_TRUE(my_engine);
  ScriptEngine::destroyInstance();
}

}  // namespace
