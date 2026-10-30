#include "gtest/gtest.h"
#include "utils/flags.h"
#include "utils/logging.h"

int main(int argc, char **argv) {
  FLAGS_log_dir = ".";
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_FATAL);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
