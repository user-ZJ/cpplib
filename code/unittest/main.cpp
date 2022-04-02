#include "gtest/gtest.h"
#include "utils/logging.h"

int main(int argc, char **argv) {
  google::InitGoogleLogging("MYUNITTEST");
  google::SetLogDestination(google::GLOG_FATAL,
                            "log/log_fatal_");
  google::SetLogDestination(google::GLOG_ERROR,
                            "log/log_error_");
  google::SetLogDestination(google::GLOG_WARNING,
                            "log/log_warning_");
  google::SetLogDestination(google::GLOG_INFO, "log/log_info_");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}