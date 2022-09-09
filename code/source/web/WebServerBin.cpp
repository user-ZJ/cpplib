#include "utils/logging.h"
#include "utils/flags.h"
#include "web/WebServer.h"

DEFINE_int32(port, 9900, "websocket listening port");

int main(int argc, char **argv) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  BASE_NAMESPACE::WebServer sv(FLAGS_port);
  sv.start();
  return 0;
}
