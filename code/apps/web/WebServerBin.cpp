#include "utils/logging.h"
#include "utils/flags.h"
#include "web/WebServer.h"

DEFINE_int32(port, 9900, "websocket listening port");
DEFINE_int32(max_threads, 64, "max concurrent connections, each websocket holds one thread");

int main(int argc, char **argv) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(std::chrono::hours(24 * 30));  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  BASE_NAMESPACE::WebServer sv(FLAGS_port, FLAGS_max_threads);
  sv.start();
  return 0;
}
