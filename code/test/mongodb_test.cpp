#include "db/MongoDBWrapper.h"
#include <string>
#include "utils/logging.h"
#include "utils/flags.h"

using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]); 
  LOG(INFO) << "mongdb test";
  MongoDBWrapper mongo;
  mongo.InsertRequest();
  mongo.QueryRequest();
  mongo.DeleteRequest();
  
  return 0;
}
