#include "db/MongoDBWrapper.h"
#include <string>
#include "utils/logging.h"
#include "utils/flags.h"



using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]); 
  LOG(INFO) << "mongdb test";
  MongoDBWrapper mongo("mongodb://admin:admin@10.12.50.209:27017/admin");
  mongo.InsertRequest();
  mongo.QueryRequest();
  mongo.DeleteRequest();
  
  return 0;
}
