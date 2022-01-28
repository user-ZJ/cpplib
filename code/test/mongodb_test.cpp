#include "db/MongoDBWrapper.h"
#include <string>
#include "utils/logging.h"

using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]); 
  LOG(INFO) << "mongdb test";
  MongoDBWrapper mongo;
  mongo.InsertRequest();
  mongo.QueryRequest();
  mongo.DeleteRequest();
  
  return 0;
}