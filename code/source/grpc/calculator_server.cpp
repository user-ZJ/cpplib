#include <grpcpp/grpcpp.h>
#include "calculator.pb.h"
#include "calculator.grpc.pb.h"


class CalculatorServiceImpl final : public calculator::Calculator::Service {
  grpc::Status Add(grpc::ServerContext* context, const calculator::AddRequest* request, calculator::AddResponse* response) override {
    response->set_result(request->a() + request->b());
    return grpc::Status::OK;
  }

  grpc::Status Multiply(grpc::ServerContext* context, const calculator::MultiplyRequest* request, calculator::MultiplyResponse* response) override {
    response->set_result(request->a() * request->b());
    return grpc::Status::OK;
  }
};

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  CalculatorServiceImpl service;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  grpc::EnableDefaultHealthCheckService(true);
  // 线程池
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS,4);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  server->Wait();
}

int main(int argc, char** argv) {
  RunServer();
  return 0;
}
