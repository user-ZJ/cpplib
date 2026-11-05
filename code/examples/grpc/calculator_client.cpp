#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include "calculator.grpc.pb.h"

// using grpc::Channel;
// using grpc::ClientContext;
// using grpc::Status;
// using calculator::Calculator;
// using calculator::AddRequest;
// using calculator::AddReply;

class CalculatorClient {
 public:
  CalculatorClient(std::shared_ptr<grpc::Channel> channel)
      : stub_(calculator::Calculator::NewStub(channel)) {}

  int Add(int a, int b) {
    calculator::AddRequest request;
    request.set_a(a);
    request.set_b(b);

    calculator::AddResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Add(&context, request, &response);

    if (status.ok()) {
      return response.result();
    } else {
      std::cout << "RPC failed: " << status.error_message() << std::endl;
      return -1;
    }
  }

  int Multiply(int a, int b) {
    calculator::MultiplyRequest request;
    request.set_a(a);
    request.set_b(b);

    calculator::MultiplyResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Multiply(&context, request, &response);

    if (status.ok()) {
      return response.result();
    } else {
      std::cout << "RPC failed: " << status.error_message() << std::endl;
      return -1;
    }
  }

 private:
  std::unique_ptr<calculator::Calculator::Stub> stub_;
};

int main(int argc, char** argv) {
  CalculatorClient calculator(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));

  int a = 10, b = 20;
  int result = calculator.Add(a, b);
  std::cout << a << " + " << b << " = " << result << std::endl;
  result = calculator.Multiply(a, b);
  std::cout << a << " * " << b << " = " << result << std::endl;

  return 0;
}
