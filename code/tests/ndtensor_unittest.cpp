#include "utils/NDTensor.h"
#include "gtest/gtest.h"

using CUDA_NAMESPACE::DataType;
using CUDA_NAMESPACE::DeviceType;
using CUDA_NAMESPACE::GetElementSize;
using CUDA_NAMESPACE::NDTensor;

TEST(NDTensor, elementSize) {
  EXPECT_EQ(4, GetElementSize(DataType::FLOAT));
  EXPECT_EQ(8, GetElementSize(DataType::DOUBLE));
  EXPECT_EQ(1, GetElementSize(DataType::INT8));
  EXPECT_EQ(2, GetElementSize(DataType::INT16));
  EXPECT_EQ(4, GetElementSize(DataType::INT32));
  EXPECT_EQ(8, GetElementSize(DataType::INT64));
  EXPECT_EQ(2, GetElementSize(DataType::HALF));
}

TEST(NDTensor, shapeAndCopy) {
  NDTensor empty;
  EXPECT_EQ(0, empty.size());
  EXPECT_TRUE(empty.shapes().empty());

  NDTensor tensor({2, 3});
  EXPECT_EQ(6, tensor.size());
  EXPECT_EQ(24, tensor.byteSize());
  EXPECT_EQ(DataType::FLOAT, tensor.getDataType());
  EXPECT_EQ(DeviceType::CPU, tensor.getDeviceType());
  EXPECT_EQ((std::vector<int>{2, 3}), tensor.shapes());
  EXPECT_EQ((std::vector<int>{3, 1}), tensor.strides());
  ASSERT_NE(nullptr, tensor.data<float>());
  EXPECT_FLOAT_EQ(0.f, tensor.data<float>()[0]);

  tensor.data<float>()[0] = 1.5f;
  tensor.data<float>()[5] = -2.f;
  NDTensor copied = tensor;
  EXPECT_FLOAT_EQ(1.5f, copied.data<float>()[0]);
  EXPECT_FLOAT_EQ(-2.f, copied.data<float>()[5]);

  NDTensor assigned({1});
  assigned = tensor;
  EXPECT_EQ(6, assigned.size());
  EXPECT_FLOAT_EQ(1.5f, assigned.data<float>()[0]);

  tensor.resize({4});
  EXPECT_EQ(4, tensor.size());
  EXPECT_EQ(16, tensor.byteSize());
  EXPECT_EQ((std::vector<int>{4}), tensor.shapes());

  tensor.clear();
  EXPECT_EQ(0, tensor.size());
  EXPECT_EQ(0, tensor.byteSize());
  EXPECT_EQ(nullptr, tensor.data<float>());
}
