
#include "ONNXAcousticEngine.h"

#include "crypto/AESCryptoWrapper.h"
#include "utils/file-util.h"
#include "utils/logging.h"

#include <cstdint>
#include <map>
#include <numeric>
#include <cmath>

namespace BASE_NAMESPACE {

static int adaptive_avg_pool1d(const CTensorfl &input,const int &output_size,CTensorfl *out){
  // input NxLxC output Nxoutput_sizexC
  auto input_shape = input.shapes();
  int input_size = input_shape[1];
  out->resize({input_shape[0],output_size,input_shape[2]});
  auto start_index = [&input_size,&output_size](int curr_i){ return (int)(std::floor((curr_i *1.0* input_size) / output_size));};
  auto end_index = [&input_size,&output_size](int curr_i){ return (int)(std::ceil(((curr_i+1) *1.0* input_size) / output_size));};
  for(int j=0;j<input_shape[2];j++){
    float sum=0;
    int start=0,end=0;
    for(int i=0;i<output_size;i++){
      int window_start = start_index(i),window_end=end_index(i);
      while(start<window_start) sum-=input.at({0,start++,j});
      while(end<window_end) sum+=input.at({0,end++,j});
      out->at({0,i,j}) = sum/(window_end-window_start);
    }
  }
  return 0;
}

int ONNXAcousticEngine::loadModel(const std::string &encoderPath,
                              const std::string &decoderPath, bool is_crypto) {
  LOG(INFO) << "load model:" << encoderPath << ";" << decoderPath<<" is_crypto:"<<is_crypto;
  auto encoderbuff = file_to_buff(encoderPath.c_str());
  auto decoderbuff = file_to_buff(decoderPath.c_str());
  if (is_crypto) {
    encoderbuff = AESCipherDecrypt(encoderbuff);
    decoderbuff = AESCipherDecrypt(decoderbuff);
  }
  int res = encoder.loadModel(encoderbuff);
  if (res != 0) {
    LOG(ERROR) << "load model error";
    return res;
  }
  res = decoder_iter.loadModel(decoderbuff);
  if (res != 0) {
    LOG(ERROR) << "load model error";
    return res;
  }
  is_init_ = true;
  LOG(INFO) << "load model end";
  return res;
}

int ONNXAcousticEngine::infer(const CTensorfl &input,int bss_len,CTensorfl *out) {
  LOG(INFO) << "ONNXAcousticEngine::infer";
  try {
    if (!is_init_) {
      LOG(ERROR) << "infer error;please call loadModel() before";
      return -1;
    }
    LOG(INFO)<<"encoder infer";
    CTensorfl encoder_out;
    encoder.infer(input,&encoder_out);
    CTensorfl memories;
    adaptive_avg_pool1d(encoder_out,bss_len,&memories);
    LOG(INFO) << "Decoder infer";
    CTensorfl memory({1,memories.shapes()[2]});
    CTensorfl frame({1,52}),rnn_h0({1,1024}),rnn_h1({1,1024}),mel,rnn_out0,rnn_out1;
    out->resize({1,memories.shapes()[1],52});
    for(int i=0;i<memories.shapes()[1];i++){
      memcpy(memory.data(),&memories.at({0,i,0}),memory.byteSize());
      decoder_iter.infer(frame,memory,rnn_h0,rnn_h1,&mel,&rnn_out0,&rnn_out1);
      memcpy(&(out->at({0,i,0})),mel.data(),mel.byteSize());
      memcpy(frame.data(),mel.data(),frame.byteSize());
      memcpy(rnn_h0.data(),rnn_out0.data(),rnn_h0.byteSize());
      memcpy(rnn_h1.data(),rnn_out1.data(),rnn_h1.byteSize());
    }
    LOG(INFO) << "ONNXAcousticEngine::infer end";
    return 0;
  } catch (Ort::Exception e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}



}; // namespace BASE_NAMESPACE