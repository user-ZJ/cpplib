#ifndef BASE_AESCrypto_H_
#define BASE_AESCrypto_H_
#include <utility>
#include <string>

namespace BASE_NAMESPACE{

/**
 * @brief 生成aes的key和iv
 * 
 * @return <hexkey,hexiv> key
 */
std::pair<std::string,std::string> GenAESKey();
std::string AESCipherEncrypt(const std::string &msg, const std::string &hexkey,const std::string &hexiv);
std::string AESCipherDecrypt(const std::string &msg, const std::string &hexkey,const std::string &hexiv);

}

#endif