#ifndef BASE_AESCrypto_H_
#define BASE_AESCrypto_H_
#include <utility>
#include <string>
#include <vector>

namespace BASE_NAMESPACE{

// AES-256-CBC + PKCS7 填充。hexkey 为 64 位十六进制，hexiv 为 32 位十六进制。
// 参数非法、密钥错误或数据损坏时返回空。

/**
 * @brief 生成aes的key和iv
 * 
 * @return <hexkey,hexiv>，随机数生成失败时均为空
 */
std::pair<std::string,std::string> GenAESKey();
std::string AESCipherEncrypt(const std::string &msg, const std::string &hexkey,const std::string &hexiv);
std::string AESCipherDecrypt(const std::string &msg, const std::string &hexkey,const std::string &hexiv);
// 以下使用内置密钥，用于模型文件加解密
std::string AESCipherEncrypt(const std::string &msg);
std::string AESCipherDecrypt(const std::string &msg);
std::vector<char> AESCipherDecrypt(const std::vector<char> &buff);

}

#endif
