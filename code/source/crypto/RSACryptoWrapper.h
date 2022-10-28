#ifndef BASE_RSACrypto_H_
#define BASE_RSACrypto_H_
#include <utility>
#include <string>

namespace BASE_NAMESPACE{


/**
 * @brief 生成RSA公钥和私钥
 * 
 * @return  <私钥，公钥> 对
 */
void GenRSAKeyFile();
/**
 * @brief 生成RSA公钥和私钥
 * 
 * @return  <私钥，公钥> 对
 */
std::pair<std::string,std::string> GenRSAKey();
/**
 * @brief 对msg进行签名
 * 
 * @param msg 待签名文本或二进制
 * @param pubKey 公钥
 * @param privKey 私钥
 * @return 签名 
 */
std::string SignSha256(const std::string &msg,const std::string &pubKey,const std::string &privKey);
/**
 * @brief 验证msg签名是否正确
 * 
 * @param msg 待签名文本或二进制
 * @param hexDig 签名
 * @param pubKey 公钥
 * @return true 签名正确
 * @return false 签名错误
 */
bool VerifySha256(const std::string &msg,const std::string &hexDig,const std::string &pubKey);
/**
 * @brief 对msg进行加密
 * 
 * @param msg 待加密文本或二进制
 * @param pubKey 公钥
 * @return 密文 
 */
std::string RSACipherEncrypt(const std::string &msg, const std::string &pubKey);
/**
 * @brief 对密文进行解密
 * 
 * @param msg 密文
 * @param privKey 私钥
 * @return 明文 
 */
std::string RSACipherDecrypt(const std::string &msg,const std::string &privKey);


}

#endif