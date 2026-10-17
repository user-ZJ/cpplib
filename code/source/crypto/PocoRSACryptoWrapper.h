#ifndef BASE_RSACrypto_H_
#define BASE_RSACrypto_H_
#include <utility>
#include <string>

namespace BASE_NAMESPACE{

// 私钥均以 passphrase 加密保存；默认值仅用于测试，正式使用需传入自己的口令

/**
 * @brief 生成RSA公钥和私钥，保存到 key.pub / key.priv
 */
void GenRSAKeyFile(const std::string &passphrase = "test_password");
/**
 * @brief 生成RSA公钥和私钥
 * 
 * @return  <私钥，公钥> 对，失败时均为空
 */
std::pair<std::string,std::string> GenRSAKey(const std::string &passphrase = "test_password");
/**
 * @brief 对msg进行签名
 * 
 * @param msg 待签名文本或二进制
 * @param pubKey 公钥
 * @param privKey 私钥
 * @return 十六进制签名，失败时为空
 */
std::string SignSha256(const std::string &msg,const std::string &pubKey,const std::string &privKey,
                       const std::string &passphrase = "test_password");
/**
 * @brief 验证msg签名是否正确
 * 
 * @param msg 待签名文本或二进制
 * @param hexDig 签名
 * @param pubKey 公钥
 * @return true 签名正确
 * @return false 签名错误，或签名/公钥格式非法
 */
bool VerifySha256(const std::string &msg,const std::string &hexDig,const std::string &pubKey);
/**
 * @brief 对msg进行加密（RSA-OAEP）
 * 
 * @param msg 待加密文本或二进制
 * @param pubKey 公钥
 * @return 密文，失败时为空
 */
std::string RSACipherEncrypt(const std::string &msg, const std::string &pubKey);
/**
 * @brief 对密文进行解密（RSA-OAEP）
 * 
 * @param msg 密文
 * @param privKey 私钥
 * @return 明文，失败时为空
 */
std::string RSACipherDecrypt(const std::string &msg,const std::string &privKey,
                             const std::string &passphrase = "test_password");


}

#endif
