#include "cryptlib.h"
#include "rijndael.h"
#include "modes.h"
#include "files.h"
#include "osrng.h"
#include "hex.h"
#include "utils/base64-util.h"

#include <iostream>
#include <string>

// 参考：https://cryptopp.com/wiki/Advanced_Encryption_Standard

using namespace BASE_NAMESPACE;
using namespace std;

int main(int argc, char* argv[])
{
    using namespace CryptoPP;

    std::string mkey = "38Wyyw/x08EOErI7Ru5nhw==";
    std::string miv = "irGysWnGg/MoWkCq+jYOYQ==";

    AutoSeededRandomPool prng;
    // HexEncoder encoder(new FileSink(std::cout));

    SecByteBlock key(AES::DEFAULT_KEYLENGTH);
    SecByteBlock iv(AES::BLOCKSIZE);

    // prng.GenerateBlock(key, key.size());
    // prng.GenerateBlock(iv, iv.size());
    memcpy(key.data(),base64_decode(mkey).c_str(),key.size());
    memcpy(iv.data(),base64_decode(miv).c_str(),iv.size());

    // for(int i=0;i<key.size();i++)
    //     std::cout<<(int)*(key.data()+i)<<std::endl;

    // std::string skey;
    // HexEncoder sencoder(new StringSink(skey));
    // sencoder.Put(key, key.size());
    // std::cout<<"skey"<<skey<<std::endl;

    std::string plain = "CBC Mode Test";
    std::string cipher, recovered;

    std::cout << "plain text: " << plain << std::endl;

    /*********************************\
    \*********************************/

    try
    {
        CBC_Mode< AES >::Encryption e;
        e.SetKeyWithIV(key, key.size(), iv);

        StringSource s(plain, true, 
            new StreamTransformationFilter(e,
                new StringSink(cipher)
            ) // StreamTransformationFilter
        ); // StringSource
    }
    catch(const Exception& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    /*********************************\
    \*********************************/

    std::cout << "key: "<<base64_encode(key, key.size())<<std::endl;
    // encoder.Put(key, key.size());
    // encoder.MessageEnd();
    std::cout << std::endl;

    std::cout << "iv: "<<base64_encode(iv, iv.size())<<std::endl;
    // encoder.Put(iv, iv.size());
    // encoder.MessageEnd();
    std::cout << std::endl;

    std::cout << "cipher text: "<<base64_encode((const byte*)&cipher[0], cipher.size());
    // encoder.Put((const byte*)&cipher[0], cipher.size());
    // encoder.MessageEnd();
    std::cout << std::endl;
    
    /*********************************\
    \*********************************/

    try
    {
        CBC_Mode< AES >::Decryption d;
        d.SetKeyWithIV(key, key.size(), iv);

        StringSource s(cipher, true, 
            new StreamTransformationFilter(d,
                new StringSink(recovered)
            ) // StreamTransformationFilter
        ); // StringSource

        std::cout << "recovered text: " << recovered << std::endl;
    }
    catch(const Exception& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}