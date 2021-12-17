#include<iostream>
#include<string>
#include "utils/logging.h"
#include "utils/flags.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h" // for stringify JSON
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

using namespace rapidjson;

int main(int argc,char *argv[]){
    std::string str = "hello";
    Document d;  // 创建一个空的Document
    d.SetObject();   // 将Document设置为Object类型
    Document::AllocatorType &allocator = d.GetAllocator();
    d.AddMember("name",StringRef(str.c_str()),allocator);  // 设置字符串
    d.AddMember("id",Value().SetInt(88888),allocator);  // 设置Int
    d.AddMember("pi",Value().SetFloat(3.1416),allocator);  // 设置float
    d.AddMember("f",Value().SetBool(true),allocator);  // 设置Boolean
    d.AddMember("n",Value(),allocator);  // 设置null
    // 设置list
    Value a(kArrayType);
    for (int i = 1; i < 5; i++)
        a.PushBack(i, allocator); 
    d.AddMember("ta",a,allocator);
    // 设置object
    Value o(kObjectType);
    o.AddMember("a",1,allocator);
    o.AddMember("b",2,allocator);
    o.AddMember("c",3,allocator);
    d.AddMember("tr",o,allocator);
    // 设置object中包含list
    Value tp(kObjectType);
    Value tp1(kArrayType);
    tp1.PushBack("he",allocator).PushBack("llo",allocator);
    Value tp2(kArrayType);
    tp2.PushBack("w",allocator).PushBack("ord",allocator);
    tp.AddMember("hello",tp1,allocator).
        AddMember("word",tp2,allocator);
    d.AddMember("tp",tp,allocator);
 
    // 把 DOM 转换（stringify）成 JSON。
    StringBuffer buffer;
    PrettyWriter<StringBuffer> writer(buffer);   // 为 JSON 加入缩进与换行,使得输出可读性更强
    d.Accept(writer);
    std::cout << buffer.GetString() << std::endl;

    StringBuffer buffer1;
    Writer<StringBuffer> writer1(buffer1);
    d.Accept(writer1);
    std::cout << buffer1.GetString() << std::endl;

    return 0;
}