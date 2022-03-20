# rapidjson使用

文件对象模型（Document Object Model, DOM）

rapidjson数据类型分为`Value`和`Document`

`Value`存储json值

`Document`表示整个 DOM，它存储了一个 DOM 树的根 `Value`

```json
{
    "name": "hello",
    "id": 888888,
    "pi": 3.1416,
    "f": true,
    "n": null,
    "ta": [
        1,
        2,
        3,
        4
    ],
    "tr": {
        "a": 1,
        "b": 2,
        "c": 3
    },
    "tp": {
        "hello": [
            "he",
            "llo"
        ],
        "word": [
            "w",
            "ord"
        ]
    }
}
```

## 安装

https://github.com/Tencent/rapidjson/

RapidJSON 是只有头文件的 C++ 库。只需把 `include/rapidjson` 目录复制至系统或项目的 include 目录中。

## 写json

```cpp
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
```



## 读取json

```cpp
#include <iostream>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"

using namespace std;
using namespace rapidjson;

int main()
{
    string jsonfile="test.json";
    FILE* fp = fopen(jsonfile.c_str(), "rb");
    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    Document d;
    ParseResult ok = d.ParseStream(is);  // 如果已经是string，使用d.Parse(char *)解析
    if (!ok)
  		cout<<"JSON parse error: "<<GetParseError_En(ok.Code())<<" ("<<ok.Offset()<<")\n";
    static const char* kTypeNames[] =  { "Null", "False", "True", "Object", "Array", "String", "Number" };
    //读取string的value;使用d["name"].IsString()判断是否是string
    cout<<d["name"].GetString()<<endl;
    //读取int的value，同样可以使用GetUint()/GetInt64()/GetUint64()
    //使用IsNumber()/IsInt()/IsUint()/IsInt64()/IsUint64()判断是否是数字以及数字类型
    cout<<d["id"].GetInt()<<endl;
    //获取浮点数据，或使用GetDouble()；使用IsNumber()/IsFloat()/IsDouble判断是不是数字或浮点数据
    cout<<d["pi"].GetFloat()<<endl;
    //获取bool类型数据，使用IsBool()判断是否是bool类型数据
    string mybool = d["f"].GetBool() ? "true" : "false";
    cout<<d["f"].GetBool()<<" "<<mybool<<endl;
    //IsNull()判断是否是null
    string mynull = d["n"].IsNull() ? "null" : "?";
    cout<<mynull<<endl;
    /*************************Array************************************/
    //a.IsArray()判断是否是Array数据
    const Value& a = d["ta"];
    //使用下标访问Array
    for (SizeType i = 0; i < a.Size(); i++) // 使用 SizeType 而不是 size_t
        cout<<"a["<<i<<"] = "<<a[i].GetInt()<<endl;
    //使用迭代器访问
    for (Value::ConstValueIterator itr = a.Begin(); itr != a.End(); ++itr)
        cout<<itr->GetInt()<<endl;
    //使用c++11的形式访问Array
    for (auto& v : a.GetArray())
        cout<<v.GetInt()<<endl;
    /*************************Array************************************/
    /*************************Dict************************************/
    // 判断key是否在json的key中
    d.HasMember("hello");
    Value::ConstMemberIterator itr = d.FindMember("hello");
    if (itr != d.MemberEnd()){
        cout<<itr->value.GetString()<<endl;
    }
    //1. 使用迭代器方式访问dict，并判断Value的类型
    for (Value::ConstMemberIterator itr = d.MemberBegin();itr != d.MemberEnd(); ++itr)
	{
    	printf("Type of member %s is %s\n",itr->name.GetString(), kTypeNames[itr->value.GetType()]);
	}
    //2. 使用c++11的方式访问dict的value
    for(auto &m:d["tr"].GetObject()){
        cout<<m.name.GetString()<<" "<<m.value.GetInt()<<endl;
    }
    for(auto &m:d["tp"].GetObject()){\
        string key = m.name.GetString();
        cout<<key<<" ";
        for(auto &v:m.value.GetArray()){
            cout<<v.GetString()<<" ";
        }
        cout<<endl;
    }
    /*************************Dict************************************/
    return 0;
}
```

## 参考

http://rapidjson.org/zh-cn/md_doc_tutorial_8zh-cn.html#ValueDocument

