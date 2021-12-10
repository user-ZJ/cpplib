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

## 读取json

```cpp
#include <iostream>
#include "rapidjson/document.h"
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
    d.ParseStream(is);
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
    //判断key是否在json的key中
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



