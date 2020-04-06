## vector遍历方式
	//第一种遍历方式，下标
	cout << "第一种遍历方式，下标访问" << endl;
	for (int i = 0; i<m_testPoint.size(); i++ )
	{
		cout << m_testPoint[i].x << "	" << m_testPoint[i].y << endl;
	}
	
	//第二种遍历方式，迭代器
	cout << "第二种遍历方式，迭代器访问" << endl;
	for (vector<Point>::iterator iter = m_testPoint.begin(); iter != m_testPoint.end(); iter++)
	{
		cout << (*iter).x << "	" << (*iter).y << endl;
	}
	
	//第三种遍历方式，auto关键字
	cout << "C++11,第三种遍历方式，auto关键字" << endl;
	for (auto iter = m_testPoint.begin(); iter != m_testPoint.end(); iter++)
	{
		cout << (*iter).x << "	" << (*iter).y << endl;
	}
	 
	//第四种遍历方式，auto关键字的另一种方式
	cout << "C++11,第四种遍历方式，auto关键字" << endl;
	for (auto i : m_testPoint)
	{
		cout << i.x << "	" << i.y << endl;
	}

## c_str()

c_str()函数返回一个指向C字符串的指针常量，内容与string相同

是为了与C语言兼容，C语言中没有string类型

```cpp
char c[20];
string s="1234";
strcpy(c,s.c_str());
```

## split

```cpp
void split(std::string& s,std::string& deli,std::vector<std::string>& output){
    size_t pos=0;
    std::string token;
    while((pos=s.find(deli)) != std::string:npos){
        token = s.substr(0,pos);
        output.push_back(token);
        s.erase(0,pos+deli.length());
    }
    output.push_back(s);
}
```

