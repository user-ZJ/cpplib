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

## vector拷贝

```cpp
vector<int> input({ 1,2,3,4,5 });
cout<<input.size();
int arr[5];
std::copy(input.begin(), input.begin()+5, arr);
for (int i = 0; i < 5; i++) {
	cout<<arr[i]<<endl;
}

vector<int> input({ 1,2,3,4,5 });
vector<int> input2;
input2 = { input.begin(), input.begin() + 5 };
for (int i = 0; i < input2.size(); i++) {
	cout << input2[i];
}
```

## vector求和

```
int arr [] = {10,20,30,40,50}
vector<int> va(&arr[0],&arr[5])
int sum = accumulate(va.begin(),va.end(),0)
```





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

