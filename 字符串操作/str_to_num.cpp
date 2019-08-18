#include <iostream>
#include <sstream>	//使用stringstream需要引入这个头文件
using namespace std;
 
//模板函数：将string类型变量转换为常用的数值类型（此方法具有普遍适用性）
template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;    
}
 
int main(int argc, char* argv[])
{
	string str("00801");
	cout << stringToNum<int>(str) << endl;
	cout << atoi(str.c_str()) << endl;
	//atoi: string to int
	//atol: string to long
	//atof: string to float
 
	system("pause");
	return 0;
}