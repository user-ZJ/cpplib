#include <iostream>
#include <array>
using namespace std;

template <typename T> 
class Array {
private:
	T* ptr;
	int size;

public:
    using value_type=T;
    using iterator=value_type*;
	using reference=value_type&;

    Array(){}
	Array(T arr[], int s);

	void print();
    virtual void show();


};

template <typename T> 
Array<T>::Array(T arr[], int s)
{
	ptr = new T[s];
	size = s;
	for (int i = 0; i < size; i++)
		ptr[i] = arr[i];
}

template <typename T> 
void Array<T>::print()
{
	for (int i = 0; i < size; i++)
		cout << " " << *(ptr + i);
	cout << endl;
}

template <typename T> 
void Array<T>::show()
{
	for (int i = 0; i < size; i++)
		cout << " " << *(ptr + i);
	cout << endl;
}


template class Array<int>; //显式实例化模板


int main()
{
	int arr[5] = { 1, 2, 3, 4, 5 };
	Array<int> a(arr, 5);
	//a.print();

	Array<int>::iterator begin; //int* begin;
	Array<int>::iterator end;

	
}
