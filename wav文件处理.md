



```cpp
#define KALDI_SWAP2(a){ \
  int t=(reinterpret_cast<char *>(&a))[0]; \
        (reinterpret_cast<char *>(&a))[0]=(reinterpret_cast<char *>(&a))[1]; \
        (reinterpret_cast<char *>(&a))[1]=t;}

struct wav_struct {
	unsigned short channel;
	unsigned long frequency;
	unsigned short sample_num_bit;
	unsigned long data_size;
	char *data;
};

bool readWav(istream &in) {
	wav_struct WAV;
	in.seekg(0x14);
	in.read(reinterpret_cast< char * >(&WAV.channel), sizeof(WAV.channel));
	in.seekg(0x18);
	in.read((char*)&WAV.frequency, sizeof(WAV.frequency));
	in.seekg(0x22);
	in.read((char*)&WAV.sample_num_bit, sizeof(WAV.sample_num_bit));
	in.seekg(0x28);
	in.read((char*)&WAV.data_size, sizeof(WAV.data_size));
	cout << WAV.channel << endl;
	cout << WAV.frequency << endl;
	cout << WAV.sample_num_bit << endl;
	cout << WAV.data_size << endl;
	WAV.data = new char[WAV.data_size];
	char* buf_c;
	vector<char> sound;
	in.seekg(0x2c);
	in.read((char *)WAV.data, sizeof(char)*WAV.data_size);
	cout << in.gcount() << endl;
	for (unsigned long i = 0; i<WAV.data_size; i++) {
		char data_sound = WAV.data[i];
		sound.push_back(data_sound);
	}
	buf_c = &sound[0];
	uint16_t *data_prt = reinterpret_cast<uint16_t*>(&sound[0]);
	for (int i = 0; i < 5000; i++) {
		uint16_t k = *data_prt++;
		//KALDI_SWAP2(k);
		//cout << (float)k << " ";
		//printf("0x%x", sound[i]);
	}

	vector<int> input({ 1,2,3,4,5 });
	vector<int> input2;
	cout<<input.size();
	int arr[5];
	std::copy(input.begin(), input.begin()+5, arr);
	for (int i = 0; i < 5; i++) {
		cout<<arr[i]<<endl;
	}
	cout << COUNT(arr) << endl;
	input2 = { input.begin(), input.begin() + 5 };
	for (int i = 0; i < input2.size(); i++) {
		cout << input2[i];
	}
	return true;
}

int main()
{
	ifstream in("D:\\project\\python\\test\\16k_3s.wav");
	//fstream in;
	//in.open("D:\\project\\python\\test\\16k_3s.wav", ios::binary | ios::in);
	readWav(in);
	system("Pause");
	return 0;
}

```

