# 头文件
	#include <iostream>     // std::cout、std::endl
	#include <sys/socket.h> // socket、bind、listen、accept
	#include <netinet/in.h> // htonl、htons、sockaddr_in、INADDR_ANY
	#include <unistd.h>     // write、close
	#include <string.h>     // bzero
	#include <arpa/inet.h>  // inet_addr

# 服务端
	#define PORT 7000  //端口号
	#define QUEUE 100  //队列长度

	/* 创建套接字 */
	// 常量 AF_INET 位于 bits/socket.h，经由 sys/socket.h 导入
    // 常量 SOCK_STREAM 位于 bits/socket_type.h，先后经由 bits/socket.h、sys/socket.h 导入
	int ss = socket(AF_INET, SOCK_STREAM, 0);
	if(ss<0){
    	cout << "create socketfd failed" <<endl;
    	return -1;
	}else{
    	cout << "created socketfd success"<<endl;
	}

	/* 准备地址端口信息 */
    // 常量 INADDR_ANY 位于 netinet/in.h
	// 结构体 sockaddr_in 位于 netinet/in.h
	struct sockaddr_in servaddr;
	bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
	servaddr.sin_port = htons(PORT);

	/* 绑定套接字与地址端口 */
	int bind_ok = bind(ss, (struct sockaddr *)&servaddr, sizeof(servaddr));
	if(bind_ok<0){
    	cout << "bind socket with server address  failed" <<endl;
    	return -1;
	}else{
    	cout<<"bind socket with server addres success"<<endl;
	}

	/* 监听端口 */
	int listen_ok = listen(ss, QUEUE);
	if(listen_ok<0){
    	cout << "listen socket failed" <<endl;
    	return -1;
	}else{
    	cout << "listen socket success" <<endl;
	}

	struct sockaddr_in client_addr;
	socklen_t length = sizeof(client_addr);
	while(1){
		// 监听到有请求，创建交互的套接字
		int conn = accept(ss, (struct sockaddr*)&client_addr, &length);
    	if(conn<0){
      		cout<<"create connection socked failed"<<endl;
      		return -1;
    	}else{
      		cout<<"create connection socked success"<<endl;
    	}
		char buffer[1024];
    	memset(buffer, 0 ,sizeof(buffer));
    	int len = recv(conn, buffer, sizeof(buffer), 0);
		cout<<"start to search "<<buffer<<endl;
    	string result = search(10010,100,buffer);
    	cout<<"search result "<<result<<endl;
    	send(conn, result.c_str(),result.length(), 0);
    	cout<<"socked disconnect"<<endl;
    	close(conn);
	}
	close(ss)

# 客户端
	string query = "test"
	int sock_cli = socket(AF_INET,SOCK_STREAM, 0);
	struct sockaddr_in servaddr;
	memset(&servaddr, 0, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(MYPORT);
	servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
	int conn = connect(sock_cli, (struct sockaddr *)&servaddr, sizeof(servaddr));
	if(conn < 0){
    	cout<< "socket connect error"<<endl;
    	return -1;
	}else{
    	cout << "socket connect success"<<endl;
	}
	char recvbuf[BUFFER_SIZE];
	memset(recvbuf, 0, sizeof(recvbuf));
	send(sock_cli, query.c_str(), query.length(),0);
	recv(sock_cli, recvbuf, sizeof(recvbuf),0);
	cout<<recvbuf<<endl;
	memset(recvbuf, 0, sizeof(recvbuf));
	close(sock_cli);
		

	

	

