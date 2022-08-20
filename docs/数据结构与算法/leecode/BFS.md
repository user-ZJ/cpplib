# BFS

BFS——广度优先算法(Breadth First Search)

BFS 问题的本质就是让你在一幅「图」中找到从**起点start到终点target的最近距离**

比如走迷宫,有的格子是围墙不能走,从起点到终点的最短距离是多少?如果这个迷宫带「传送门」可以瞬间传送呢?

再比如说两个单词,要求你通过某些替换,把其中一个变成另一个,每次只能替换一个字符,最少要替换几次?
再比如说连连看游戏,两个方块消除的条件不仅仅是图案相同,还得保证两个方块之间的最短连线不能多于两个拐点。你玩连连看,点击两个坐标,游
戏是如何判断它俩的最短连线有几个拐点的?

```cpp
// 计算从起点 start 到终点 target 的最近距离
int BFS(Node start, Node target) {
	Queue<Node> q; // 核心数据结构
	Set<Node> visited; // 避免走回头路
	q.offer(start); // 将起点加入队列
	visited.add(start);
	int step = 0; // 记录扩散的步数
	while (q not empty) {
		int sz = q.size();
		/* 将当前队列中的所有节点向四周扩散 */
		for (int i = 0; i < sz; i++) {
			Node cur = q.poll();
			/* 划重点:这里判断是否到达终点 */
        	if (cur is target)
				return step;
			/* 将 cur 的相邻节点加入队列 */
			for (Node x : cur.adj())
				if (x not in visited) {
					q.offer(x);
					visited.add(x);
				}
		}
		/* 划重点:更新步数在这里 */
		step++;
	}
}
```

