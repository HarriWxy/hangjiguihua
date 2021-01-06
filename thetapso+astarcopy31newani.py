#coding: utf-8
import matplotlib.animation as Animation
import numpy as np 
import random
import matplotlib.pyplot as plt 
from queue import PriorityQueue

class Pso(object):
    # 粒子群算法类
    class Grid(object):
        # 产生一个格子地图类
        def __init__(self,dim):
            self.grid=np.zeros([dim,dim])
            for i in range(dim-1):
                for j in range(dim-1):
                    if random.random() < 0.3: # 有0.3倍的概率生成一个障碍
                        self.grid[i][j]=1
            self.grid[0][0]=0
                    # 0 1矩阵,1代表障碍

    class Astar(object):
        # A*算法类
        dx=[1,-1,0,0]
        dy=[0,0,1,-1]
        def calH(self,x,y):
            # 计算启发函数估计值
            # x,y是此时粒子的位置
            su= 0.5 * (self.w1+self.w2) * (self.x_des-x+self.y_des-y) # 1.2倍的两边之长
            return su

        def calKey(self,x1,y1,route):
            # 返回该点的节点值
            r=route[:]
            r.append([x1,y1])
            return [self.g[x1][y1]+self.calH(x1,y1),[x1,y1],r]

        def __init__(self,grid,x_line,p_dim,x_des,y_des):
            # x_des,y_des是目标位置
            super().__init__()
            self.x_des=x_des
            self.y_des=y_des
            self.grid=grid
            self.g=np.zeros_like(self.grid)
            self.w1=1 # 路径长度系数
            self.w2=99 # 障碍系数
            self.u=PriorityQueue()
            self.g[0][0]=0
            self.u.put(self.calKey(0,0,route=[]))
            self.res=self.getBlock(x_line,p_dim)

        def getBlock(self,x_line,p_dim):
            x_left=0
            y_left=0
            x_right=0
            y_right=0
            for i in range(p_dim+1):
                # 划分搜索区域:
                # 前一个粒子到下一个粒子之间的区域
                if i !=p_dim:
                    x_des= int( x_line[i] )
                    y_des= int( x_line[p_dim+i] )
                else:
                    x_des=self.x_des
                    y_des=self.y_des
                if x_des >= x_left:
                    x_right = x_des
                elif x_des < x_left:
                    x_right = x_left
                    x_left = x_des 
                if y_des >= y_left:
                    y_right = y_des
                elif y_des < y_left:
                    y_right = y_left
                    y_left = y_des
                self.Bfs(x_des,y_des,x_left,y_left,x_right,y_right)
                x_left=x_des
                y_left=y_des
            return self.u.get()
            

        def Bfs(self,x_des,y_des,x_left,y_left,x_right,y_right):
            # 想法是用A*的算法思想连接PSO搜索出的路径节点
            visited=np.zeros([x_right-x_left+1,y_right-y_left+1]) # 访问过的点
            # BFS
            while self.u.empty() == False:
                s=self.u.get()
                x=s[1][0]
                y=s[1][1]
                visited[x-x_left][y-y_left]=1
                if x==x_des and y==y_des:
                    self.u.__init__()
                    self.u.put(s)
                    break
                for k in range(4):
                    newS_x=x+self.dx[k]
                    newS_y=y+self.dy[k]
                    if newS_x < x_left or newS_x > x_right or \
                        newS_y < y_left or newS_y > y_right:
                        continue
                    elif visited[newS_x-x_left][newS_y-y_left]==1:
                        continue
                    # elif self.grid[newS_x][newS_y] > 0:
                    #     # self.g[newS_x][newS_y]=float("inf")-10
                    #     continue
                    visited[newS_x-x_left][newS_y-y_left]=1
                    self.g[newS_x][newS_y]=self.g[x][y] + self.w2*self.grid[newS_x][newS_y] + self.w1
                    self.u.put(self.calKey(newS_x,newS_y,s[2]))

    def __init__(self,p_num,dim,max_iter,x,y): 
        # p_num:粒子数目,允许有相同的粒子出现
        # dim:地图的维度,max_iter:最大迭代次数,x,y为目的地
        self.a=1 #每一段的最小距离
        self.grid=self.Grid(dim+1).grid # 生成的网格图
        self.p_num = p_num  # 粒子数量
        self.dim = dim  
        self.p_dim= 5 # 这是粒子的维度,后期考虑用ai调参
        self.max_iter = max_iter  # 迭代次数
        self.x = np.zeros((self.p_num, 2*self.p_dim))  # 所有粒子的位置和角度
        self.theta = np.zeros((self.p_num, 2*self.p_dim))
        self.pbest = np.zeros((self.p_num, 2*self.p_dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, 2*self.p_dim))
        self.p_fit = np.zeros(self.p_num)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值
        self.run = 0 # 已经走过的路程代价
        self.des_x=x # 目的
        self.des_y=y
        self.star=[float("inf")-10,0,0]
        self.ax=plt.subplot(1,2,1)
        self.scats,=self.ax.scatter([],[],color='r')
        self.lins,=self.ax.plot([], [], 'o-', lw=2)
        self.fitness=[]
        self.__init_Population()

    def fThetatoX(self):
        # theta To x
        for i in range(self.p_num):
            for j in range(2*self.p_dim):
                self.x[i][j] = self.dim * (np.sin(self.theta[i][j]) + 1)/2
                if (self.x[i][j]>=self.dim):
                    self.x[i][j]=self.dim-1
                elif (self.x[i][j]<0):
                    self.x[i][j]=0   

    def fThetatoXmax(self):
        x=np.zeros_like(self.gbest)
        for j in range(2*self.p_dim):
            x[j] = self.dim * (np.sin(self.gbest[j]) + 1)/2
            if x[j]>=self.dim:
                x[j]=self.dim-1
            elif x[j]<0:
               x[j]=0
        return x  

    def fit_func(self,x_line):
        # 代价函数,代价函数设定为距离之和,即两个粒子之间的距离之和
        # 加上这条直线是否会穿过障碍计算这两个点之间的矩形范围内的障碍之和
        # 由于一个粒子就是一条路径,于是代价函数设定为直接计算这条路径的代价
        star=self.Astar(self.grid,x_line,self.p_dim,self.des_x,self.des_y).res
        if self.star[0] > star[0]:
            self.star=star
        return  star[0]

    def fit_simple_func(self,x_line):
        # 由于每次迭代用大量的粒子群搜索比较复杂,由此设计简单计算路径代价的方法
        # 先走横纵线再走横线
        x_left=0
        y_left=0
        length=0
        for i in range(self.p_dim):
            x_des= int(x_line[i])
            y_des= int(x_line[self.p_dim+i])
            if x_des >= x_left:
                x_step=1
            else :
                x_step=-1
            if y_des >= y_left:
                y_step=1
            else:
                y_step=-1
            for x in range(x_left,x_des+x_step,x_step):
                length += 50 + 50 * self.grid[x][y_left]
            for y in range(y_left,y_des+y_step,y_step):
                length += 50 + 50 * self.grid[x_des][y]
            # length+=np.sqrt((-x_left+x_des)**2+(-y_left+y_des)**2)
            x_left=x_des
            y_left=y_des
        for x in range(x_left,self.des_x):
            length += 1 + 99 * self.grid[x][y_left]
        for y in range(y_left,self.des_y):
            length += 1 + 99 * self.grid[self.des_x][y]
        return length

    def __init_Population(self):
        # 初始化种群
        for i in range(self.p_num):
            for j in range(2*self.p_dim):
                self.theta[i][j]=random.uniform(-np.pi/2,np.pi/2)
                self.x[i][j] = self.dim * (np.sin(self.theta[i][j]) + 1)/2
            self.pbest[i]=self.theta[i]
            temp=self.fit_func(self.x[i])
            self.p_fit[i]=temp
            if temp < self.fit:
                self.fit=temp
                self.gbest=self.theta[i]
    
    def iter(self,temp):
        # 迭代函数
        F=1
        CR=0.5 # 选择变异的系数
        
        for k in range(self.max_iter):

            gamma=np.zeros_like(self.theta)
            for i in range(self.p_num):
                u = random.uniform(0,1)
                gamma[i] = u*self.pbest[i] + (1-u)*self.gbest
                for j in range(2*self.p_dim):
                    if gamma[i][j] > np.pi/2:
                        gamma[i][j]=np.pi/2
                    elif gamma[i][j] < -np.pi/2:
                        gamma[i][j]= -np.pi/2
                
            for i in range(self.p_num):
                # mutation
                li=list(range(self.p_num))
                li.remove(i)
                r = random.sample(li,3) # 随机选择的三个系数作为突变来源的选择
                v = gamma[r[0]] + F*(gamma[r[1]] - gamma[r[2]]) # F是突变的系数
                li=list(range(2*self.p_dim))
                rnbr=random.sample(li,random.randint(1,self.p_dim))
                for j in range(2*self.p_dim):
                    # crossover
                    if (random.random()<CR) or (j in rnbr):
                        if v[j]>np.pi/2:
                            v[j]=np.pi/2
                        elif v[j]<-np.pi/2:
                            v[j]=-np.pi/2
                        self.theta[i][j] = v[j]
                    else:
                        self.theta[i][j]  = gamma[i][j]

            self.fThetatoX()
            self.scats.set_data(self.x[0],self.x[1])
            x=[0.5]
            y=[self.dim-0.5]
            route=self.star[2]
            for i in route:
                x.append(i[1]+0.5)
                y.append(self.dim-i[0]-0.5)
            self.lins.set_data(x,y)

            for i in range(self.p_num):
                temp=self.fit_func(self.x[i])
                if temp < self.p_fit[i]: # 更新个体最优
                    temp=self.fit_func(self.x[i])
                    self.p_fit[i] = temp 
                    self.pbest[i] = self.theta[i]
                    if temp < self.fit:
                        self.gbest=self.theta[i]
                        self.fit=temp
            self.fitness.append(self.fit)

    def trans(self,x):  
        # 标注图像中的路径
        for i in range(self.p_dim): # 这里需要限制一下范围
            self.grid[int(x[i])][int(x[self.p_dim+i])]=5
            
    def animate_init(self):
        self.lins.setdata([],[])
        self.scats.setdata([],[])

    def drawLine(self):
        # 画出无人机路径图,直线连接
        # 方格边长
        a=1
        fig = plt.figure(tight_layout=True)
        for i in range(self.dim+1):
            for j in range(self.dim+1):
                if self.grid[i][j] > 0:
                    self.ax.fill_between([j,j+1,j+1,j],[self.dim-i-1,self.dim-i-1,self.dim-i,self.dim-i],color='k',alpha=0.2)
                else :
                    self.ax.fill_between([j,j+1,j+1,j],[self.dim-i-1,self.dim-i-1,self.dim-i,self.dim-i],color='grey',alpha=1)
        
        
        x=[0.5]
        y=[self.dim-0.5]
        route=self.star[2]
        for i in route:
            x.append(i[1]+0.5)
            y.append(self.dim-i[0]-0.5)
        plt.plot(x,y)
        plt.subplot(1,2,2)
        # self.trans(xxx)
        # plt.imshow(np.array(self.grid))
        # plt.plot(fitness)
        ani=Animation.FuncAnimation(fig,self.iter,range(self.max_iter),interval=50, blit=True, init_func=self.animate_init)
        # 输出代价值
        fitness=np.array(self.fitness)
        print(fitness)

        plt.show()

if __name__ == "__main__":
    # 随机产生一张图
    psodemo=Pso(50,30,50,29,29)
    psodemo.drawLine()