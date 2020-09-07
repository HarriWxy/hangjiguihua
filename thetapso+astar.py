#coding: utf-8
import numpy as np 
import random
import matplotlib.pyplot as plt 
from queue import PriorityQueue

class Pso(object):
    # 粒子群算法类
    class Grid(object):
        # 产生一个格子地图类
        grid=[]
        def __init__(self,dim):
            for i in range(dim):
                temp=[]
                for j in range(dim):
                    temp.append(10*random.random)
                    # 0 1矩阵,1代表障碍
                self.grid.append(temp)

    class Astar(object):
        # A*算法类
        def calH(self,x,y):
            # 计算启发函数估计值
            su=0 
            for i in range(x+1,self.grid.shape[0]+1):
                su+=self.grid[i][0] * self.w2 # 障碍系数
                su+=1 * self.w1 # 路径长度系数
            for j in range(y+1,self.grid.shape[1]+1):
                su+=self.grid[j][j] * self.w2
                su+=1 * self.w1
            return su

        def calKey(self,x1,x2,y1,y2):
            
            return [g[x1][x2]+self.calH(x1,x2)]

        def __init__(self,grid,x_line):
            super().__init__()
            self.grid=np.array(grid)
            self.g=np.zeros_like(self.grid)
            self.w1=0.3 # 路径长度系数
            self.w2=0.7 # 障碍系数
            self.u=PriorityQueue()
            self.g[0][0]=0
            self.u.put(self.calKey(0,0))
            self.Bfs(x_line)
            
        def Bfs(self,x_line):

            while ~self.u.empty:
                pass
            return True
            
        def calObs(self, x1, y1, x2, y2):
            # 使用a*算法计算这个矩形中最少的路径开销
            
            u=PriorityQueue()
            temp=0
            return temp


    def __init__(self,p_num,dim,max_iter,x,y): 
        # p_num:粒子数目,允许有相同的粒子出现
        # dim:地图的维度,max_iter:最大迭代次数,x,y为目的地
        self.a=1 #每一段的最小距离
        self.grid=Grid(dim).grid # 生成的网格图
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.p_num = p_num  # 粒子数量
        self.dim = dim  
        self.k=1
        self.p_dim=self.k*dim # 这是粒子的维度,后期考虑用ai调参
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

    def fit_func(self,x_line):
        # 代价函数,代价函数设定为距离之和,即两个粒子之间的距离之和
        # 加上这条直线是否会穿过障碍计算这两个点之间的矩形范围内的障碍之和
        # 由于一个粒子就是一条路径,于是代价函数设定为直接计算这条路径的代价
        len_Of_Rute = np.sqrt(x_line[0]**2 + x_line[self.p_dim]**2) # 这个或许可以不用
        the_of_rute = self.calObs(0,x_line[0],0,x_line[self.p_dim])
        for i in range(1,self.p_dim):
            len_Of_Rute += np.sqrt( (x_line[i]-x_line[i-1])**2 + (x_line[i]-x_line[i-1])**2 )
        len_Of_Rute+=np.sqrt( (self.des_x-x_line[self.p_dim-1])**2 + (self.des_y-x_line[2*self.p_dim-1])**2 )
        

        return  len_Of_Rute+the_of_rute

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
    
    def iter(self):
        # 迭代函数
        F=1
        CR=0.5 # 选择变异的系数
        fitness=[]
        for k in range(self.max_iter):
            for i in range(self.p_num):
                temp=self.fit_func(self.x[i])
                if temp < self.p_fit[i]: # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.theta[i]
                    if self.p_fit[i] < self.fit:
                        self.gbest=self.theta[i]
                        self.fit=self.p_fit[i]

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
                    
            fitness.append(self.fit)
        # self.trans()
        return fitness

    def trans(self):  
        # 标注图像中的路径
        temp=0
        for i in range(self.dim): # 这里需要限制一下范围
            self.grid[i][int(self.gbest[i])]=50
            for j in range(min(temp,int(self.gbest[i])),max(temp,int(self.gbest[i]))):
                self.grid[i][j]=30
            temp=int(self.gbest[i])

    def drawLine(self):
        # 画出无人机路径图,直线连接
        # 方格边长
        a=1
        plt.figure()
        fitness=np.array(self.iter())
        print(fitness)
        plt.subplot(1,2,1)
        for i in range(self.dim):
            for j in range(self.dim):
                if self.grid[i][j] > 0:
                    plt.fill_between([j,j+1,j+1,j],[self.dim-i-1,self.dim-i-1,self.dim-i,self.dim-i],color='k',alpha=0.5)
                else :
                    plt.fill_between([j,j+1,j+1,j],[self.dim-i-1,self.dim-i-1,self.dim-i,self.dim-i],color='grey',alpha=0.5)
        # plt.imshow(np.array(self.grid))
        # psodemo.trans()
        plt.subplot(1,2,2)
        plt.imshow(np.array(self.grid))
        # plt.plot(fitness)
        plt.show()


if __name__ == "__main__":
    # 随机产生一张图
    psodemo=Pso(30,10,30,29,29)
    psodemo.drawLine()