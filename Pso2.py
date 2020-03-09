#coding: utf-8
import numpy as np 
import random
import matplotlib.pyplot as plt 

grid=[]
# 随机产生一张图
class Grid(object):
    global grid
    def __init__(self,dim):
        for i in range(dim):
            temp=[]
            for j in range(dim):
                temp.append(random.randint(0,10))
            grid.append(temp)
# 粒子群种群
class Pso(object):
    global grid
    def __init__(self,p_num,dim,max_iter,x,y): 
        Grid(dim)
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.p_num = p_num  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.x = np.zeros((self.p_num, self.dim))  # 所有粒子的位置和速度
        self.v = np.zeros((self.p_num, self.dim))
        self.pbest = np.zeros((self.p_num, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.p_num)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值
        self.run=0 #已经走过的路程代价
        self.des_x=x
        self.des_y=y
        self.__init_Population()
    # 代价函数
    def func(self,x):
        ans=0
        temp=0
        for i in range(self.dim):
            if (x[i]>9):
                x[i]=9
            if (x[i]<0):
                x[i]=0
            ans+=grid[i][int(x[i])]
            for j in range(min(temp,int(x[i])),max(temp,int(x[i]))):
                ans+=grid[i][j]
            temp=int(x[i])
        return  ans
    # 初始化种群
    def __init_Population(self):
        for i in range(self.p_num):
            for j in range(1,self.dim-1):
                self.x[i][j]=random.randint(0,9)
                self.v[i][j]=random.randint(-1,1)
            self.x[i][self.dim-1] = (self.dim-1)
            self.pbest[i]=self.x[i]
            temp=self.func(self.x[i])
            self.p_fit[i]=temp
            if temp < self.fit:
                self.fit=temp
                self.gbest=self.x[i]
    
    # 迭代函数
    def iter(self):
        fitness=[]
        for k in range(self.max_iter):
            for i in range(self.p_num):
                temp=self.func(self.x[i])
                if temp < self.p_fit[i]: #更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.x[i]
                    if self.p_fit[i] < self.fit:
                        self.gbest=self.x[i]
                        self.fit=self.p_fit[i]
            for i in range(self.p_num):
                self.v[i]=self.w*self.v[i]+self.c1*self.r1*(self.pbest[i]-self.x[i])+\
                            self.c2*self.r2*(self.gbest-self.x[i])
                self.x[i]=self.x[i]+self.v[i]
            fitness.append(self.fit)
        # self.trans()
        return fitness
    # 标注图像中的路径
    def trans(self):
        temp=0
        for i in range(self.dim):
            if (self.gbest[i]>10):
                self.gbest[i]=9
            if (self.gbest[i]<0):
                self.gbest[i]=0
            grid[i][int(self.gbest[i])]=50
            for j in range(min(temp,int(self.gbest[i])),max(temp,int(self.gbest[i]))):
                grid[i][j]=30
            temp=int(self.gbest[i])

psodemo=Pso(30,10,50,10,10)
plt.figure(1)
fitness=np.array(psodemo.iter())
print(fitness)
plt.subplot(1,2,1)
plt.imshow(np.array(grid))
psodemo.trans()
plt.subplot(1,2,2)
plt.imshow(np.array(grid))
# plt.plot(fitness)
plt.show()