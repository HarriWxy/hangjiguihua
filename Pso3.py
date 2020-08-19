#coding: utf-8
import numpy as np 
import random
import matplotlib.pyplot as plt 
# 重新画格子吧
class Grid(object):
    grid=[]
    def __init__(self,dim):
        for i in range(dim):
            temp=[]
            for j in range(dim):
                temp.append(random.randint(0,1))
                # 0 1矩阵,1代表障碍
            self.grid.append(temp)
# 粒子群种群
class Pso(object):
    # p_num:粒子数目,dim:地图的维度,max_iter:最大迭代次数,x,y为目的地
    def __init__(self,p_num,dim,max_iter,x,y): 
        self.grid=Grid(dim).grid # 生成的网格图
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.p_num = p_num  # 粒子数量
        self.dim = dim  
        self.p_dim=2*dim # 这是粒子的维度,后期考虑用ai调参
        self.max_iter = max_iter  # 迭代次数
        self.x = np.zeros((self.p_num, self.p_dim))  # 所有粒子的位置和速度
        self.v = np.zeros((self.p_num, self.p_dim))
        self.pbest = np.zeros((self.p_num, self.p_dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.p_dim))
        self.p_fit = np.zeros(self.p_num)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值
        self.run = 0 # 已经走过的路程代价
        self.des_x=x # 目的
        self.des_y=y
        self.__init_Population()
    # 代价函数,代价函数设定为前后两个粒子之间的差值最小
    # 由于一个粒子就是一条路径,于是代价函数设定为直接计算这条路径的代价
    def fit_func(self,x_line):
        ans=0
        for i in range(1,self.dim):
            ans+=1
            ans+=x_line[i]-x_line[i-1]
            if x_line[i]==self.des_x:
                break
        return  ans
    # 初始化种群
    def __init_Population(self):
        for i in range(self.p_num):
            for j in range(1,self.dim-1):
                self.x[i][j]=random.randint(0,self.dim)
                self.v[i][j]=random.randint(-1,1)
            self.x[i][self.dim-1] = (self.dim-1)
            self.pbest[i]=self.x[i]
            temp=self.fit_func(self.x[i])
            self.p_fit[i]=temp
            if temp < self.fit:
                self.fit=temp
                self.gbest=self.x[i]
    
    # 迭代函数
    def iter(self):
        fitness=[]
        for k in range(self.max_iter):
            for i in range(self.p_num):
                temp=self.fit_func(self.x[i])
                if temp < self.p_fit[i]: # 更新个体最优
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
        for i in range(self.dim): # 这里需要限制一下范围，但我不记得为什么
            if (self.gbest[i]>self.dim):
                self.gbest[i]=self.dim
            if (self.gbest[i]<0):
                self.gbest[i]=0
            grid[i][int(self.gbest[i])]=50
            for j in range(min(temp,int(self.gbest[i])),max(temp,int(self.gbest[i]))):
                grid[i][j]=30
            temp=int(self.gbest[i])

if __name__ == "__main__":
    # 方格边长
    a=1 
    # 随机产生一张图
    psodemo=Pso(30,10,30,29,29)
    grid=psodemo.grid
    # 画图
    plt.figure()
    fitness=np.array(psodemo.iter())
    print(fitness)
    plt.subplot(1,2,1)
    plt.imshow(np.array(grid))
    psodemo.trans()
    plt.subplot(1,2,2)
    plt.imshow(np.array(grid))
    # plt.plot(fitness)
    plt.show()