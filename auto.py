# # import numpy as np
# # import matplotlib
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
 
# # # 指定渲染环境
# # # %matplotlib notebook
# # # %matplotlib inline
 
# # x = np.linspace(0, 2*np.pi, 100)
# # y = np.sin(x)

# # fig = plt.figure(tight_layout=True)
# # plt.plot(x,y)
# # plt.grid(ls="--")
# # plt.show()


# # # def update_points(num):
# # #     '''
# # #     更新数据点
# # #     '''
# # #     point_ani.set_data(x[num], y[num])
# # #     return point_ani,
 
# # # x = np.linspace(0, 2*np.pi, 100)
# # # y = np.sin(x)
 
# # # fig = plt.figure(tight_layout=True)
# # # plt.plot(x,y)
# # # point_ani, = plt.plot(x[0], y[0], "ro")
# # # plt.grid(ls="--")
# # # # 开始制作动画
# # # ani = animation.FuncAnimation(fig, update_points, np.arange(0, 100), interval=100, blit=True)
 
# # # # ani.save('sin_test2.gif', writer='imagemagick', fps=10)
# # # plt.show()


# # def update_points(num):
# #     if num%5==0:
# #         point_ani.set_marker("*")
# #         point_ani.set_markersize(12)
# #     else:
# #         point_ani.set_marker("o")
# #         point_ani.set_markersize(8)
 
# #     point_ani.set_data(x[num], y[num])    
# #     text_pt.set_text("x=%.3f, y=%.3f"%(x[num], y[num]))
# #     return point_ani,text_pt,
 
# # x = np.linspace(0, 2*np.pi, 100)
# # y = np.sin(x)
 
# # fig = plt.figure(tight_layout=True)
# # plt.plot(x,y)
# # point_ani, = plt.plot(x[0], y[0], "ro")
# # plt.grid(ls="--")
# # text_pt = plt.text(4, 0.8, '', fontsize=16)
 
# # ani = animation.FuncAnimation(fig, update_points, np.arange(0, 100), interval=100, blit=True)
 
# # # ani.save('sin_test3.gif', writer='imagemagick', fps=10)
# # plt.show()



# from numpy import sin, cos
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.integrate as integrate
# import matplotlib.animation as animation

# G = 9.8  # acceleration due to gravity, in m/s^2
# L1 = 1.0  # length of pendulum 1 in m
# L2 = 1.0  # length of pendulum 2 in m
# M1 = 1.0  # mass of pendulum 1 in kg
# M2 = 1.0  # mass of pendulum 2 in kg


# def derivs(state, t):

#     dydx = np.zeros_like(state)
#     dydx[0] = state[1]

#     delta = state[2] - state[0]
#     den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
#     dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
#                 + M2 * G * sin(state[2]) * cos(delta)
#                 + M2 * L2 * state[3] * state[3] * sin(delta)
#                 - (M1+M2) * G * sin(state[0]))
#                / den1)

#     dydx[2] = state[3]

#     den2 = (L2/L1) * den1
#     dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
#                 + (M1+M2) * G * sin(state[0]) * cos(delta)
#                 - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
#                 - (M1+M2) * G * sin(state[2]))
#                / den2)

#     return dydx

# # create a time array from 0..100 sampled at 0.05 second steps
# dt = 0.05
# t = np.arange(0, 20, dt)

# # th1 and th2 are the initial angles (degrees)
# # w10 and w20 are the initial angular velocities (degrees per second)
# th1 = 120.0
# w1 = 0.0
# th2 = -10.0
# w2 = 0.0

# # initial state
# state = np.radians([th1, w1, th2, w2])

# # integrate your ODE using scipy.integrate.
# y = integrate.odeint(derivs, state, t)

# x1 = L1*sin(y[:, 0])
# y1 = -L1*cos(y[:, 0])

# x2 = L2*sin(y[:, 2]) + x1
# y2 = -L2*cos(y[:, 2]) + y1

# fig = plt.figure()
# ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
# ax.set_aspect('equal')
# ax.grid()

# line, = ax.plot([], [], 'o-', lw=2)
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


# def init():
#     line.set_data([], [])
#     time_text.set_text('')
#     return line, time_text


# def animate(i):
#     thisx = [0, x1[i], x2[i]]
#     thisy = [0, y1[i], y2[i]]

#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (i*dt))
#     return line, time_text


# ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
#                               interval=dt*1000, blit=True, init_func=init)
# plt.show()

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def beta_pdf(x, a, b):
    return (x**(a-1) * (1-x)**(b-1) * math.gamma(a + b)
            / (math.gamma(a) * math.gamma(b)))


class UpdateDist:
    def __init__(self, ax, prob=0.5):
        self.success = 0
        self.prob = prob
        self.line, = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 200)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 15)
        self.ax.grid(True)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.ax.axvline(prob, linestyle='--', color='black')

    def init(self):
        self.success = 0
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            return self.init()

        # Choose success based on exceed a threshold with a uniform pick
        if np.random.rand(1,) < self.prob:
            self.success += 1
        y = beta_pdf(self.x, self.success + 1, (i - self.success) + 1)
        self.line.set_data(self.x, y)
        return self.line,

# Fixing random state for reproducibility
np.random.seed(19680801)


fig, ax = plt.subplots()
ud = UpdateDist(ax, prob=0.7)
anim = FuncAnimation(fig, ud, frames=np.arange(100), init_func=ud.init,
                     interval=100, blit=True)
plt.show()