# ready to run example: PythonClient/multirotor/hello_drone.py
import airsim
import os
import numpy as np 
# connect to the AirSim simulator 
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
x=2
y=2
z=2
k=100
client.takeoffAsync()

# client.moveByVelocityAsync()
# while (True):
#     client.moveByVelocityAsync(1,1,-1,2)

while(client.simGetCollisionInfo().has_collided==False):
    client.moveByVelocityAsync(1,1,-1,2)
print(client.getGpsData())
coll=client.simGetCollisionInfo()
print(coll.position) 

# path=np.array([[-10, 10, -10, 5],[-15, 15, -15, 5],[-20,-20, -20, 5]]).join()
# client.moveOnPathAsync(path)
print("plause")
client.simPause(True)
