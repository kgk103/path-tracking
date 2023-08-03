#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

import numpy as np
import math
import matplotlib.pyplot as plt


class APFPlanner:
    def __init__(self, env_size=100, alpha=0.1, n=1, zeta=1, U_star=0.1, q_star=2, d_star=5,
                 robot_radius=2, obstacle_size=3, sensor_range=7, path=[]):
        self.env_size = env_size
        self.alpha = alpha
        self.n = n
        self.zeta = zeta
        self.U_star = U_star
        self.q_star = q_star
        self.d_star = d_star
        self.robot_radius = robot_radius
        self.obstacle_size = obstacle_size
        self.sensor_range = sensor_range
        self.path = path

    def get_path_coordinates(self):
        unique_coords = []
        for coord in self.path:
            if coord not in unique_coords:
            	unique_coords.append(coord)
        return unique_coords

    def distance(self, q1, q2, rr=0, so=0):
        return round(np.linalg.norm(np.abs(q2 - q1)), 2) - rr - so

    def u_att(self, q, q0):
        d = self.distance(q, q0)
        if d <= self.d_star:
            return self.zeta * (d ** 2) * 0.5
        else:
            return self.d_star * self.zeta * d - 0.5 * self.zeta * (self.d_star ** 2)

    def att_grad(self, q, q0):
        d = self.distance(q, q0)
        if d <= self.d_star:
            return self.zeta * (q - q0)
        else:
            return self.d_star * self.zeta * (q - q0) / d

    def u_rep(self, q, oq):
        r = []
        for array in oq.T:
            array = array.reshape(2, 1)
            d = self.distance(q, array, self.robot_radius, self.obstacle_size)
            if d <= self.sensor_range:
                r.append(0.5 * self.n * ((1 / d) - (1 / self.q_star)) ** 2)
            else:
                r.append(0)
        return sum(r)

    def rep_grad(self, q, oq):
        r = np.zeros((2, 1))
        for array in oq.T:
            array = array.reshape(2, 1)
            d = self.distance(q, array, self.robot_radius, self.obstacle_size)
            if d <= self.sensor_range:
                r = np.append(r, (self.n * ((1 / self.q_star) - (1 / d)) * ((q - array) / (d ** 3))), axis=1)
            else:
                r = np.append(r, np.zeros((2, 1)), axis=1)
        return np.sum(r, axis=1).reshape(2, 1)

    def gradient_descent(self, q, oq, q0, max_iter):
        U = self.u_rep(q, oq) + self.u_att(q, q0)
        U_hist = [U]
        q_hist = q

        for i in range(max_iter):
            if U > self.U_star:
                grad_total = self.rep_grad(q, oq) + self.att_grad(q, q0)
                q = q - self.alpha * (grad_total / np.linalg.norm(grad_total))
                U = self.u_rep(q, oq) + self.u_att(q, q0)
                q_hist = np.hstack((q_hist, q))
                U_hist.append(U)

                if i % 15 == 0:
                    self.path.append([(q[0, 0]), (q[1, 0])])

            else:
                print("Algorithm converged successfully, and the robot has reached the goal location")
                return q_hist, U_hist

        print("Robot is either at a local minimum or the loop ran out of maximum iterations")
        return q_hist, U_hist
        
        
    def plan(self, q, oq, q0, max_iter):
        self.path = []
        q_hist, U_hist = self.gradient_descent(q, oq, q0, max_iter)
        
        return q_hist, U_hist
        
  
        # Parameters
k = 0.1  # look forward gain
Lfc = 2.0# [m] look-ahead distance
#Lfc = 2.5
Kp = 1.0
#Ki = 0.1  # Integral gain
#Kd = 0.01  # Derivative gain
#prev_error = 0.0
#integral = 0.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 0.3  # [m] wheel base of vehicle
linear_velocity = 5.0 #[m/s]
#linear_velocity = 2.5

show_animation = True
'''
def pid_control(target, current):
    global prev_error, integral

    # Calculate the error (difference between target and current state)
    error = target - current

    # Calculate the integral and prevent windup
    integral += error * dt
    if abs(integral) > 1.0:
        integral = np.sign(integral) * 1.0

    # Calculate the derivative
    derivative = (error - prev_error) / dt

    # Calculate the PID control input
    control_input = Kp * error + Ki * integral + Kd * derivative

    # Update previous error for the next iteration
    prev_error = error

    return control_input
'''
def proportional_control(target, current):
    a = Kp * (target - current)

    return a



class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)
        
class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)
        
class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf
'''
def sgn (num):  
    if num >= 0:
        return 1
    else:
        return -1
'''

def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    closest_point = None
    lookahead_distance = Lfc  # Set the desired lookahead distance

    if ind < len(trajectory.cx):
        for i in range(ind, len(trajectory.cx)):
            tx = trajectory.cx[i]
            ty = trajectory.cy[i]
            distance = math.hypot(state.rear_x - tx, state.rear_y - ty)
            if lookahead_distance < distance:
                closest_point = (tx, ty)
                ind = i
                break
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        distance = math.hypot(state.rear_x - tx, state.rear_y - ty)
        if lookahead_distance < distance:
            closest_point = (tx, ty)
            ind = len(trajectory.cx) - 1

    if closest_point is not None:
        tx, ty = closest_point
        alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw  # Target Heading - Current Heading 
        delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)
        angular_vel =(linear_velocity * delta) / WB  
    else:
        alpha = math.atan2(trajectory.cx[1] - state.rear_y, trajectory.cx[0] - state.rear_x) - state.yaw
        delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)
        angular_vel = (linear_velocity * delta) / WB
        #delta = 0.0
        #angular_vel = 0.0
        
    if delta > math.pi:
        delta -= 2 * math.pi
    elif delta < -math.pi:
        delta += 2 * math.pi
    if delta > math.pi/6 or delta < -math.pi/6:
        sign = 1 if delta > 0 else -1
        delta = sign * math.pi/4
        
    return delta, ind, angular_vel


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

state = State()

def odom_callback(odom_msg):
	global state

    # Extract position and orientation from Odometry message
	position = odom_msg.pose.pose.position
	orientation = odom_msg.pose.pose.orientation

    # Convert orientation quaternion to yaw angle (in radians)
	_, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])


	while yaw > math.pi:
		yaw -= 2 * math.pi
	while yaw < -math.pi:
		yaw += 2 * math.pi

	# Update the state with the received information
	state.x = position.x
	state.y = position.y
	state.yaw = yaw
	state.v = odom_msg.twist.twist.linear.x
	print("Received Odometry:")
	print("Position: ({}, {})".format(state.x, state.y))
	print("Orientation (Yaw):", state.yaw)
	print("Linear Velocity:", state.v)
    #state.angular_vel = odom_msg.twist.twist.angular.z


def main():
	rospy.init_node('usv_vel_ctrl', anonymous=False)
	pub = rospy.Publisher('volta_base_controller/cmd_vel', Twist, queue_size = 100)
	sub = rospy.Subscriber('/poseupdate', Odometry, odom_callback, queue_size = 100)
	#path_publisher = rospy.Publisher('/volta_base_controller/odom', Path, queue_size=1)

	

	apf_planner = APFPlanner()
	env_size = 100
	q = np.array([[0], [0]])
	#np.random.seed(123)
	#oq= np.array([[30, 40], [90, 30]]).T
	oq= np.array([[21, 20], [60, 60]]).T
#     	oq=np.random.randint(10,env_size,size=(2,20))
#     	oq = np.array([[21, 20], [90, 100], [92, 84], [83, 84], [82, 97], [40, 43]]).T

	q0 = np.array([[10], [0]])
	print("The start position is:\n", q)
	print("The goal position is:\n", q0)
	print("The obstacles are:\n", oq)
#     	print("The path generated is: ")
	max_iter = 2000

	q_hist, _ = apf_planner.plan(q, oq, q0, max_iter)
	
	#path_coordinates = apf_planner.get_smoothed_path_coordinates()    
	#print("Path coordinates:", path_coordinates)	
        
        
	path= apf_planner.get_path_coordinates() 
	print(path)
    
	cx=[]
	cy=[]
	
	goal= Point()
	goal.x= path[len(path)-1][0]
	goal.y= path[len(path)-1][1]
	
	rate = rospy.Rate(70)
    
	for i in range(len(path)):
                cx.append(path[i][0])
                cy.append(path[i][1])
                
                

	target_speed = 10.0 / 3.6  # [m/s]
	
	
	T = 200.0  # max simulation time
	global state
	
	state = State()
	
    	# initial state
	#state = State(x=-0.0, y=-2.0, yaw=0.0, v=0.0)

	lastIndex = len(cx) - 1
	time = 0.0
	states = States()
	states.append(time, state)
	target_course = TargetCourse(cx, cy)
	target_ind, _ = target_course.search_target_index(state)
	#rate = rospy.Rate(50)
	
	while not rospy.is_shutdown() and  T >= time and lastIndex > target_ind:

                # Calc control input
                ai = proportional_control(target_speed, state.v)
                di, target_ind ,ang_vel = pure_pursuit_steer_control(
                state, target_course, target_ind)
             
                print("ang_velocity",ang_vel)
                #print("target speed is ", target_speed)
                move_cmd = Twist()
                move_cmd.linear.x = target_speed
                move_cmd.angular.z = ang_vel
        
                pub.publish(move_cmd)

                state.update(ai, di)  # Control vehicle

                time += dt
                states.append(time, state)
                rate.sleep()
                

                if show_animation:  # pragma: no cover
                	plt.cla()
                	# for stopping simulation with the esc key.
                	plt.gcf().canvas.mpl_connect(
                	'key_release_event',
                	lambda event: [exit(0) if event.key == 'escape' else None])
                	plot_arrow(state.x, state.y, state.yaw)
                	plt.plot(cx, cy, "-r", label="course")
                	plt.plot(states.x, states.y, "-b", label="trajectory")
                	plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                	plt.axis("equal")
                	plt.grid(True)
                	#plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
                	plt.pause(0.001)
	


    # Test
	assert lastIndex >= target_ind, "Cannot goal"
    

	if show_animation:  # pragma: no cover
        	plt.cla()
        	plt.plot(cx, cy, ".r", label="course")
        	plt.plot(states.x, states.y, "-b", label="trajectory")
        	plt.legend()
        	plt.xlabel("x[m]")
        	plt.ylabel("y[m]")
        	plt.axis("equal")
        	plt.grid(True)
        

        	plt.subplots(1)
        	plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        	plt.xlabel("Time[s]")
        	plt.ylabel("Speed[km/h]")
        	plt.grid(True)
        	plt.show()


if __name__ == '__main__':
    main()
    


