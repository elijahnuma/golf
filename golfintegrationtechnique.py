import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time as time

g = 9.81                # gravity
epsilon = 0.04          # collison detector

class ball:
    """ 
    ball class
    
    attributes: 
        x: float
            x position of object
        y: float
            y position of object
        mass: float
            mass of object
        restitution: float
            determines how much energy is lost during a bounce
            it is a percent, so 0-100%
        velocity_x: float
            velocity of object in x direction; default zero
        velocity_y: float
            velocity of object in y direction; default zero
        rolling: bool
            true if ball is rolling on a surface; default false
        in_hazard: bool
            true if ball is ball is in a hazard; default false
    """
    def __init__(self, x, y, mass, restitution, velocity_x=0, velocity_y=0, rolling=False, in_hazard=False):       
        self.x = x
        self.y = y
        self.mass = mass
        self.restitution = restitution
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.rolling = rolling
        self.in_hazard = in_hazard
    
    def reset(self, reset_pos):
        """ resets position of ball to starting position after hazard
        """
        self.x, self.y = reset_pos
        self.in_hazard = False
    
    def hit(self, impulse, angle):
        """ 
        hits ball in certain direction with a given impulse at an angle
        params:
            impulse: float
                impulse that generates a velocity (units of momentum)
            angle: float
                the angle the ball is hit at with respect to horizontal
        """
        velocity = impulse/self.mass
        self.velocity_x = velocity*np.cos(np.deg2rad(angle))
        self.velocity_y = velocity*np.sin(np.deg2rad(angle))
    
    def move(self, env, current_time, dt):
        """ 
        moves object at every time step
        params:
            env: level object
                environment that ball is moving in
            current_time: float 
                current time in simulation  
            dt: float
                time step for each calculation
        """
        # assumes ball is not rolling and checks rolling conditions
        self.rolling = False
        # makes sure that it doesn't check at the beginning of the simulation
        if (current_time > 10*dt):
            # checks every obstacle for collison
            for o in env.obstacles:
                # for horizontal obstacles; checks if vertical distance is less than epsilon
                # and if the horizontal component is within the bounds of the obstacle
                if o.orientation == 'horizontal' and abs(self.y - o.pos_y) < epsilon and o.pos_x <= self.x <= o.pos_x + o.length:
                    # checks if obstacle is a hazard and places ball in hazard
                    if o.hazard:
                        self.in_hazard = True
                    # if the ball doesn't sufficient velocity to get off obstacle, sets ball to 
                    # rolling mode
                    if abs(self.velocity_y) < epsilon:
                        # sets y position to obstacle's
                        self.y = o.pos_y        
                        # sets vertical velocity to 0 so ball doesn't move
                        self.velocity_y = 0
                        # ball is now rolling
                        self.rolling = True
                        # friction is equal to the obstacles coeff of friction
                        fric = o.mu
                    # otherwise deflects ball; checks if ball is coming from correct direction
                    # by checking if the sign of the velocity matches with the side the ball
                    # approaches the obstacle from
                    elif (self.y < o.pos_y) == (self.velocity_y > 0):
                        # velocity is dampened at proportionally to the obstacle's coeff of 
                        # friction and the ball's restitution; the velocity in the horizontal
                        # direction is damped less than the vertical direction; the reason
                        # for this is purely heuristic after I bounced some balls outside 
                        self.velocity_x *= np.sqrt((1-o.mu)*self.restitution)
                        self.velocity_y *= -((1-o.mu)*self.restitution)
                # logic follows from above 
                elif o.orientation == 'vertical' and abs(self.x - o.pos_x) < epsilon and o.pos_y <= self.y <= o.pos_y + o.length:
                    if (self.x < o.pos_x) == (self.velocity_x > 0):
                        self.velocity_x *= -self.restitution
                        self.velocity_y *= np.sqrt(self.restitution)
        # if ball isn't rolling, the frictional force calculated below will be 0
        if not self.rolling:
            fric = 0   
        
        def projectile(t, r):
            """ 
            dummy function for integration and ODEs describe equations of motion;
            check scipy documentation for more; dr_dt is velocity, dvr_dt is acceleration
            """
            x, v_x, y, v_y = r
            dx_dt = v_x
            # environment's wind force; frictional force acts opposite to motion of ball
            dvx_dt = env.wind_x - np.sign(self.velocity_x)*fric*g*self.mass
            dy_dt = v_y
            # gravitational force; environment's wind force
            dvy_dt = -g + env.wind_y
            # if ball is rolling there is no acceleration in y direction
            if self.rolling:
                dvy_dt = 0
            return dx_dt, dvx_dt, dy_dt, dvy_dt
        # solves and returns subsequent position from current conditions, LSODA is fastest method
        sol = si.solve_ivp(projectile, (current_time, current_time+dt), (self.x, self.velocity_x, self.y, self.velocity_y), method='RK23', t_eval=(current_time+dt,))
        self.x, self.velocity_x, self.y, self.velocity_y = [sol.y[i][0] for i in range(len(sol.y))]
        
class obstacle:
    """ 
    obstacle class
   
    attributes:
        pos_x: float
            x position of leftmost/bottom end 
        pos_y: float
            y position of leftmost/bottom end 
        length: float
            length of obstacle
        orientation: str
            orientation of obstacle (horizontal or vertical)
        mu: float
            coefficient of friction ; default 0
        color: str
            color of obstacle; default black
        hazard: bool
            True if obstacle is a hazard; default False
    """
    def __init__(self, pos_x, pos_y, length, orientation, mu=0, color='black', hazard=False):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = length
        self.orientation = orientation
        self.mu = mu
        self.color = color
        self.hazard = hazard
        
class level:
    """
    level class
    
    attributes:
        flag_location : float
            where the physical flag is, halfway between left and right
            bound of goal
        goal_height: float
            height of goal; same as height of the flag
        goal_left_bound: float
            takes left bound of goal
        goal_right_bound: float
            takes right bound of goal  
        obstacles: list of obstacle objects
            collection of obstacle objects for this level; default no obstacles 
        wind_x: float
            force of wind in level in x direction; calculated from wind and angle;
            default 0
        wind_y: float
            force of wind in level in y direction; calculated from wind and angle;
            default 0
        angle: float
            angle of wind in level with respect to horizontal        
    """
    def __init__(self, flag_location, flag_height, obstacles=None, wind=0, angle=0):
        self.flag_location = flag_location
        self.goal_height = flag_height
        self.goal_left_bound = self.flag_location - 0.05
        self.goal_right_bound = self.flag_location + 0.05
        self.obstacles = obstacles
        self.wind_x = wind*np.cos(np.deg2rad(angle))
        self.wind_y = wind*np.sin(np.deg2rad(angle))      
           
## balls
golf_ball = ball(0, 0, 0.05, 0.68)
tennis_ball = ball(0, 0, 0.05, 0.75)
basketball = ball(0, 0, 0.625, 0.88)
bowling_ball = ball(0, 0, 5, 0.05)

## obstacles
# obstacle unpack list
grass = ['horizontal', 0.4, '#567d46']
ice = ['horizontal', 0, '#a5f2f3']
sand = ['horizontal', 0.96, '#c2b280']
water = ['horizontal', 0, '#0f5e9c', True]
# horizontal obstacles
left_bound, right_bound, lower_bound, upper_bound = 0, 10, 0, 10
hor_obstacle_1_1 = obstacle(4.5, 1.5, 1, *grass)
hor_obstacle_6_1 = obstacle(7.5, 7, 1, *grass)
hor_obstacle_6_2 = obstacle(5, 2, 1, *grass)
hor_obstacle_6_3 = obstacle(6, 2, 0.5, *sand)
hor_obstacle_6_4 = obstacle(6.5, 2, 1, *grass)
hor_obstacle_6_5 = obstacle(3.5, 4, 1, *grass)

# vertical obstacles
ver_obstacle_3_1 = obstacle(5, 4, 1, 'vertical')
ver_obstacle_6_1 = obstacle(2, lower_bound, 2, 'vertical')

# ground obstacles and level bounds
ice_ground = [obstacle(left_bound, lower_bound, right_bound, *ice)]
grass_ground = [obstacle(left_bound, lower_bound, right_bound, *grass)]
sand_ground = [obstacle(left_bound, lower_bound, right_bound, *sand)]
ground_5 = [obstacle(left_bound, lower_bound, 2, *grass), 
            obstacle(2, lower_bound, right_bound, *water)]
level_bounds = [obstacle(left_bound, lower_bound, upper_bound, 'vertical'), 
                obstacle(left_bound, upper_bound, right_bound, 'horizontal'), 
                obstacle(right_bound, lower_bound, upper_bound, 'vertical')] 
                # [left boundary, upper boundary, right boundary]

## levels
lvl_1 = level(flag_location=5, flag_height=0, obstacles=[*ice_ground, *level_bounds]) # Blank level
lvl_2 = level(flag_location=5, flag_height=1.5, obstacles=[hor_obstacle_1_1, *grass_ground, *level_bounds]) # Elevated flag
lvl_3 = level(flag_location=5, flag_height=1.5, obstacles=[hor_obstacle_1_1, ver_obstacle_3_1, *grass_ground, *level_bounds]) # Elevated flag with vertical wall
lvl_4 = level(flag_location=5, flag_height=1.5, obstacles=[hor_obstacle_1_1, ver_obstacle_3_1, *sand_ground, *level_bounds]) # Sand level
lvl_5 = level(flag_location=5, flag_height=1.5, obstacles=[hor_obstacle_1_1, *ground_5, *level_bounds]) # Water Level
lvl_6 = level(flag_location=8, flag_height=7, obstacles=[hor_obstacle_6_1, hor_obstacle_6_2, hor_obstacle_6_3,
                                                         hor_obstacle_6_4, hor_obstacle_6_5, ver_obstacle_6_1, 
                                                         *ground_5, *level_bounds]) # The chef's special
lvl_7 = level(flag_location=5, flag_height=1.5, obstacles=[hor_obstacle_1_1, *grass_ground, *level_bounds], wind=-10, angle=45) # Windy level
lvls = [lvl_1, lvl_2, lvl_3, lvl_4, lvl_5, lvl_6, lvl_7]                                                         

# %%
## for bug testing, set impulse/mass = 50, angle 60; level 5; should rest at 6.49m

run = True
selected_ball = golf_ball
try:
    lvl = lvls[int(input('Enter level: '))-1]
except: 
    print('Enter valid level number.')
while run:
    hor = []
    ver = []
    hor_v = []
    ver_v = []
    t = 0
    dt = 0.0005
    starting_pos = (selected_ball.x, selected_ball.y)
    try:
        selected_impulse, selected_angle = float(input('Input impulse:')), float(input('Input angle:'))
    except:
        print('Exiting game.')
        break
    selected_ball.hit(selected_impulse, selected_angle)
    begin_time = time.time()
    while True:
        hor.append(selected_ball.x), ver.append(selected_ball.y)
        hor_v.append(selected_ball.velocity_x), ver_v.append(selected_ball.velocity_y)
        selected_ball.move(lvl, t, dt)
        t += dt  
        if lvl.goal_left_bound <= selected_ball.x <= lvl.goal_right_bound and lvl.goal_height <= selected_ball.y <= lvl.goal_height + epsilon:
            msg = 'Goal hit!'
            run = False
            break
        elif (selected_ball.x < left_bound) or (selected_ball.x > right_bound) or (selected_ball.y < lower_bound) or (selected_ball.y > upper_bound):
            selected_ball.reset(starting_pos)
            msg = 'Ball outside bounds.'
            break   
        elif selected_ball.in_hazard:
            selected_ball.reset(starting_pos)
            msg = 'Hazard encountered.'
            break
        elif (abs(selected_ball.velocity_x) < epsilon/10) and selected_ball.rolling:
            msg = 'Ball stopped due to insufficient speed.'
            break
        elif len(ver)*dt > 20:
            msg = 'In-game run time exceeded 20 seconds.'
            break
        
    end_time = time.time()
    run_time = end_time - begin_time
    # trajectory, figure is how plot displays in console, axis is plot's axes
    figure_length, figure_height = 10, 10
    axis_length, axis_height = right_bound, upper_bound
    obstacle_linewidth = 3.5
    plt.figure(figsize=(figure_length,figure_height))
    plt.plot(hor, ver, linewidth=2, color='0.5') # grey color
    plt.title('Ball trajectory')
    plt.xlabel('distance (m)')
    plt.ylabel('height (m)')
    plt.xlim([0, axis_length]) 
    plt.ylim([0, axis_height]) 
    for o in lvl.obstacles:
        if o.orientation == 'horizontal':
            plt.hlines(o.pos_y, o.pos_x, o.pos_x + o.length, linewidth=obstacle_linewidth, color=o.color)
        if o.orientation == 'vertical':
            plt.vlines(o.pos_x, o.pos_y, o.pos_y + o.length, linewidth=obstacle_linewidth, color=o.color) 
    plt.vlines(lvl.flag_location, lvl.goal_height, lvl.goal_height + 0.5)
    plt.gca().add_patch(patches.Rectangle((lvl.flag_location, lvl.goal_height + 0.4), width=0.25, height=0.1, facecolor='red', edgecolor='black'))
    plt.show()
    plt.close()
    print(f'Real run time = {run_time} s, In-game run time = {len(ver)*dt} s')
    print(f'Final ball position: {hor[-1]}')
    print(msg)
