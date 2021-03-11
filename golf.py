import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time as time
import itertools 

g = 9.81                                                            # gravity
epsilon = 0.04                                                      # collison detector
left_bound, right_bound, lower_bound, upper_bound = 0, 10, 0, 10    # wall coordinates
figure_length, figure_height = 10, 10                               # plot size in console
corner = (epsilon, epsilon)                                         # lower left corner
ground_spawn = (left_bound+epsilon, right_bound-epsilon)            # entire ground
obstacle_limit = 15                                                 # number of obstacles allowed in a level that can be represented

class Game:
    """ 
    game object
    
    args:
        level (Level object or int): current level in play
        ball: (Ball object or str): current ball in use, check balls below for different types; default golf ball
        max_score (int): max score allowed for game until exiting; default 10
        mode (str): manual allows for manual input, replay plots trajectory only, training hides all info; default training
        randomize (bool): determines if level randomizes flag and ball position
    
    attributes:
        max_score (int): max score allowed for game until exiting; default 10
        mode (str): manual allows for manual input, replay plots trajectory only, training hides all info; default training
        score (int): current score
    
    """
    def __init__(self, level, ball='golf', max_score=10, mode='training', randomize=False):
        try:
            if type(level) == int:
                self.level = lvls[level]
                self.level.randomize = randomize
            elif type(level) == Level:
                self.level = level
        except KeyError:
            raise KeyError(f'{level} is not valid level') from None
        try:
            if type(ball) == str:
                self.ball = balls[ball]
            elif type(ball) == Ball:
                self.ball = ball
        except KeyError:
            raise KeyError(f'{ball.capitalize()} is not valid ball') from None
        self.max_score = max_score
        self.mode = mode
        self.level.reset()
        self.reset()
        
    def reset(self):
        """ reinitializes game """
        self.score = 0
        self.ball.reset(self.level.origin)
        
    def hit_ball(self, impulse=0, angle=0, return_trajectory=False, save_trajectory=False):
        """  
        hits ball
        for bug testing, set impulse/mass = 50, angle 60; level 5 (water level); should rest at 6.48m
        
        for args, check Ball.hit() documentation
        args:
            return_trajectory (bool): returns hor, ver as tuple of floats; default False
            save_trajectory (bool): saves trajectory path; default False

        """
        if self.mode == 'manual':
            plot_trajectory, manual = True, True
        elif self.mode == 'training':
            plot_trajectory, manual = False, False
        else:
            raise KeyError(f'{self.mode.capitalize()} mode does not exist')
        hor, ver, hor_v, ver_v = [], [], [], []
        t = 0
        dt = 0.001
        starting_pos = (self.ball.x, self.ball.y)
        self.ball.hit(impulse, angle)
        begin_time = time.time()
        while True:
            self.ball.move(self.level, t, dt)
            t += dt
            hor.append(self.ball.x), ver.append(self.ball.y)
            hor_v.append(self.ball.velocity_x), ver_v.append(self.ball.velocity_y)
            # checks if ball is within goal bounds
            if (abs(self.ball.x - self.level.goal_distance) < epsilon) and self.level.goal_height <= self.ball.y <= self.level.goal_height + epsilon:
                self.ball.in_goal = True
                msg = 'Goal hit!'
                self.score += 1
                break
            # checks out of bounds
            elif (self.ball.x < left_bound) or (self.ball.x > right_bound) or (self.ball.y < lower_bound) or (self.ball.y > upper_bound):
                self.ball.reset(starting_pos)
                msg = 'Ball outside bounds.'
                break   
            # checks hazard
            elif self.ball.in_hazard:
                self.ball.reset(starting_pos)
                msg = 'Hazard encountered.'
                self.score += 1
                break
            # checks insufficient speed
            elif (abs(self.ball.velocity_x) < epsilon/2) and self.ball.rolling:
                # if wind in x direction is 0
                if self.level.wind*np.cos(np.deg2rad(self.level.angle)) == 0:
                    msg = 'Ball stopped due to insufficient speed.'
                    self.score += 1
                    break
                else:
                    # ball gets stuck
                    if len(set(hor[-5:-1])) == len(set(ver[-5:-1])) == 1:
                        msg = 'Ball stopped due to insufficient speed.'
                        self.score += 1
                        break
                    else:
                        pass
            # exceed time limit
            elif len(ver)*dt > 10:
                msg = 'In-game run time exceeded 10 seconds.'
                self.score += 1
                break 
        end_time = time.time()
        run_time = end_time - begin_time
        if self.score == self.max_score:
            msg = 'Maximum allowed score reached.'
        if self.score > self.max_score:
            msg = 'Game quit.'
        if return_trajectory:
            return hor, ver
        if plot_trajectory:
            self.render_trajectory(hor, ver, save_trajectory=save_trajectory)
        if manual:
            print(f'Impulse = {impulse}, Angle = {angle}')
            print(msg)
            print(f'Real run time = {run_time} s, In-game run time = {len(ver)*dt} s')
            print(f'Starting ball position: ({hor[0]}, {ver[0]}), \nFinal ball position: ({hor[-1]}, {ver[-1]})')
            print(f'Current score: {self.score}') 
            
    def get_state(self):
        """ gets current game state; returns list """
        ball_state = self.ball.get_state()
        level_state = self.level.get_state()
        # removes level bounds as they are constant
        obstacle_fitler = list(itertools.chain(*bounds.values()))
        obstacles = [obs for obs in self.level.obstacles if obs not in obstacle_fitler]
        # obstacle attrs list
        attrs_list = [obs.get_state() for obs in obstacles] 
        # null obstacles
        zeros_list = [[0 for j in range(obstacle_state_size)] for i in range(obstacle_limit - len(obstacles))]
        obstacle_state = attrs_list + zeros_list
        return [ball_state] + [level_state] + obstacle_state 
    
    @staticmethod
    def get_state_length():
        return len(Game(1).get_state())
    
    def is_done(self):
        """ checks if game is done; returns bool """
        return (self.score >= self.max_score) or (self.ball.in_goal)
    
    def render_game(self, save_render=False):
        """ 
        renders plot of current game state with ball position 
        
        args:
            save_render (bool): saves game image; default False
            
        """
        self.level.render_level(ball_position=(self.ball.x, self.ball.y), save_render=save_render)
      
    def render_trajectory(self, hors, vers, save_trajectory):
        """ 
        renders ball trajectory
        
        args:
            hors (list of floats): horizontal data
            vers (list of floats): vertical data
            save_trajectory (bool): saves trajectory path
            
        """
        fig, ax = self.level.render_level(title='Ball trajectory')
        ax.plot(hors, vers, linewidth=2, color='0.5') # grey color
        plt.show()
        plt.close()
        if save_trajectory:
            self.level.save_render(fig)
    
    def replay_episode(self, infos):
        """
        replays an episode given its transitions or tuple of action tuples
        
        args:
            infos (list of tuples or tuples of tuples): see golfenv.play_episode() documentation for transitions; 
                action tuples are tuples of tuples with form (impulse, action)
                
        """
        hors, vers = [], []
        self.reset()
        if type(infos[0]) == float:
            infos = [infos]
        elif type(infos[0]) == tuple:
            if len(infos[0]) > 2:
                infos = [actions[infos[i][1]] for i in range(len(infos))]
            elif len(infos[0]) == 2:
                infos = [infos[i] for i in range(len(infos))]
        for info in infos:
            hor, ver = self.hit_ball(*info, return_trajectory=True)
            hors = hors + hor
            vers = vers + ver
        self.render_trajectory(hors, vers)
    
    def validate_actions(self, action_values):
        """
        returns whether taking this action will return goal
        
        args:
            action_values (list of tuples): check agents.Agent.get_action_values() documentation
        
        returns list of tuples
        """
        for i, action in enumerate([a[0] for a in action_values]):
            self.reset()
            self.hit_ball(*action)
            action_values[i] = action_values[i] + (self.is_done(),)
        return action_values

class Ball:
    """ 
    ball class
    
    args: 
        x (float): x position of object
        y (float): y position of object
        mass (float): mass of object
        restitution (float): determines how much energy is lost during a bounce; it is a percent, so 0-100%
        rolling (bool): True if ball is rolling on a horizontal surface; default False
        in_hazard (bool): True if ball is ball is in a hazard; default False
        in_goal (bool): True if ball is in goal; default False
        
    attributes:
        x (float): x position of object
        y (float): y position of object
        mass (float): mass of object
        restitution (float): determines how much energy is lost during a bounce; it is a percent, so 0-100%
        rolling (bool): True if ball is rolling on a horizontal surface
        in_hazard (bool): True if ball is ball is in a hazard
        in_goal (bool): True if ball is in goal; default False
        velocity_x (float): velocity in x direction; default False
        velocity_y (float): velocity in y direction; default False
        
    """
    def __init__(self, x, y, mass, restitution, rolling=False, in_hazard=False, in_goal=False):       
        self.x = x
        self.y = y
        self.mass = mass
        self.restitution = restitution
        self.rolling = rolling
        self.in_hazard = in_hazard
        self.in_goal = in_goal
        self.velocity_x = 0
        self.velocity_y = 0
    
    def reset(self, reset_pos):
        """ 
        resets position of ball to position, sets hazard and goal status to False
        
        args:
            reset_pos (tuple of floats): x and y coordinate of reset position 
            
        """
        self.x, self.y = reset_pos
        self.in_hazard = False
        self.in_goal = False
    
    def hit(self, impulse, angle):
        """ 
        hits ball in certain direction with a given impulse at an angle
        
        args:
            impulse (float): impulse that generates a velocity (units of momentum)
            angle (float): the angle the ball is hit at with respect to horizontal
            
        """
        velocity = impulse/self.mass
        self.velocity_x = velocity*np.cos(np.deg2rad(angle))
        self.velocity_y = velocity*np.sin(np.deg2rad(angle))
        
    def move(self, level, current_time, dt):
        """ 
        moves object at every time step
        
        args:
            level (Level object): level that ball is moving in
            current_time (float): current time in simulation  
            dt (float): time step for each calculation
            
        """
        self.rolling = False
        fric = 0
        roll_orientation = None
        # checks every obstacle for collison
        for o in level.obstacles:
            # for horizontal obstacles; checks if vertical distance is less than epsilon
            # and if the horizontal component is within the bounds of the obstacle
            if o.orientation == 0 and abs(self.y - o.pos_y) <= epsilon and o.pos_x <= self.x <= o.pos_x + o.length:
                # checks if obstacle is a hazard and places ball in hazard
                if o.hazard:
                    self.in_hazard = True
                # if the ball doesn't have sufficient velocity to get off obstacle, sets ball to 
                # rolling mode
                elif abs(self.velocity_y) < epsilon:
                    self.rolling = True
                    roll_orientation = 0
                    # sets y position to obstacle's
                    self.y = abs(o.pos_y + epsilon/2)
                    self.velocity_y = 0
                    fric = o.mu
                # otherwise deflects ball; checks if ball is coming from correct direction
                # by checking if the sign of the velocity matches with the side the ball
                # approaches the obstacle from; this checks if the ball bounces from the top of the 
                # obstacle, in which case it uses the terrain type
                elif (self.y > o.pos_y) == (self.velocity_y < 0):
                    # velocity is dampened at proportionally to the obstacle's coeff of 
                    # friction and the ball's restitution; the velocity in the horizontal
                    # direction is damped less than the vertical direction -- the reason
                    # for this is purely heuristic after I bounced some balls outside 
                    self.velocity_x *= np.sqrt((1-o.mu)*self.restitution)
                    self.velocity_y *= -((1-o.mu)*self.restitution)
                # otherwise if the bounce comes from the bottom of the obstacle, it acts like
                # a level bound wall
                elif (self.y < o.pos_y) == (self.velocity_y > 0):
                    self.velocity_x *= np.sqrt(self.restitution)
                    self.velocity_y *= -self.restitution
            # logic follows from above 
            elif o.orientation == 1 and abs(self.x - o.pos_x) <= epsilon and o.pos_y <= self.y <= o.pos_y + o.length:
                if abs(self.velocity_x) < epsilon:
                    roll_orientation = 1
                    # lets ball roll on correct side
                    self.x = abs(o.pos_x - epsilon/2) if self.x < o.pos_x else abs(o.pos_x + epsilon/10)
                    self.velocity_x = 0
                elif (self.x < o.pos_x) == (self.velocity_x > 0):
                    self.velocity_x *= -self.restitution
                    self.velocity_y *= np.sqrt(self.restitution) 
        # if ball is stuck on vertical wall, ball doesn't move
        if roll_orientation == 1:
           self.velocity_x = 0  
        else:
            # wind and frictional force which points opposite of ball velocity
            self.velocity_x += (level.wind*np.cos(np.deg2rad(level.angle)) - np.sign(self.velocity_x)*fric*g*self.mass)*dt
        if roll_orientation == 0:
            self.velocity_y = 0
        else:
            # wind and gravitational force
            self.velocity_y += (-g + level.wind*np.sin(np.deg2rad(level.angle)))*dt
        self.x += self.velocity_x*dt
        self.y += self.velocity_y*dt
    
    def get_state(self):
        """ get ball state; returns list"""
        return [self.x, self.y, self.mass, self.restitution]
  
class Level:
    """
    level class
    
    args:
        level_number (int): level number
        goal_distance (float): where the goal is horizontally
        goal_height (float): height of goal
        obstacles (list of Obstacle objects): collection of Obstacle objects for this level
        description (str): description of level
        origin (tuple of floats): ball starts here; default lower left corner
        wind (float): wind force strength; default 0
        angle (float): angle of wind in level with respect to horizontal; default 0
        ball_spawn (tuple): area which ball can spawn; tuple specifies distance bounds; default None
        goal_spawn (tuple): area which flag can spawn; default None
        randomize (bool): determines if level randomizes flag and ball position; default False 
    
    attributes: 
        level_number (int): level number
        goal_distance (float): where the goal is horizontally
        goal_height (float): height of goal
        obstacles (list of Obstacle objects): collection of Obstacle objects for this level
        description (str): description of level
        origin (tuple of floats): ball starts here; default lower left corner
        wind (float): wind force strength; default 0
        angle (float): angle of wind in level with respect to horizontal; default 0
        ball_spawn (tuple): area which ball can spawn; tuple specifies distance bounds; default None
        goal_spawn (tuple): area which flag can spawn; default None
        randomize (bool): determines if level randomizes flag and ball position; default False 
        
    """
    def __init__(self, level_number, goal_distance, goal_height, obstacles, description, origin=corner, wind=0, angle=0, ball_spawn=None, goal_spawn=None, obstacle_spawn=None, randomize=False):
        self.level_number = level_number
        self.goal_distance = goal_distance
        self.goal_height = goal_height
        self.obstacles = obstacles
        self.description = description
        self.wind = wind
        self.angle = angle
        self.origin = origin
        # if spawn not specified, always spawns on given origin and goal position
        self.ball_spawn = ball_spawn if ball_spawn is not None else (origin[0], origin[0])
        self.goal_spawn = goal_spawn if goal_spawn is not None else (goal_distance, goal_distance)
        self.obstacle_spawn = obstacle_spawn
        self.randomize = randomize
        self.reset()
            
    def render_level(self, save_render=False, ball_position=None, title=None):
        """ 
        renders level with ball position
        figure is how plot displays in console, axis is plot's axes
        
        args: 
            save_render (bool): saves render of level; default False
            ball_position (tuple of floats): ball position; default None
            title (str): plot title; default None     
                
        returns tuple of matplotlib Figure object and matplotlib AxesSubplot object 
        """    
        if title is None:
            title = f'Level {self.level_number}: {self.description}'
        # actual bounds of level
        axis_length, axis_height = (left_bound, right_bound), (lower_bound, upper_bound)
        obstacle_linewidth = 2
        arrow_scale = 0.25
        fig, ax = plt.subplots(figsize=(figure_length, figure_height))
        ax.set_title(f'{title}')
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('height (m)')
        ax.set_xlim(*axis_length) 
        ax.set_ylim(*axis_height) 
        # plots obstacles
        for o in self.obstacles:
            if o.orientation == 0:
                ax.hlines(o.pos_y, o.pos_x, o.pos_x + o.length, linewidth=obstacle_linewidth, color=o.color)
                if o not in bounds['level_bounds']:
                    ax.hlines(o.pos_y-0.03, o.pos_x, o.pos_x + o.length, linewidth=obstacle_linewidth, color='black')
            if o.orientation == 1:
                ax.vlines(o.pos_x, o.pos_y, o.pos_y + o.length, linewidth=obstacle_linewidth, color=o.color) 
        ax.vlines(self.goal_distance, self.goal_height, self.goal_height + 0.5)
        ax.add_patch(patches.Rectangle((self.goal_distance, self.goal_height + 0.4), width=0.25, height=0.1, facecolor='red', edgecolor='black'))
        ball_x, ball_y = self.origin if ball_position is None else ball_position
        # arrow pointing at ball
        ax.arrow(ball_x, ball_y+2*arrow_scale, 0, -arrow_scale, length_includes_head=False, head_width=arrow_scale/2, 
                 head_length=arrow_scale/2)
        # arrow pointing wind
        if self.wind != 0:
            ax.arrow(9, 9, 2*arrow_scale*np.cos(np.deg2rad(self.angle)), 
                     2*arrow_scale*np.sin(np.deg2rad(self.angle)), length_includes_head=False, 
                     head_width=arrow_scale/2, head_length=arrow_scale/2, facecolor='grey', edgecolor='black')
        # saves level render
        if save_render:
            self.save_render(fig)
        return fig, ax
    
    def save_render(self, figure):
        """
        saves render of level
        
        args:
            figure (matplotlib Figure): figure to save
            
        """
        # lists renders
        dirs = os.listdir(os.path.join(os.getcwd(), 'levelrenders'))
        # sums number of renders of level in levelrenders
        render_count = sum(self.description in s for s in dirs) 
        # onyl subsequent renders are logged under numbers
        figure.savefig(f'levelrenders/{self.description + " " + str(render_count) if render_count != 0 else self.description} render')
    
    def get_state(self):
        """ get level state; return list """
        return [self.goal_distance, self.goal_height, self.wind, self.angle]
    
    def reset(self):
        """ reinitializes level """
        if self.randomize: 
            while True:
                # randomizes ball and goal position
                ball_distance, goal_distance = rd.uniform(*self.ball_spawn), rd.uniform(*self.goal_spawn)
                self.origin = (ball_distance, self.origin[1])
                self.goal_distance = goal_distance
                # filtering bounds
                obstacle_fitler = list(itertools.chain(*bounds.values()))
                # keep obstacles the same for each obstacle if no obstacle spawn
                if self.obstacle_spawn is not None:
                    obstacles = [obs for obs in self.obstacles if obs not in obstacle_fitler]
                    obstacle_distances = [rd.uniform(*self.obstacle_spawn) for i in range(len(obstacles))]
                    for i, obs in enumerate(self.obstacles):
                        # filtering bounds
                        if obs not in obstacle_fitler:
                            obs.pos_x = obstacle_distances[i]
                else:
                    obstacle_distances = [obs.pos_x for obs in self.obstacles if obs not in obstacle_fitler]      
                distances = [ball_distance, goal_distance, *obstacle_distances]
                # makes sure the two don't spawn on top of each other
                if not any([abs(i-j)<=epsilon for i, j in itertools.permutations(distances, 2)]):
                    break

class Obstacle:
    """
    obstacle class
    
    attributes:
        pox_x (float): x position of leftmost/bottom end
        pos_y (float): y position of leftmost/bottom end
        length (float): length of obstacle
        orientation (str): orientation of obstacle (0 or 1 for horizontal and vertical, respectively)
        mu (float): coefficient of friction; default 0
        color (str): color of obstacle; default black
        hazard (bool): True if obstacle is a hazard; default False
        
    """
    def __init__(self, pos_x, pos_y, length, orientation, mu=0, color='black', hazard=False):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = length
        self.orientation = orientation
        self.mu = mu
        self.color = color
        self.hazard = hazard
        
    def get_state(self):
        """ get obstacle state; return list"""
        return [self.pos_x, self.pos_y, self.length, self.orientation, self.mu, self.hazard]
    
## actions
impulse_options = [0.2, 0.35, 0.5]
angle_options = [5, 30, 60, 120, 150, 175]
actions = [(i, a) for i in impulse_options for a in angle_options]
num_available_actions = len(actions)
           
## balls
balls = {'golf': Ball(*corner, mass=0.05, restitution=0.68), 
         'tennis': Ball(*corner, mass=0.05, restitution=0.75), 
         'basketball': Ball(*corner, mass=0.625, restitution=0.88), 
         'bowling': Ball(*corner, mass=5, restitution=0.05)}

## obstacles
# obstacle unpack list
grass = [0, 0.4, '#56b000']
ice = [0, 0.05, '#a5f2f3']
sand = [0, 0.9999, '#c2b280']
water = [0, 0, '#0f5e9c', True]

# horizontal obstacles
hor_obstacles = {2: {1: Obstacle(3.5, 1.5, 3, *grass)}.values(), 
                 5: {1: Obstacle(3, 3, 2, *grass), 2: Obstacle(5, 2, 1, *grass),
                     3: Obstacle(6, 2, 0.5, *sand), 4: Obstacle(6.5, 2, 1, *grass),
                     5: Obstacle(7.5, 5, 2.5, *grass)}.values(),
                 7: {1: Obstacle(2, 8, 8, *ice)}.values(),
                 13: {1: Obstacle(3, 4, 4, *grass)}.values(),
                 18: {1: Obstacle(0, 3, 5, *ice)}.values(),
                 19: {1: Obstacle(0, 4, 6, *grass), 2: Obstacle(4, 2, 6, *grass)}.values(),
                 20: {1: Obstacle(0, 8, 4, *grass), 2: Obstacle(6, 6, 4, *grass),
                      3: Obstacle(0, 4, 4, *grass), 4: Obstacle(6, 2, 4, *grass)}.values()}

# vertical obstacles
# note the linspace lists are made to exclude the endpoints from being obstacles
ver_obstacles = {5: {1: Obstacle(2, lower_bound, 2, 1)}.values(),
                 9: [Obstacle(i, lower_bound, 2, 1) for i in np.linspace(left_bound, right_bound, 4+2)[1:-1]],
                 10: [Obstacle(i, lower_bound, 2, 1) for i in np.linspace(left_bound, right_bound, 10+2)[1:-1]],
                 16: [Obstacle(i, lower_bound, 2, 1) for i in np.linspace(left_bound, right_bound, 2+2)[1:-1]],
                 20: {1: Obstacle(6, lower_bound, 2, 1)}.values(),
                 21: {1: Obstacle(5, lower_bound, 2, 1)}.values(),
                 26: [Obstacle(i, lower_bound, 2, 1) for i in np.linspace(left_bound, right_bound, 3+2)[1:-1]]}

# ground obstacles and level bounds
bounds = {'ice': [Obstacle(left_bound, lower_bound, right_bound, *ice)],
          'grass': [Obstacle(left_bound, lower_bound, right_bound, *grass)],
          'sand': [Obstacle(left_bound, lower_bound, right_bound, *sand)],
          'water': [Obstacle(left_bound, lower_bound, right_bound, *water)],
          'grass_water': [Obstacle(left_bound, lower_bound, right_bound, *grass), Obstacle(2, lower_bound, 8, *water)],
          19: [Obstacle(left_bound, lower_bound, 2, *grass), Obstacle(2, lower_bound, 6, *water), 
               Obstacle(8, lower_bound, right_bound, *grass)],
          'level_bounds': [Obstacle(left_bound, lower_bound, upper_bound, 1), Obstacle(left_bound, upper_bound, right_bound, 0), 
                           Obstacle(right_bound, lower_bound, upper_bound, 1)]}
                           # [left boundary, upper boundary, right boundary]

## levels
lvls = {1: Level(**{'level_number': 1, 'description': 'Ice level', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*bounds['ice'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn}), 
        2: Level(**{'level_number': 2, 'description': 'Grass elevated flag', 'goal_distance': 5, 'goal_height': 1.5, 'obstacles': [*hor_obstacles[2], *bounds['grass'], *bounds['level_bounds']], 'ball_spawn': (epsilon, 2-epsilon)}),
        3: Level(**{'level_number': 3, 'description': 'Sand elevated flag', 'goal_distance': 5, 'goal_height': 1.5, 'obstacles': [*hor_obstacles[2], *bounds['sand'], *bounds['level_bounds']], 'ball_spawn': (epsilon, 2-epsilon)}),
        4: Level(**{'level_number': 4, 'description': 'Elevated water flag', 'goal_distance': 5, 'goal_height': 1.5, 'obstacles': [*hor_obstacles[2], *bounds['grass_water'], *bounds['level_bounds']], 'ball_spawn': (epsilon, 2-epsilon)}),
        5: Level(**{'level_number': 5, 'description': 'The chef\'s special', 'goal_distance': 9, 'goal_height': 5, 'obstacles': [*hor_obstacles[5], *ver_obstacles[5], *bounds['grass_water'], *bounds['level_bounds']], 'ball_spawn': (epsilon, 2-epsilon), 'goal_spawn': (8, 10-epsilon)}),
        6: Level(**{'level_number': 6, 'description': 'Windy ice level', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*bounds['ice'], *bounds['level_bounds']], 'wind': 2, 'angle': 45, 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn}),
        7: Level(**{'level_number': 7, 'description': 'Elevated ice ground', 'goal_distance': 9, 'goal_height': 8, 'obstacles': [*hor_obstacles[7], *bounds['grass'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': (5, 10-epsilon)}),
        8: Level(**{'level_number': 8, 'description': 'Grass level', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*bounds['grass'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn}),
        9: Level(**{'level_number': 9, 'description': 'Ice four barriers', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*ver_obstacles[9], *bounds['ice'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        10: Level(**{'level_number': 10, 'description': 'Ice ten barriers', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*ver_obstacles[10], *bounds['ice'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        11: Level(**{'level_number': 11, 'description': 'Grass four barriers', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*ver_obstacles[9], *bounds['grass'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        12: Level(**{'level_number': 12, 'description': 'Sand four barriers', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*ver_obstacles[9], *bounds['sand'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        13: Level(**{'level_number': 13, 'description': 'Very elevated water level', 'goal_distance': 5, 'goal_height': 4, 'obstacles': [*hor_obstacles[13], *bounds['grass_water'], *bounds['level_bounds']],}), 
        14: Level(**{'level_number': 14, 'description': 'Windy four barriers', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*ver_obstacles[9], *bounds['ice'], *bounds['level_bounds']], 'wind': 2, 'angle': 45, 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        15: Level(**{'level_number': 15, 'description': 'Windy ten barriers', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*ver_obstacles[10], *bounds['ice'], *bounds['level_bounds']], 'wind': 2, 'angle': 45, 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        16: Level(**{'level_number': 16, 'description': 'Ice two barriers', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*ver_obstacles[16], *bounds['ice'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        17: Level(**{'level_number': 17, 'description': 'Grass two barriers', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*ver_obstacles[16], *bounds['grass'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        18: Level(**{'level_number': 18, 'description': 'Two-story level', 'goal_distance': 1, 'goal_height': 3, 'obstacles': [*hor_obstacles[18], *bounds['grass'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': (epsilon, 2)}),
        19: Level(**{'level_number': 19, 'description': 'Pond level', 'goal_distance': 9, 'goal_height': 0, 'obstacles': [*hor_obstacles[19], *bounds[19], *bounds['level_bounds']], 'origin': (epsilon, 4+epsilon)}),
        20: Level(**{'level_number': 20, 'description': 'Three-story level', 'goal_distance': 1, 'goal_height': 8, 'obstacles': [*hor_obstacles[20], *ver_obstacles[20], *bounds['grass'], *bounds['level_bounds']], 'origin': (epsilon, 4+epsilon)}),
        21: Level(**{'level_number': 21, 'description': 'Ice one barrier', 'goal_distance': 7, 'goal_height': 0, 'obstacles': [*ver_obstacles[21], *bounds['ice'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        22: Level(**{'level_number': 22, 'description': 'Grass one barrier', 'goal_distance': 7, 'goal_height': 0, 'obstacles': [*ver_obstacles[21], *bounds['grass'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        23: Level(**{'level_number': 23, 'description': 'Windy ice one barrier', 'goal_distance': 7, 'goal_height': 0, 'obstacles': [*ver_obstacles[21], *bounds['ice'], *bounds['level_bounds']], 'wind': 2, 'angle': 45, 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        24: Level(**{'level_number': 24, 'description': 'Windy grass level', 'goal_distance': 5, 'goal_height': 0, 'obstacles': [*bounds['grass'], *bounds['level_bounds']], 'wind': 2, 'angle': 45, 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn}),
        25: Level(**{'level_number': 25, 'description': 'Windy grass one barrier', 'goal_distance': 7, 'goal_height': 0, 'obstacles': [*ver_obstacles[21], *bounds['ice'], *bounds['level_bounds']], 'wind': 2, 'angle': 45, 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn}),
        26: Level(**{'level_number': 26, 'description': 'Ice three barriers', 'goal_distance': 9, 'goal_height': 0, 'obstacles': [*ver_obstacles[26], *bounds['ice'], *bounds['level_bounds']], 'ball_spawn': ground_spawn, 'goal_spawn': ground_spawn, 'obstacle_spawn': ground_spawn})}

## dictionary for the par of each holes
pars = {None: list(lvls.keys()),
         1: [1, 2, 3, 4, 6, 8, 9, 11, 12, 13, 14, 16, 17, 18, 21],
         2: [5, 7, 10], 
         3: [19, 20]}

## sets state sizes
ball_state_size = len(balls['golf'].get_state())
level_state_size = len(lvls[1].get_state())
obstacle_state_size = len(bounds['sand'][0].get_state())

if __name__ == '__main__':
    pass
# %% save level renders  
if __name__ == '__main__': 
    for i in range(1, len(lvls)+1):
        lvls[i].render_level(save_render=True)
# %% brute forcer  
if __name__ == '__main__':   
    action_combinations = {1: [(actions[i]) for i in range(len(actions))],
                           2: [(actions[i], actions[j]) for i in range(len(actions)) for j in range(len(actions))],
                           3: [(actions[i], actions[j], actions[k]) for i in range(len(actions)) for j in range(len(actions)) for k in range(len(actions))]}
    # par investigating
    par = 1
    for lvl in [25]:
        print(f'Level {lvl}')
        game = Game(level=lvl)
        optimal_plays = []
        action_count = 0
        for action_combination in action_combinations[par]:
            if action_count % 100 == 0:
                print(f'Action {action_count+1}')
            game.reset()
            for i in range(par):
                if par == 1:
                    game.hit_ball(*action_combination)
                elif par > 1:
                    game.hit_ball(*action_combination[i])
            if game.is_done():
                optimal_plays.append(action_combination)
            action_count += 1
        print(f'Level {lvl}: \nHole-in-{par}: ' + ', '.join([str(p) for p in optimal_plays]))
