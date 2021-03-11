import random as rd
import golf

class Env:
    """
    environment class
    
    args:
        levels (list of int): environment game levels
            
    attributes:
        game (golf.Game object): environment game
        levels (list of int): environment game levels
        randomize (bool): determines if environment randomizes flag and ball position in game
        
    """
    def __init__(self, levels, randomize=True):
        self.levels = levels
        self.randomize = randomize
        self.reset()
         
    def step(self, action):
        """ 
        takes action and updates environment 
        
        args:
            action (tuples of floats): impulse, angle for hitting ball; see golf.Ball documentation
        
        returns new environment state and reward as tuple
        """
        if action not in range(golf.num_available_actions):
            raise ValueError(f'Action {action} is not a valid action.')
        impulse, angle = golf.actions[action]
        self.game.hit_ball(impulse, angle)
        next_state = self.game.get_state()
        done = self.game.is_done()
        return next_state, -1, done, {}

    def reset(self):
        """ resets environment state; returns new environment state as tuple """
        level = rd.choice(self.levels)
        self.game = golf.Game(level, randomize=self.randomize)
        next_state = self.game.get_state()
        return next_state, 0, False, {}
    
class ReplayBuffer:
    """
    replay buffer object
    
    args:
        max_capacity (int or None): maximum amount of transitions allowed; default None
        
    """
    def __init__(self, max_capacity=None):
        self.replay_buffer = []
        self.max_capacity = max_capacity
        
    def __len__(self):
        """ length of replay buffer; returns int """
        return len(self.replay_buffer)
        
    def store_transitions(self, transitions):
        """ stores transitions in replay buffer, limits size to max_capacity """
        self.replay_buffer = self.replay_buffer + transitions
        if self.max_capacity != None:
            if len(self.replay_buffer) > self.max_capacity:
                self.replay_buffer = self.replay_buffer[:1]
    
    def return_buffer(self):
        """ gives complete replay buffer; returns numpy array of dimension (number of transitions, 5) """
        return self.replay_buffer
    
    def sample_buffer(self, batch_size):
        """ gives random sample of replay buffer; returns numpy array of dimension (batch_size, 5) """
        return rd.sample(self.replay_buffer, batch_size)
    
def play_episode(env, agent):
    """
    plays one game given environment and agent behavior
    
    args:
        env (Env object): enviroment to play episode in
        agent (Agent object): any object that returns an action from a game state
    
    returns tuple of transitions as list, reward as float
    """
    state, _, done, _ = env.reset()
    total_reward = 0
    transition_list = []
    while not done:
        begin_state = state
        action = agent.get_action(state,)
        state, reward, done, info = env.step(action)
        total_reward += reward
        transition_list.append((begin_state, action, reward, state, done))
    return transition_list, total_reward
