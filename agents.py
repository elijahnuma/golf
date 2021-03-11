import numpy as np
import random as rd
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import golf
import golfenv

save_path = 'Learning'

class Agent:
    """ 
    agent parent class
    
    args:
        replay_buffer (golfenv.ReplayBuffer object, takes int buffer_max_capacity as parameter; 
            default ReplayBuffer with max capacity of 100000)
        
    attributes:
        replay_buffer (golfenv.ReplayBuffer object, takes int buffer_max_capacity as parameter; 
            default ReplayBuffer with max capacity of 100000)
        
    """
    buffer_max_capacity = 100000
    
    def __init__(self, replay_buffer=golfenv.ReplayBuffer(max_capacity=buffer_max_capacity)):
        self.replay_buffer = replay_buffer
        
class RandomAgent(Agent):
    """ agent that acts randomly; subclass of Agent """
    def __init__(self):
        super().__init__()
    
    def get_action(self, state):
        """ 
        gets random action
        
        args:
            state (list): game state
        
        returns action as int
        """
        return rd.randint(0, golf.num_available_actions-1)
    
class DQNAgent(Agent):
    """
    deep learning agent; subclass of Agent
    
    args: 
        min_episode (int): episode at which epsilon reaches minimum
        architecture (str): neural net architecture 
        
    attributes:
        model (keras Model object): makes predictions for q values which are used to make an action; 
            default regular NN 
        model_target (keras Model object): prediction of future rewards; default clone of model
        architecture (str): neural net architecture 
        gamma (float): discount factor
        batch_size (int): batch size 
        max_epsilon (int): starting epsilon
        min_epsilon (int): minimum epsilon
        epsilon (float): epsilon factor that controls greedy actions
        episode_decay_rate (float): rate at which epsilon decays per episode
        
    """
    
    def __init__(self, min_episode, architecture):
        super().__init__()
        self.architecture = architecture
        self.model = self.create_q_model()
        self.model_target = keras.models.clone_model(self.model)
        self.gamma = 1
        self.batch_size = 50 
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.epsilon = self.max_epsilon
        self.epsilon_decay_rate = (self.max_epsilon-self.min_epsilon)/min_episode
    
    @property
    def state_size(self):
        """ state size of input layer for dense architecture; returns int """
        return np.array([ar.tolist() for ar in self.numpy_state(golf.Game(1).get_state())], dtype=object).size
     
    def create_q_model(self):
        """ creates neural network; returns keras Model object """
        # standard dense layer
        if self.architecture == 'dense':
            input_layer = layers.Input(shape=(self.state_size,),name='input')    
            hidden_layer = layers.Dense(500, activation='relu')(input_layer)
            hidden_layer = layers.Dense(500, activation='relu')(hidden_layer)
            hidden_layer = layers.Dense(500, activation='relu')(hidden_layer)
            final_layer = layers.Dense(golf.num_available_actions, activation='linear')(hidden_layer)
            model = keras.Model(inputs=input_layer, outputs=final_layer)
            model.compile(optimizer='adam', loss='mse')
            #shared_layer = layers.Dense(500, activation='relu')
            #hidden_layers = {f'hidden_layer{i}': shared_layer(obstacle_layers[f'obstacle_input{i}']) for i in range(1, golf.obstacle_limit+1)} 
        # shared layers
        elif self.architecture == 'shared':
            ball_layer = layers.Input(shape=(golf.ball_state_size,),name='ball_input')
            level_layer = layers.Input(shape=(golf.level_state_size,),name='level_input')
            obstacle_layers = {f'obstacle_input{i}': layers.Input(shape=(golf.obstacle_state_size,), name=f'obstacle_input{i}') for i in range(1, golf.obstacle_limit+1)}
            obstacle_set_layer = layers.Add()(obstacle_layers.values())  
            concat = layers.Concatenate()([ball_layer, level_layer, obstacle_set_layer])
            concat = layers.Dense(1000, activation='relu')(concat)
            concat = layers.Dense(1000, activation='relu')(concat)
            output = layers.Dense(golf.num_available_actions, activation='linear')(concat)
            model = keras.Model(inputs=[ball_layer, level_layer, *obstacle_layers.values()], outputs=output)
            model.compile(optimizer='adam', loss='mse')   
        return model
    
    def numpy_state(self, state):
        """
        changes state to readable array for NN

        args: 
            state (list): game state
                
        returns list
        """
        if self.architecture == 'dense':
            # completely flatten list
            flat = lambda *state: (e for a in state for e in (flat(*a) if isinstance(a, (tuple, list)) else (a,)))
            # returns array of batch_size 1
            state = list(flat(state))
            return np.reshape(np.array(state), (1, len(state)))
        elif self.architecture == 'shared':
            return [np.reshape(np.array(state[i], dtype=float), (1, len(state[i]))) for i in range(len(state))]
    
    def get_action(self, state):
        """ check RandomAgent documentation """
        state = self.numpy_state(state)
        self.epsilon = max(self.epsilon, self.min_epsilon)
        if self.epsilon > rd.random():
            action = rd.randint(0, golf.num_available_actions-1)
        else:
            q_values = self.model.predict(state)
            action = np.argmax(q_values)
        return action
    
    def update_epsilon(self):
        """ lowers epsilon """
        self.epsilon -= self.epsilon_decay_rate
        
    def train(self):
        """ sample from replay buffer and perform updates on the Q function; returns loss as float """
        batch = np.array(self.replay_buffer.sample_buffer(self.batch_size), dtype=object).T
        states, actions, rewards, next_states, dones = [batch[i] for i in range(batch.shape[0])]
        # groups inputs together into shape (batch_size, input_size)
        if self.architecture == 'dense':
            states = np.array([self.numpy_state(states[i])[0] for i in range(self.batch_size)])
            next_states = np.array([self.numpy_state(next_states[i])[0] for i in range(self.batch_size)])    
        if self.architecture == 'shared':
            states = [np.array([states[i][j] for i in range(self.batch_size)]) for j in range(self.state_size)]
            next_states = [np.array([next_states[i][j] for i in range(self.batch_size)]) for j in range(self.state_size)]
        target = self.model_target.predict(states)
        future_q_values = self.model_target.predict(next_states) 
        target_q_values = rewards + self.gamma*np.max(future_q_values, axis=1)*(1-dones)
        for i in range(len(target)):
            target[i][actions[i]] = target_q_values[i]
        loss = self.model.train_on_batch(states, target) 
        return loss
    
    def copy_weights(self):
        """ copies weights from model to target model """
        weights = self.model.get_weights()
        target_weights = self.model_target.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.model_target.set_weights(target_weights)
        
    def get_action_values(self, game, state=None):
        """ 
        grabs q-value for each action in a given state in given game
        
        args: 
            game (golf.Game object): game object
            state (tuple of floats): ball position; default None takes current state
        
        returns list of tuples
        """
        game_state = game.get_state()
        if state != None:
            game_state[0][:2] = state
        game_state = self.numpy_state(game_state)
        q_values = self.model.predict(game_state)
        pairs = [(golf.actions[i], q) for i, q in enumerate(q_values[0])]
        pairs = sorted(pairs, key=lambda tup:tup[1], reverse=True)
        return pairs 
    
    def save_agent(self, description):
        """
        saves agent model
        
        args:
            description (str): description
            
        """
        self.model.save(f'{save_path}/Model {description}.h5')
        # cannot pickle keras model, set to None
        self.model = self.model_target = None
        with open(f'{save_path}/Agent {description}', 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
    
    @staticmethod
    def load_agent(description):
        """
        loads agent model
        
        args:
            description (str): description
        
        returns Agent object
        """
        model = keras.models.load_model(f'{save_path}/Model {description}.h5')
        model_target = keras.models.clone_model(model)
        with open(f'{save_path}/Agent {description}', 'rb') as pickle_file:
            Agent = pickle.load(pickle_file) 
        Agent.model, Agent.model_target = model, model_target
        return Agent
# %%
if __name__ == '__main__':
    env = golfenv.Env(levels=[1])
    agent = DQNAgent(min_episode=20, architecture='shared')
    agent.batch_size = 20
    for i in range(15):
        transitions, episode_reward = golfenv.play_episode(env, agent)
        agent.replay_buffer.store_transitions(transitions)
        agent.update_epsilon()
    agent.train()
