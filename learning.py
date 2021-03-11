import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import golf
import golfenv
import agents

lvls = [1, 21, 16, 26, 6, 23]                                       # individual levels
inclusive_lvls = [lvls[:i] for i in range(1, len(lvls)+1)][1:]      # combined levels
all_lvls = [[lvl] for lvl in lvls] + inclusive_lvls                 # above together
 
class LearningHandler:
    """
    learning handler class
    
    args: 
        architecture (str): neural net architecture
        min_episode (int): episode at which epsilon reaches minimum; default 50000
        train_lvls (list of int or None): levels that agent is trained on; default None
        test_lvl (list of int or None): level that agent tests training on; default None
        agent (str): agent of Handler description; default None
        description (str): description for level; default None 
        additional_description (str): additional info; default None
    
    attributes:
        agent (agents.Agent object): agent
        env (golfenv.Env object): environment 
        current_episode (int): current episode; starts at 1
        min_episode (int): episode at which epsilon reaches minimum; default 50000
        episode_rewards (list): total episode rewards; default empty
        running_reward (float): average of recent episode rewards
        loss_table (list): loss for training; default empty
        transition_list (list): stores each episode's complete transitions; default empty
        description (str): description for plots; default None
        is_done (bool): checks if training is done
        
    """
    def __init__(self, architecture, min_episode=50000, train_lvls=None, test_lvl=None, description=None, additional_description=None):
        description = 'Untrained' if train_lvls is None else self.make_description(train_lvls) if description is None else description
        # putting space at beginning
        additional_description = '' if additional_description is None else ' ' + additional_description
        self.description = description + additional_description
        # use untrained agent if either the test levels or the trained levels are not given
        self.agent = agents.DQNAgent(min_episode, architecture=architecture) if (train_lvls and test_lvl) is None else agents.DQNAgent.load_agent(self.description)
        self.env = golfenv.Env(test_lvl) if test_lvl is not None else golfenv.Env(train_lvls) if train_lvls is not None else golfenv.Env(levels=[1])
        self.current_episode = 1
        self.min_episode = min_episode
        self.episode_rewards = []
        self.loss_table = []
        self.transition_list = []
        self.is_done = False
    
    def train_episodes(self, num_episodes, stopping_reward=None):
        """
        runs multiple episodes
        
        args: 
            num_episodes (int): runs this number of episodes
            stopping_reward (float or None): stops training when running_reward equals this; 
                default None and does not stop
                
        """
        for _ in range(num_episodes):
            transitions, episode_reward = golfenv.play_episode(self.env, self.agent)
            self.agent.replay_buffer.store_transitions(transitions)
            self.episode_rewards.append(episode_reward)
            print(f'Episode {self.current_episode}')
            self.current_episode += 1
            self.agent.update_epsilon()
            if self.current_episode % 100 == 0:
                self.agent.copy_weights()
                self.transition_list.append(transitions)
        self.running_reward = np.mean(self.episode_rewards[-250:])
        print(f'Mean reward: {self.running_reward}')
        # calculates loss
        if len(self.agent.replay_buffer) > self.agent.batch_size:
            loss = self.agent.train()
            self.loss_table.append(loss)
            print(f'Loss = {loss}')
        # stopping conditions    
        if stopping_reward != None:
            if self.running_reward > stopping_reward:
                if (self.current_episode > self.min_episode/2) and (len(self.episode_rewards) > 100):
                    print(f'Solved at episode {self.current_episode}!')
                    self.is_done = True
        else:
            if self.current_episode > self.min_episode:
                print('Training complete.')
                self.is_done = True
    
    def play_episodes(self, num_episodes):
        # picks max Q-value action
        self.agent.epsilon = 0
        for _ in range(num_episodes):
            if self.current_episode % 100 == 0:
                print(f'Episode {self.current_episode}')
            _, episode_reward = golfenv.play_episode(self.env, self.agent)
            self.episode_rewards.append(episode_reward)
            self.current_episode += 1
        return np.mean(self.episode_rewards)
        
    def plot_tables(self, save_image=False):
        """ 
        plots loss and episode rewards 
        
        args:
            save_image (bool): saves image; default False
            
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(golf.figure_length, golf.figure_height))
        dot_size = 5
        titles = {'Running Loss': f'Running loss over episodes ({self.description})', 
                  'Running Rewards': f'Running rewards over episodes ({self.description})'}
        lt = self.loss_table
        running_loss = [np.mean(lt[:i+1]) if i<250 else np.mean(lt[i-250:i+1]) for i in range(len(lt))]
        er = self.episode_rewards
        running_rewards = [np.mean(er[:i+1]) if i<250 else np.mean(er[i-250:i+1]) for i in range(len(er))]
        # running loss plot
        ax1.set_title(titles['Running Loss'])
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Running Loss')
        ax1.scatter(np.arange(0,len(running_loss))*10, running_loss, s=dot_size)
        # running reward plot
        ax2.set_title(titles['Running Rewards'])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Running reward')
        ax2.scatter(range(len(running_rewards)), running_rewards, s=dot_size)
        # saves plot
        if save_image:
            fig.savefig(f'levelplots/{self.description}.png')
    
    def save_handler(self, description=None):
        """ 
        pickles LearningHandler object 
        
        args:
            description (str): description of Handler; default None
        
        """
        if description is None:
            description = self.description
        # saving model and model_target as it gets set to None 
        model, model_target = self.agent.model, self.agent.model_target
        self.agent.save_agent(f'{description}')
        with open(f'Learning/Handler {description}', 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
        self.agent.model, self.agent.model_target = model, model_target
    
    @staticmethod
    def load_handler(description):
        """ 
        loads pickled LearningHandler object 
        
        args:
            description (str): description of LearningHandler
        
        returns LearningHandler object
        """
        with open(f'Learning/Handler {description}', 'rb') as pickle_file:
            NewHandler = pickle.load(pickle_file) 
        NewHandler.agent = agents.DQNAgent.load_agent(f'{description}') 
        return NewHandler
    
    @staticmethod
    def make_description(levels):
        """
        makes readable description of levels
        
        args:
            levels (list of int): levels 
            
        """
        return ' , '.join([golf.Game(level).level.description for level in levels])
# %% Training
for lvl in [all_lvls[1]]:
    Handler = LearningHandler(architecture='shared', min_episode=50000, train_lvls=lvl, test_lvl=None, additional_description='Shared')
    while not Handler.is_done:
        Handler.train_episodes(num_episodes=10, stopping_reward=None)
    Handler.plot_tables(save_image=True)
    Handler.save_handler()
# %% Testing trained models
environments = {'Test level only': [[lvl] for lvl in lvls], 
                'Cumulative including test level': [lvls[:i] for i in range(1, len(lvls)+1)],
                'Cumulative excluding test level': [None] + [lvls[:i] for i in range(1, len(lvls))]}
df_dict = {key: [] for key in environments.keys()}
for environment in environments.keys():
    print(f'Environment: {environment}')
    for i, train_lvls in enumerate(environments[environment]):
        train_levels = train_lvls
        test_level = [lvls[i]]
        Handler = LearningHandler(architecture='shared', train_lvls=train_levels, test_lvl=test_level, additional_description='Shared')
        print(f'Train levels: {train_levels}, Test levels: {test_level}')
        mean_reward = Handler.play_episodes(500)
        print(f'Mean reward: {mean_reward}')
        df_dict[environment].append(mean_reward)
df = pd.DataFrame(data=df_dict, index=lvls)
df.to_csv('traineddf.csv', index=False)
# %% Random agent
total_rewards = []
for lvl in lvls:
    Handler = LearningHandler(min_episode=50, train_lvls=None, test_lvl=[lvl])
    print(f'Level {lvl}')
    total_rewards.append(Handler.play_episodes(500))
df = pd.DataFrame(data={'Untrained': total_rewards}, index=lvls)
df.to_csv('randomdf.csv', index=False)
# %% Save dataframe to clipboard
df.to_clipboard(index=False)
