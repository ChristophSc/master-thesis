import os
import matplotlib.pyplot as plt
from config import config

class TrainingProcessLogger():
  
  def __init__(self, type, n_epochs, epoch_per_test):
    self.rewards = []
    self.losses = []
    self.mrrs = []
    self.hit10s = []
    self.n_epochs = n_epochs
    self.type = type
    self.epoch_per_test = epoch_per_test
   
    
  def log_loss_reward(self, epoch, loss, reward = None):
    if epoch % config().log.graph_every_nth_epoch == 0:   
      self.losses.append(loss)
      if reward:
        self.rewards.append(reward.item())
      
  def log_performance(self, mrr, hit10):
    self.mrrs.append(mrr)
    self.hit10s.append(hit10)    
  
    
  def create_figure(self, title, x, y, x_label, y_label, color):
    plt.figure()
    plt.plot(x, y, color=color)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True)  
    return plt

  def create_and_save_figures(self, log_dir):
    graph_dir = os.path.join(log_dir, "graphs")
    os.mkdir(graph_dir) 
      
    filename = os.path.join(graph_dir, self.type + '_' + config().task.dir)
    n_points = int(self.n_epochs/config().log.graph_every_nth_epoch)
    if len(self.rewards) > 0:
      self.create_figure(config().task.dir.upper() + ' Rewards', [x+1 for x in range(n_points)], self.rewards, 'Epochs', 'Rewards','blue').savefig(filename + '_rewards')
      
    self.create_figure(config().task.dir.upper() + ' Losses', [x+1 for x in range(n_points)], self.losses, 'Epochs', 'Losses', 'red').savefig(filename + '_losses')
    self.create_figure(config().task.dir.upper() + ' Validation MRR', [x+1 for x in range(self.n_epochs) if (x + 1) % self.epoch_per_test == 0], self.mrrs,'Epochs', 'MRR',  'green').savefig(filename + '_mrr')
    self.create_figure(config().task.dir.upper() + ' Validation H@10', [x+1 for x in range(self.n_epochs) if (x + 1) % self.epoch_per_test == 0], self.hit10s, 'Epochs', 'H@10', 'orange').savefig(filename + '_hit10')
    

  # TODO: load plots from original KBGAN approach an compare them 
  #   -> for each model
  #   -> for each dataset
  # TODO: combine plots of my approach for each model and each dataset
    