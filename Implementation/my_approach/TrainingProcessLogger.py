import os
import matplotlib.pyplot as plt
from config import config
import logging

class TrainingProcessLogger():
  
  def __init__(self, type, n_epochs, epoch_per_test):
    self.rewards = []
    self.losses = []
    self.mrrs = []
    self.hit3s = []
    self.hit5s = []
    self.hit10s = []
    self.n_epochs = n_epochs
    self.type = type
    self.epoch_per_test = epoch_per_test
   
    
  def log_loss_reward(self, loss, reward = None):  
    self.losses.append(loss)
    if reward != None:
      self.rewards.append(reward)
      
  def log_performance(self, mrr, hits):
    self.mrrs.append(mrr)
    self.hit3s.append(hits[0])    
    self.hit5s.append(hits[1])    
    self.hit10s.append(hits[2])    
  
    
  def create_figure(self, title, x, y, x_label, y_label, color):
    plt.figure()
    plt.plot(x, y, color=color)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xlim([0, self.n_epochs])
    plt.grid(True)  
    return plt

  def write_lists_to_file(self, log_dir, losses, mrrs, hits10s, hits5s, hits3s, rewards = None):  
    losses_str = "losses = [" + ",".join([str(element) for element in losses]) + "]" 
    rewards_str = "rewards = []" if rewards is None else  "rewards = [" + ",".join([str(element) for element in rewards]) + "]"    
    mrrs_str = "mrrs = []" if mrrs == None else "mrrs = ["  + ",".join([str(elem) for elem in mrrs]) + "]"   
    hits10s_str =   "hits10s = [" + ",".join([str(elem) for elem in hits10s]) + "]"   
    hits5s_str =   "hits5s = [" + ",".join([str(elem) for elem in hits5s]) + "]"   
    hits3s_str =   "hits3s = [" + ",".join([str(elem) for elem in hits3s])  + "]"   
    
    with open(os.path.join(log_dir, "logged_lists.txt"), "w") as text_file:
      text_file.write("%s\n%s\n%s\n%s\n%s\n%s" % (losses_str, rewards_str,  mrrs_str, hits10s_str, hits5s_str, hits3s_str))
  
    


  def create_and_save_figures(self, log_dir):
    graph_dir = os.path.join(log_dir, "graphs")
    os.mkdir(graph_dir) 
    logging.getLogger('matplotlib.font_manager').disabled = True
      
    # log all lists in a separate file -> recreate graphs afterward
    self.write_lists_to_file(log_dir, self.losses, self.mrrs, self.hit10s, self.hit5s, self.hit3s, self.rewards)
    
    filename = os.path.join(graph_dir, self.type + '_' + config().task.dir)
    n_points = [x for x in range(0, self.n_epochs+1, config().log.graph_every_nth_epoch)]
    if len(self.rewards) > 1:
      self.create_figure(config().task.dir.upper() + ' Rewards', n_points, self.rewards, 'Epochs', 'Rewards','blue').savefig(filename + '_rewards')
      
    self.create_figure(config().task.dir.upper() + ' Losses', n_points, self.losses, 'Epochs', 'Losses', 'red').savefig(filename + '_losses')
    n_points = [x for x in range(0, self.n_epochs+1, self.epoch_per_test)]
    print(n_points)
    self.create_figure(config().task.dir.upper() + ' Validation MRR', n_points, self.mrrs,'Epochs', 'MRR',  'green').savefig(filename + '_mrr')
    
    self.create_figure(config().task.dir.upper() + ' Validation H@3', n_points, self.hit3s, 'Epochs', 'H@3', 'orange').savefig(filename + '_hit3')
    self.create_figure(config().task.dir.upper() + ' Validation H@5', n_points, self.hit5s, 'Epochs', 'H@5', 'orange').savefig(filename + '_hit5')
    self.create_figure(config().task.dir.upper() + ' Validation H@10', n_points, self.hit10s, 'Epochs', 'H@10', 'orange').savefig(filename + '_hit10')
    

  # TODO: load plots from original KBGAN approach an compare them 
  #   -> for each model
  #   -> for each dataset
  # TODO: combine plots of my approach for each model and each dataset
    