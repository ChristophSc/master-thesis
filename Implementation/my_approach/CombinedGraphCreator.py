from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CombinedGraphCreator():
  def __init__(self, dataset, models, n_epochs, train_type, sampling_type = ""):
    self.dataset = dataset
    self.models = models # list with models, pretrain: list of models, gan_train: list of lists with models of generator and discriminator
    self.train_type = train_type
    self.sampling_type = sampling_type
    self.n_epochs = n_epochs
    
   
  def create_figure(self, title, logged_values, y_label):
    plt.figure()

    plt.title(title, fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    for model_name in logged_values.keys():
      x =  int(self.n_epochs / (len(logged_values[model_name])-1))
      n_points = [x for x in range(0, self.n_epochs+1, x)]
      plt.plot(n_points, logged_values[model_name], label = model_name)
    plt.legend()
    
    plt.grid(True)  
    return plt
  
   
   
  def get_values(self, value_name, line):
    return list(map(float, line.replace(value_name + " = ", "").replace("\n", "").replace("[", "").replace("]", "").split(",")))   
   
  def create_combined_graph(self):
    # init arrays with logged values
    logged_mrrs, logged_hits10s, logged_rewards, logged_losses = dict(), dict(), dict(), dict()
    
    # read all files
    for i in range(len(self.models)):
      models = self.models[i]
      dir = "logs/backup/"
      model_name = None
      if self.train_type == "pretrain":
        dir += self.train_type + "/" +  self.dataset + "/" + models.lower() 
        model_name = models.lower()
      elif self.train_type == "gan_train":
        dir += self.train_type + "/" + self.sampling_type  + "/" + self.dataset + "/" + models[0].lower() +  "_" + models[1].lower()
        model_name = models[0].lower() + " + " + models[1].lower()
        
      mrrs, hits10s, rewards, losses = [], [], [], []      
      with open(dir + '/logged_lists.txt') as f:
        lines = f.readlines()
        for line in lines:          
          if line.startswith('rewards') and self.train_type == "gan_train":
            rewards = self.get_values("rewards", line)
          elif line.startswith('losses'):
            losses = self.get_values("losses", line)
          elif line.startswith('mrrs'):
            mrrs = self.get_values("mrrs", line)
          elif line.startswith('hits10s'):
            hits10s = self.get_values("hits10s", line)
             
        logged_mrrs[model_name] = mrrs
        logged_hits10s[model_name] = hits10s
        logged_rewards[model_name]  = rewards
        logged_losses[model_name] = losses   
         
    if self.train_type == "pretrain":
      dir = 'combined_figures/' + self.train_type + '/' + self.dataset + '/'
    elif self.train_type == "gan_train":    
      dir = 'combined_figures/' + self.train_type + '/' + self.sampling_type + '/' + self.dataset + '/'
    self.create_figure(title = self.train_type + " " + self.dataset.upper() + " Validation MRR " + self.sampling_type, 
                       logged_values = logged_mrrs, 
                       y_label = "MRR").savefig(dir + 'mrrs')
    self.create_figure(title = self.train_type + " " + self.dataset.upper() + " Validation H@10 " + self.sampling_type, 
                       logged_values = logged_hits10s, 
                       y_label ="H@10").savefig(dir + 'hit10s')   
    self.create_figure(title = self.train_type + " " + self.dataset.upper() + " Validation Losses " + self.sampling_type, 
                       logged_values = logged_losses, 
                       y_label ="Losses").savefig(dir + 'losses')
    
    if self.train_type != "pretrain":
      self.create_figure(title = self.train_type + " " + self.dataset.upper() + " Validation Rewards " + self.sampling_type, 
                        logged_values = logged_rewards, 
                        y_label ="Rewards").savefig(dir + 'rewards')
    
    
    
# cgc = CombinedGraphCreator(dataset = "umls", 
#                            models = ["DistMult", "ComplEx", "TransE", "TransD"], 
#                            n_epochs= 1000,
#                            train_type = "pretrain")

# cgc.create_combined_graph()



cgc = CombinedGraphCreator(dataset = "wn18rr", 
                           models = [["DistMult", "TransE"]], 
                           n_epochs= 5000,
                           train_type = "gan_train", 
                           sampling_type = "random")

cgc.create_combined_graph()