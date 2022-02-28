from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CombinedGraphCreator():
  def __init__(self, dataset, models, n_epochs, train_type, sampling_type = "", logs = []):
    self.dataset = dataset
    self.models = models # list with models, one model for pretraining and two models gan_train (for generator and discriminator)
    self.train_type = train_type
    self.sampling_type = sampling_type
    self.logs = logs
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
   
  def create_combined_graph(self):
    # init arrays with logged values
    logged_mrrs, logged_hits10s, logged_rewards, logged_losses = dict(), dict(), dict(), dict()
    
    # read all files
    for i in range(len(self.logs)):
      log = self.logs[i]
      models = self.models[i]
      dir = "logs/"
      model_name = None
      if self.train_type == "pretrain":
        dir += log + "_" + self.train_type + "_" +  models.lower()  + "_" + self.dataset 
        model_name = models.lower()
      elif self.train_type == "gan_train":
        dir += log + "_" + self.train_type + "_" + models[0].lower() +  "_" + models[1].lower()  + "_" + self.dataset 
        model_name = models[0].lower() + " + " + models[1].lower()
        
      mrrs, hits10s, rewards, losses = [], [], [], []      
      with open(dir + '/logged_lists.txt') as f:
        lines = f.readlines()
        for line in lines:          
          if line.startswith('rewards') and self.train_type == "gan_train":
            rewards = line.replace("rewards = ", "").replace("\n", "").replace("[", "").replace("]", "")
            rewards = rewards.split(",")            
            rewards = list(map(float, rewards))
          elif line.startswith('losses'):
            losses = line.replace("losses = ", "").replace("\n", "").replace("[", "").replace("]", "")
            losses = losses.split(",")            
            losses = list(map(float, losses))
          elif line.startswith('mrrs'):
            mrrs = line.replace("mrrs = ", "").replace("\n", "").replace("[", "").replace("]", "")
            mrrs = mrrs.split(",")            
            mrrs = list(map(float, mrrs))
          elif line.startswith('hits10s'):
            hits10s = line.replace("hits10s = ", "").replace("\n", "").replace("[", "").replace("]", "")
            hits10s = hits10s.split(",")            
            hits10s = list(map(float, hits10s))
             
        logged_mrrs[model_name] = mrrs
        logged_hits10s[model_name] = hits10s
        logged_rewards[model_name]  = rewards
        logged_losses[model_name] = losses   
             
    self.create_figure(title = self.train_type + " " + self.dataset.upper() + " Validation MRR " + self.sampling_type, 
                       logged_values = logged_mrrs, 
                       y_label = "MRR").savefig('combined_figures/' + self.train_type + '_' + self.dataset + '_combined_mrrs')
    self.create_figure(title = self.train_type + " " + self.dataset.upper() + " Validation H@10 " + self.sampling_type, 
                       logged_values = logged_hits10s, 
                       y_label ="H@10").savefig('combined_figures/' + self.train_type + '_' + self.dataset + '_combined_hit10s')   
    self.create_figure(title = self.train_type + " " + self.dataset.upper() + " Validation Losses " + self.sampling_type, 
                       logged_values = logged_losses, 
                       y_label ="Losses").savefig('combined_figures/' + self.train_type + '_' + self.dataset + '_combined_losses')
    
    if self.train_type != "pretrain":
      self.create_figure(title = self.train_type + " " + self.dataset.upper() + " Validation Rewards " + self.sampling_type, 
                        logged_values = logged_rewards, 
                        y_label ="Rewards").savefig('combined_figures/' + self.train_type + '_' + self.dataset + '_combined_rewards')
    
    
    
# cgc = CombinedGraphCreator(dataset = "wn18rr", 
#                            models = ["DistMult", "TransE"], 
#                            n_epochs= 1000,
#                            train_type = "pretrain", 
#                            logs = ["2022.02.28.132428", "2022.02.28.130247"])

# cgc.create_combined_graph()



cgc = CombinedGraphCreator(dataset = "wn18rr", 
                           models = [["DistMult", "TransE"]], 
                           n_epochs= 5000,
                           train_type = "gan_train", 
                           sampling_type = "Random Sampling",
                           logs = ["2022.02.23.121626"])

cgc.create_combined_graph()