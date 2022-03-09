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
      try:
        f = open(dir + '/logged_lists.txt')
      except IOError:
        print("Could not find 'logged_lists.txt' in " + dir)
        return
      else:
        with f:
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
      filename = self.train_type + '_' + self.dataset + '_'
      training_type = "GAN Training" 
    elif self.train_type == "gan_train":    
      dir = 'combined_figures/' + self.train_type + '/' + self.sampling_type + '/' + self.dataset + '/'
      filename = self.train_type + '_' + self.sampling_type.replace('/', '_') + '_' + self.dataset + '_' 
      training_type = "Pretraining" 
    
    self.create_figure(title = training_type + " - " + self.dataset.upper() + " - Validation MRR - " + self.sampling_type, 
                       logged_values = logged_mrrs, 
                       y_label = "MRR").savefig(dir + filename + 'mrrs')
    self.create_figure(title = training_type + " - " + self.dataset.upper() + " - Validation H@10 - " + self.sampling_type, 
                       logged_values = logged_hits10s, 
                       y_label ="H@10").savefig(dir + filename + 'hit10s')   
    self.create_figure(title = training_type + " - " + self.dataset.upper() + " - Training Losses - " + self.sampling_type, 
                       logged_values = logged_losses, 
                       y_label ="Losses").savefig(dir + filename + 'losses')
    
    if self.train_type != "pretrain":
      self.create_figure(title = training_type + " - " + self.dataset.upper() + " - Training Rewards - " + self.sampling_type, 
                        logged_values = logged_rewards, 
                        y_label ="Rewards").savefig(dir + filename + 'rewards')
    print("Successfully created figures in " + dir)
  
  def create_compare_graph(self):
    ''' Compare best random sampling with best uncertainty sampling approach
    '''
    logged_mrrs, logged_hits10s, logged_rewards, logged_losses = dict(), dict(), dict(), dict()

    dir = 'combined_figures/compared/' + self.dataset + '/' + 'RandomVsUncertainty' + '_' 
    filename = 'RandomVsUncertainty_' + self.dataset + '_' 
    self.create_figure(title = self.train_type + " - " + self.dataset.upper() + " - Random vs Uncertainty Sampling Validation MRR - " + self.sampling_type, 
                       logged_values = logged_mrrs, 
                       y_label = "MRR").savefig(dir + filename + 'mrrs')
    self.create_figure(title = self.train_type + " - " + self.dataset.upper() + " - Random vs Uncertainty Sampling Validation H@10 - " + self.sampling_type, 
                       logged_values = logged_hits10s, 
                       y_label ="H@10").savefig(dir + filename + 'hit10s')   
    self.create_figure(title = self.train_type + " - " + self.dataset.upper() + " - Random vs Uncertainty Sampling Training Losses - " + self.sampling_type, 
                       logged_values = logged_losses, 
                       y_label ="Losses").savefig(dir + filename + 'losses')
    self.create_figure(title = self.train_type + " - " + self.dataset.upper() + " - Random vs Uncertainty Sampling Training Rewards - ", 
                       logged_values = logged_rewards, 
                       y_label ="Rewards").savefig(dir + filename + 'rewards')
    
    
  
  
  
  
datasets = ["umls", "wn18rr", "wn18", "fb15k237"]
gen_models = ["DistMult", "ComplEx"]
dis_models = ["TransE", "TransD"]
all_models = gen_models + dis_models
sampling_types = ["random", "uncertainty/max/entropy"]


for dataset in datasets:
    CombinedGraphCreator(dataset = dataset, 
                         models = all_models, 
                         n_epochs= 1000,
                         train_type = "pretrain").create_combined_graph()
    model_pairs =[ [gen_model, dis_model] for gen_model in gen_models for dis_model in dis_models]
    
    for sampling_type in sampling_types:
      CombinedGraphCreator(dataset = dataset, 
                           models = model_pairs, 
                           n_epochs= 5000,
                           train_type = "gan_train", 
                           sampling_type = sampling_type).create_combined_graph()

    # compare best uncertainty vs random sampling approach in one graph for each dataset
    # CombinedGraphCreator(dataset = dataset, 
    #                        models = model_pairs, 
    #                        n_epochs= 5000,
    #                        train_type = "gan_train", 
    #                        sampling_type = sampling_type).create_compare_graph()