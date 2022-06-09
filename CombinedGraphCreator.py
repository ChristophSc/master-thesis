import matplotlib.pyplot as plt
from os import path

class CombinedGraphCreator():
  def __init__(self, dataset, models, n_epochs, train_type, 
               pretrained = "", 
               sampling_type = "", 
               uncertainty_sampling_type = "",
               uncertainty_measure = ""):
    
    self.dataset = dataset    
    self.models = models # list with models, pretrain: list of models, gan_train: list of lists with models of generator and discriminator
    self.n_epochs = n_epochs  
    self.train_type = train_type
    
    # only for gan_train: 
    self.pretrained = pretrained
    self.sampling_type = sampling_type
    
    # only for uncertainty sampling
    self.uncertainty_sampling_type = uncertainty_sampling_type
    self.uncertainty_measure = uncertainty_measure
    
   
  def create_figure(self, title, logged_values, y_label):
    plt.figure()

    plt.title(title, fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    for model_name in logged_values.keys():      
        x =  50 if self.n_epochs == 5000 else 10 # int(self.n_epochs / (len(logged_values[model_name])-1))
        n_points = [x for x in range(0, self.n_epochs+1, x)]
        if len(logged_values[model_name]) > 1000:   # for rewards
          logged_list = [logged_values[model_name][i] for i in range(0, len(logged_values[model_name]), 10)]
        elif len(logged_values[model_name]) > 500:   # for rewards
           logged_list = [logged_values[model_name][i] for i in range(0, len(logged_values[model_name]), 5)]     
        else:
          logged_list = logged_values[model_name]
        plt.plot(n_points, logged_list, label = model_name)
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
      dir = path.join("logs", "backup")
      model_name = None
      if self.train_type == "pretrain":
        dir = path.join(dir, self.train_type,  self.dataset, models.lower())
        model_name = models # .lower()
      elif self.train_type == "gan_train":
        dir = path.join(dir, self.train_type, self.pretrained, self.sampling_type)
        if self.sampling_type == "uncertainty":
          dir = path.join(dir, self.uncertainty_sampling_type, self.uncertainty_measure)
        dir = path.join(dir, self.dataset, models[0].lower() +"_"+ models[1].lower())
        model_name = models[0] + " + " + models[1]
        
      mrrs, hits10s, rewards, losses = [], [], [], []      
      try:
        f = open(path.join(dir, 'logged_lists.txt'))
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
      dir = path.join('combined_figures', self.train_type, self.dataset)
      filename = self.train_type + '_' + self.dataset + '_'  
    elif self.train_type == "gan_train":   
      dir = path.join('combined_figures', self.train_type, self.pretrained, self.sampling_type)
      if self.sampling_type == "uncertainty":
        dir = path.join(dir, self.uncertainty_sampling_type, self.uncertainty_measure)
      dir = path.join(dir, self.dataset)
      filename = self.sampling_type.replace('/', '_') + '_' + self.dataset + '_' 

    
    
    self.create_figure(title = self.dataset.upper() + " - Validation MRR ", 
                      logged_values = logged_mrrs, 
                      y_label = "MRR").savefig(path.join(dir, filename + 'mrrs'))
    self.create_figure(title = self.dataset.upper() + " - Validation H@10 ", 
                      logged_values = logged_hits10s, 
                      y_label ="H@10").savefig(path.join(dir,filename + 'hit10'))
    self.create_figure(title = self.dataset.upper() + " - Training Losses ", 
                      logged_values = logged_losses, 
                      y_label ="Losses").savefig(path.join(dir, filename + 'losses'))
    
    if self.train_type != "pretrain":
      self.create_figure(title = self.dataset.upper() + " - Training Rewards ", 
                        logged_values = logged_rewards, 
                        y_label ="Rewards").savefig(path.join(dir,filename + 'rew'))
    print("Successfully created figures in " + dir)
  
  def create_compare_graph(self):
    ''' Compare best random sampling with best uncertainty sampling approach
    '''
    logged_mrrs, logged_hits10s, logged_rewards, logged_losses = dict(), dict(), dict(), dict()

    dir = path.join('combined_figures', 'compared', self.dataset, 'RandomVsUncertainty') + '_' 
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
    
    
  
  
  
  
datasets = ["umls"] # "umls", "kinship", "wn18", "wn18rr", "fb15k237",  "fb15k", "yago3_10"]
gen_models = ["DistMult", "ComplEx"]
dis_models = ["TransE", "TransD"]
all_models = gen_models + dis_models
pretraining_cases = ["not_pretrained"]  # "pretrained", "not_pretrained"
sampling_types = ["uncertainty", "random"]   #  ["uncertainty", "random"]
uncertainty_sampling_types = ["max", "max_distribution"] # "max_distribution"
uncertainty_measures = ["entropy"] # ["entropy", "least_confidence", "confidence_margin", "confidence_ratio"]

for dataset in datasets:
    CombinedGraphCreator(dataset = dataset, 
                          models = all_models, 
                         n_epochs= 1000,
                         train_type = "pretrain").create_combined_graph()
    model_pairs =[ [gen_model, dis_model] for gen_model in gen_models for dis_model in dis_models]
    
    for pretrained in pretraining_cases:      
      for sampling_type in sampling_types:
        for uncertainty_sampling_type in uncertainty_sampling_types:
          for uncertainty_measure in uncertainty_measures:
              CombinedGraphCreator(dataset = dataset, 
                                  models = model_pairs, 
                                  n_epochs = 5000,
                                  train_type = "gan_train", 
                                  pretrained = pretrained,
                                  sampling_type = sampling_type,
                                  uncertainty_sampling_type = uncertainty_sampling_type,
                                  uncertainty_measure = uncertainty_measure).create_combined_graph()