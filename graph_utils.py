import pandas as pd
import matplotlib.pyplot as plt
   
   
   
def create_figure(title, x, y, x_label, y_label, color):
  plt.figure()
  plt.plot(x, y, color=color)
  plt.title(title, fontsize=14)
  plt.xlabel(x_label, fontsize=14)
  plt.ylabel(y_label, fontsize=14)
  plt.grid(True)  
  return plt

def plot_example():
  Data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010],
          'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
        }    
  df = pd.DataFrame(Data,columns=['Year','Unemployment_Rate'])    
  create_figure('Unemployment Rate Vs Year', df['Year'], df['Unemployment_Rate'], 'Year', 'Unemployment Rate', 'red')
