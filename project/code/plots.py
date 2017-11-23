#************************************************************
# module: plots - plot helper methods
#***********************************************************

#Loading libraries
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['figure.figsize'] = (10.0, 8.0)

#****************************************************************
# method: bar_plot(df, x, y, save_as = '')
# purpose: 
#*****************************************************************
def bar_plot(df, x, y, save_as = ''):

    sns.set(style="whitegrid", color_codes=True)
    sns.barplot(x = x, y = y, data=df)
    plt.xticks(rotation = 90)
   
    save_plot(plt, save_as)
    plt.show()
    
#end bar_plot

#****************************************************************
# method: histogram_plot(data)
# purpose: 
#*****************************************************************
def histogram_plot(data):
    print('hello')
#end histogram_plot

#****************************************************************
# method: scatter_plot(data)
# purpose: 
#*****************************************************************
def scatter_plot(data):
    print('hello')
#end scatter_plot

#****************************************************************
# method: correlation_plots(data)
# purpose: 
#*****************************************************************
def correlation_plots(data):
    print('hello')
#end correlation_plots

#****************************************************************
# method: box_plot(data)
# purpose: 
#*****************************************************************
def box_plot(x, y, **args):

    sns.boxplot(x=x,y=y)
    x = plt.xticks(rotation=90)

#end box_plot

#****************************************************************
# method: render_plots(data)
# purpose: 
#*****************************************************************
def render_plots(data, plotType, x, y):

    #p = pd.melt(data, id_vars='SalePrice', value_vars=cat)
    #g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
    #g = g.map(boxplot, 'value','SalePrice')
    print('helllo')
#end render_plots 

#****************************************************************
# method: save_as(plot, file_name
# purpose: 
#*****************************************************************
def save_plot(plot, file_name, height=300, width=200):
    
    #to avoid labels being chopped
    plot.tight_layout()
    
    if (file_name != ''):
        
        save_as = '../images/' + file_name + '.pdf'
        
        print('..saving plot as ', save_as)
        
        plot.savefig(save_as)
    #end if

#end save_as 
