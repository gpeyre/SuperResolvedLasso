import numpy as np
import matplotlib.pyplot as plt

def prune(a,x,tol):
    idx = np.where(np.abs(a)>tol)[0]
    if len(x.shape)>1:
        x = x[idx,:]
    else:
        x = x[idx]
    return a[idx], x



def generate_plots(Plots, labels,filename,xlabel='',ylabel='', xaxis = [], color = (1,0,0)):
    p = len(Plots)
    for i in range(len(Plots)):
        E = Plots[i]
        if p>1:
            color = (i / (p - 1), 0, 1 - i / (p - 1))
        
        mean_curve = np.mean(E, axis=1)
        std_curve = np.std(E, axis=1)
        lower_bound = mean_curve - 1 * std_curve
        upper_bound = mean_curve + 1 * std_curve
        if len(xaxis)==0:
            xaxis = np.arange(len(mean_curve))
        plt.plot(xaxis, mean_curve, color=color,label=labels[i])
        plt.fill_between(xaxis, lower_bound, upper_bound, color=color, alpha=0.2)
    plt.legend(loc='best',fontsize=16)

    #plt.xticks([0,0.1,0.2])
    #plt.ylabel('Error', fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.savefig(filename, bbox_inches='tight',dpi=200, transparent=True)
    plt.show()
