import matplotlib 
# matplotlib.rcParams['text.usetex'] = True 
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import numpy as np

props = dict(boxstyle='square', facecolor='white', alpha=0.7) 
ftsze = 10
font = FontProperties()    
font.set_family('serif')    
font.set_name('Times New Roman')    
font.set_size(ftsze)

color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()
color3 = 'gray'

one_patch = mpatches.Patch(color=color1, label='ELM')
two_patch = mpatches.Patch(color=color2, label='AlexNet')
three_patch = mpatches.Patch(color=color3, label='MC-CNN')

elm_aucs = np.asarray([0.729, 0.9, 0.969, 0.573, 0.725, 0.692, 0.723, 0.815, 0.704, 0.838])
alex_net_aucs = np.asarray([0.848, 0.905, 0.975, 0.532, 0.833, 0.808, 0.836, 0.883, 0.827, 0.905])
mc_cnn_aucs = np.asarray([0.786, 0.932, 0.977, 0.6, 0.764, 0.737, 0.759, 0.857, 0.764, 0.876])

fig, axes = plt.subplots(1, 1)

x_a = np.r_[np.arange(1,11,1) - (0.25), 11.25]
x_b = np.r_[np.arange(1,11,1), 11.5]
x_c = np.r_[np.arange(1,11,1) + (0.25), 11.75]

axes.bar(x_a, np.r_[elm_aucs, elm_aucs.mean()], color=color1, label='ELM', width=0.25)
axes.bar(x_b, np.r_[alex_net_aucs, alex_net_aucs.mean()], color=color2, label='AlexNet', width=0.25)
axes.bar(x_c, np.r_[mc_cnn_aucs, mc_cnn_aucs.mean()], color=color3, label='MC-CNN', width=0.25)

axes.plot(np.arange(11.125,11.75, 0.25), np.ones(3) * 0.85, linewidth=1, color="black")
axes.plot(np.arange(11.125,12, 0.25), np.ones(4) * 0.875, linewidth=1, color="black")
axes.plot(np.arange(11.375,12, 0.25), np.ones(3) * 0.9, linewidth=1, color="black")

plt.axvline(11.125, 0.68, 0.7, color='black', lw=1)
plt.axvline(11.625, 0.68, 0.7, color='black', lw=1)
plt.axvline(11.125, 0.73, 0.75, color='black', lw=1)
plt.axvline(11.875, 0.73, 0.75, color='black', lw=1)
plt.axvline(11.375, 0.78, 0.8, color='black', lw=1)
plt.axvline(11.875, 0.78, 0.8, color='black', lw=1)

plt.text(11.3125, 0.8525, 'c', fontsize=ftsze)
plt.text(11.4375, 0.8775, 'b', fontsize=ftsze)
plt.text(11.55, 0.9025, 'a', fontsize=ftsze)

handles, labels = axes.get_legend_handles_labels()
axes.legend([handles[0], handles[1], handles[2]], [labels[0], labels[1], labels[2]], 
	fontsize=ftsze, loc='upper center', bbox_to_anchor=(0.65, 1), fancybox=True, shadow=True, ncol=3)   

# plt.legend(handles=handles, labels=labels, fontsize=ftsze, loc='upper center', bbox_to_anchor=(0, 1.0), fancybox=True, shadow=True, ncol=3) 
plt.xlabel('participant', fontsize=ftsze) 
plt.ylabel('AUC', fontproperties=font, fontsize=ftsze) 
plt.axvline(10.75, color='black')
plt.gca().set_xticks(np.r_[np.arange(1,11,1), 11.5])
plt.gca().set_xticklabels(['{}'.format(i) for i in range(1,11,1)] + ['average'])
plt.xlim([0.25,12.25]) 
plt.ylim([0.5, 1]) 
plt.gcf().set_size_inches(6.25, 5)    
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/ordered/scatter_csp_hil_new.png', dpi=300, bbox_inches='tight') 
plt.close('all') 