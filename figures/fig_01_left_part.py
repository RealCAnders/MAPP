import matplotlib 
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
matplotlib.rcParams['text.usetex'] = True 
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import load
import meet
import scipy


def scale_to_same_rms_as_baseline(bipolar, ccar, csp): 
    rms_bipolar = np.sqrt(np.mean((bipolar ** 2), 0)).mean(0) 
    rms_ccar = np.sqrt(np.mean((ccar ** 2), 0)).mean(0) 
    rms_csp = np.sqrt(np.mean((csp ** 2), 0)).mean(0) 
     
    rms_ccar_factor = rms_bipolar / rms_ccar 
    rms_csp_factor = rms_bipolar / rms_csp 
     
    return (rms_ccar_factor, rms_csp_factor)


def get_indices_for_noise(triggers_to_get_indices_for):
  """
  General Idea:
  If we defined a window of 'safety', which we have to randomly place the noise-window in,
  then we can safely use random noise-window placements in this window.
  The window around a stimulus that we're interested in is 150ms long: 
  [<<-55ms Noise? -10ms>> <<-8ms Intrplt 2ms>> <<5ms hfSEP 45ms>> <<45ms Noise? 95ms>>]:= 150ms
  --> Follows the noise-window can start in [-55ms to -50ms] and [45ms to 55ms].
  """
  possible_neg_noise_starting_indices = np.arange(-550, -500, 1)
  possible_pos_starting_indices = np.arange(450, 550)
  possible_noise_indices_per_sample = np.append(possible_neg_noise_starting_indices, possible_pos_starting_indices)
  return np.asarray(triggers_to_get_indices_for) + np.random.choice(possible_noise_indices_per_sample, len(triggers_to_get_indices_for))


subject_id = 3
triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % subject_id, allow_pickle=True) 

# basic plot  
intrplt_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_kx_data_combined.npy' % subject_id, allow_pickle=True) 
intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % subject_id, allow_pickle=True)  
hfsep_around_artifact_500_900 = meet.epochEEG(intrplt_filt_500_900_kx, triggers_k3_combined, [0, 600])  
hfsep_around_artifact = meet.epochEEG(intrplt_kx, triggers_k3_combined, [0, 600])  
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)  
std_basic_prep_500_900 = np.std(hfsep_around_artifact_500_900[0,:,:]-hfsep_around_artifact_500_900[5,:,:], axis=1)  
scaling_factor = 100 
 
a_nb, b_nb, s_nb = meet.spatfilt.CCAvReg(hfsep_around_artifact_500_900[:8,150:350,:]) 
ccar_hfSEP = np.tensordot(a_nb[:,0], hfsep_around_artifact_500_900[:8,:,:], axes=(0,0)) 
std_ccar = np.std(ccar_hfSEP, axis=1) 
 
nb_filters, nb_eigenvals = meet.spatfilt.CSP(hfsep_around_artifact_500_900[:8,150:250,:].reshape(8, -1, order='F'), hfsep_around_artifact_500_900[:8,350:600,:].reshape(8, -1, order='F')) # ToDo: Understand // Make this with correct indices: meet.epochEEG(intrplt_filt_500_900_kx[:8,:350:600,:], get_indices_for_noise(triggers_k3_combined), [450, 750]) 
csp_hfSEP = np.tensordot(nb_filters[:,0].T, hfsep_around_artifact_500_900[:8], axes=(0,0))
std_csp = np.std(csp_hfSEP, axis=1) 

bipolar_hfsep_for_rms = ((hfsep_around_artifact_500_900[0,350:,:]-hfsep_around_artifact_500_900[5,350:,:]))
ccar_for_rms = ccar_hfSEP[350:,]
csp_for_rms = csp_hfSEP[350:,]
ccar_factor, csp_factor = scale_to_same_rms_as_baseline(bipolar_hfsep_for_rms, ccar_for_rms, csp_for_rms)
example_2 = (hfsep_around_artifact[0,:,:] - hfsep_around_artifact[5,:,:]).mean(1) * scaling_factor

fig, axes = plt.subplots(4, 1, sharex=True) 
slice_idx = 413 
props = dict(boxstyle='square', facecolor='white', alpha=0.7) 
ftsze = 10
font = FontProperties()    
font.set_family('serif')    
font.set_name('Times New Roman')    
font.set_size(ftsze)
ticks = np.arange(0, 700, 50)  
ticklabels = ['%d' % tick for tick in (ticks / 10)]  

axes[0].spines['top'].set_visible(False) 
axes[0].spines['bottom'].set_visible(False) 
axes[1].spines['top'].set_visible(False) 
axes[1].spines['bottom'].set_visible(False) 
axes[2].spines['top'].set_visible(False) 
axes[2].spines['bottom'].set_visible(False) 
axes[3].spines['top'].set_visible(False) 

color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

ln0 = axes[0].plot(np.arange(0, 600), example_2 - example_2[0], label='N=%d' % hfsep_around_artifact.shape[2], color='black', linewidth=1)  
# ln1 = axes[0].fill_between(np.arange(0, 600), -std_basic_prep * scaling_factor, std_basic_prep * scaling_factor, color='gray', label='+/- STD across single-trials', alpha=0.3) 
# ln3 = axes[0].plot(np.arange(0, 600), np.abs(scipy.signal.hilbert(((hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0]), axis=0)) * scaling_factor, label='Grand-Average amplitude', linewidth=1)  
 
ln00 = axes[1].plot(np.arange(0, 600), (hfsep_around_artifact_500_900[0,:,:]-hfsep_around_artifact_500_900[5,:,:]).mean(1) * scaling_factor, label='Averaged SEP', color='black', linewidth=1) 
ln10 = axes[1].fill_between(np.arange(0, 100), -std_basic_prep_500_900[:100] * scaling_factor, std_basic_prep_500_900[:100] * scaling_factor, color='gray', label='+/- STD Single-Trials') 
ln11 = axes[1].fill_between(np.arange(100, 350), -std_basic_prep_500_900[100:350] * scaling_factor, std_basic_prep_500_900[100:350] * scaling_factor, color=color1, label='+/- STD Single-Trials') 
axes[1].fill_between(np.arange(350, 450), -std_basic_prep_500_900[350:450] * scaling_factor, std_basic_prep_500_900[350:450] * scaling_factor, color='gray', label='+/- STD Single-Trials') 
ln12 = axes[1].fill_between(np.arange(450, 600), -std_basic_prep_500_900[450:600] * scaling_factor, std_basic_prep_500_900[450:600] * scaling_factor, color=color2, label='+/- STD Single-Trials') 
ln30 = axes[1].plot(np.arange(0, 600), np.abs(scipy.signal.hilbert(((hfsep_around_artifact_500_900[0,:,:]-hfsep_around_artifact_500_900[5,:,:])) * scaling_factor, axis=0)).mean(1), label='Averaged Amplitude (Analytic Signal)', linewidth=1, color='#e7298a')  

ln02 = axes[2].plot(np.arange(0, 600), ccar_hfSEP.mean(1) * ccar_factor * scaling_factor * -1, label='Averaged SEP', color='black', linewidth=1) 
ln02 = axes[2].fill_between(np.arange(0, 100), -std_ccar[:100] * ccar_factor * scaling_factor, std_ccar[:100] * ccar_factor * scaling_factor, color='gray', label='+/- STD Single-Trials') 
axes[2].fill_between(np.arange(100, 350), -std_ccar[100:350] * ccar_factor * scaling_factor, std_ccar[100:350] * ccar_factor * scaling_factor, color=color1, label='+/- STD Single-Trials') 
axes[2].fill_between(np.arange(350, 450), -std_ccar[350:450] * ccar_factor * scaling_factor, std_ccar[350:450] * ccar_factor * scaling_factor, color='gray', label='+/- STD Single-Trials') 
axes[2].fill_between(np.arange(450, 600), -std_ccar[450:600] * ccar_factor * scaling_factor, std_ccar[450:600] * ccar_factor * scaling_factor, color=color2, label='+/- STD Single-Trials') 
ln02 = axes[2].plot(np.arange(0, 600), np.abs(scipy.signal.hilbert(((ccar_hfSEP)), axis=0)).mean(1) * ccar_factor * scaling_factor, label='Averaged Amplitude (Analytic Signal)', linewidth=0.75, color='#e7298a') 

ln03 = axes[3].plot(np.arange(0, 600), csp_hfSEP.mean(1) * csp_factor * scaling_factor * -1, label='Averaged SEP', color='black', linewidth=1)
ln03 = axes[3].fill_between(np.arange(0, 100), -std_csp[:100] * csp_factor * scaling_factor, std_csp[:100] * csp_factor * scaling_factor, color='gray', label='+/- STD Single-Trials') 
axes[3].fill_between(np.arange(100, 350), -std_csp[100:350] * csp_factor * scaling_factor, std_csp[100:350] * csp_factor * scaling_factor, color=color1, label='+/- STD Single-Trials') 
axes[3].fill_between(np.arange(350, 450), -std_csp[350:450] * csp_factor * scaling_factor, std_csp[350:450] * csp_factor * scaling_factor, color='gray', label='+/- STD Single-Trials') 
axes[3].fill_between(np.arange(450, 600), -std_csp[450:600] * csp_factor * scaling_factor, std_csp[450:600] * csp_factor * scaling_factor, color=color2, label='+/- STD Single-Trials') 
ln03 = axes[3].plot(np.arange(0, 600), np.abs(scipy.signal.hilbert(((csp_hfSEP)), axis=0)).mean(1) * csp_factor * scaling_factor, label='Averaged Amplitude (Analytic Signal)', linewidth=0.75, color='#e7298a')

plt.xticks(ticks=ticks, labels=ticklabels)  
plt.xlim([0, 600])  
#-# plt.xscale('log')  
plt.xlabel('time relative to stimulus (ms)', fontproperties=font)  
#-#-#axes[0].set_ylabel('amplitude in nV', fontproperties=font, position=(1.5, -0.1))  
#-#-##axes[1].set_ylabel('amplitude in nV', fontproperties=font, position=(1.0, 0.5))  
#-#-#axes[2].set_ylabel('amplitude in a.u.', fontproperties=font, position=(2.25, -0.1)) 
#-#-##axes[3].set_ylabel('amplitude in a.u.', fontproperties=font, position=(1.0, 0.5)) 
axes[0].set_ylabel('ampl. (nV)', fontproperties=font, position=(1.5, 0.5))  
axes[1].set_ylabel('ampl. (nV)', fontproperties=font, position=(1.0, 0.5))  
axes[2].set_ylabel('ampl. (a.u.)', fontproperties=font, position=(2.25, 0.5)) 
axes[3].set_ylabel('ampl. (a.u.)', fontproperties=font, position=(1.0, 0.5)) 
 
axes[0].set_ylim([-750, 750]) 
axes[0].set_yticks([-750, 0, 750])
axes[0].set_yticklabels(['-750', '0', '750'])
axes[1].set_ylim([-75, 75]) 
axes[2].set_ylim([-75, 75]) 
axes[3].set_ylim([-75, 75]) 

class AnyObject(object):
  pass

class data_handler(object):
  def legend_artist(self, legend, orig_handle, fontsize, handlebox):
    x0, y0 = handlebox.xdescent, handlebox.ydescent
    width, height = handlebox.width, handlebox.height
    pa1 = mpatches.Rectangle([x0, y0], width/3, height, facecolor='gray', edgecolor='black', transform=handlebox.get_transform())
    pa2 = mpatches.Rectangle([x0 + width/3, y0], width/3, height, facecolor=color1, edgecolor='black', transform=handlebox.get_transform())
    pa3 = mpatches.Rectangle([x0 + (2*width/3), y0], width/3, height, facecolor=color2, edgecolor='black', transform=handlebox.get_transform())
    #-#-#        patch_sq = mpatches.Rectangle([x0, y0 + height/2 * (1 - scale) ], height * scale, height * scale, facecolor='0.5',
    #-#-#                edgecolor='0.5', transform=handlebox.get_transform())
    #-#-#        patch_circ = mpatches.Circle([x0 + width - height/2, y0 + height/2], height/2 * scale, facecolor='none',
    #-#-#                edgecolor='black', transform=handlebox.get_transform())

    handlebox.add_artist(pa1)
    handlebox.add_artist(pa2)
    handlebox.add_artist(pa3)
    return pa1

handles, labels = axes[1].get_legend_handles_labels() 
axes[0].legend([handles[0], handles[1], AnyObject()], [labels[0], labels[1], labels[2]], handler_map={AnyObject: data_handler()},
  fontsize=ftsze, loc='upper center', bbox_to_anchor=(0.495, 1.625), fancybox=True, shadow=True, ncol=3)   
 
#-#handles, labels = axes[1].get_legend_handles_labels() 
#-#axes[1].legend([handles[1], handles[2], handles[0]], [labels[1], labels[2], labels[0]], fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.12), fancybox=True, shadow=True, ncol=3)   
 
axes[0].tick_params(labelsize=ftsze)  
axes[1].tick_params(labelsize=ftsze)  
axes[2].tick_params(labelsize=ftsze) 
axes[3].tick_params(labelsize=ftsze) 
 
props = dict(boxstyle='square', facecolor='white', alpha=0.7) 

plt.text(-93, 655, 'A', fontsize=2*ftsze)
axes[0].text(0.10825, 1.100225, '0.5 Hz - 5000 Hz\nFz - CP5', transform=axes[0].transAxes, fontsize=ftsze, verticalalignment='top', ha='center', bbox=props) 
axes[1].text(0.1045, 1.1, '500 Hz - 900 Hz\nFz - CP5', transform=axes[1].transAxes, fontsize=ftsze, verticalalignment='top', ha='center', bbox=props) 
axes[2].text(0.1045, 1.1, '500 Hz - 900 Hz\nCCAr', transform=axes[2].transAxes, fontsize=ftsze, verticalalignment='top', ha='center', bbox=props) 
axes[3].text(0.1045, 1.1, '500 Hz - 900 Hz\nCSP', transform=axes[3].transAxes, fontsize=ftsze, verticalalignment='top', ha='center', bbox=props)
 
from matplotlib.patches import ConnectionPatch 
 
con2 = ConnectionPatch(xyA=(227, 250), coordsA=axes[0].transData, xyB=(227, 35), coordsB=axes[1].transData, color='gray', alpha=0.65, linestyle='dashed', linewidth=1) 
fig.add_artist(con2) 
con3 = ConnectionPatch(xyA=(242, 250), coordsA=axes[0].transData, xyB=(242, 35), coordsB=axes[1].transData, color='gray', alpha=0.65, linestyle='dashed', linewidth=1) 
fig.add_artist(con3) 
con4 = ConnectionPatch(xyA=(257.5, 0), coordsA=axes[0].transData, xyB=(257.5, 30), coordsB=axes[1].transData, color='gray', alpha=0.65, linestyle='dashed', linewidth=1) 
fig.add_artist(con4) 
 
con2a = ConnectionPatch(xyA=(227, -20), coordsA=axes[1].transData, xyB=(227, 42.5), coordsB=axes[2].transData, color='gray', alpha=0.65, linestyle='dashed', linewidth=1) 
fig.add_artist(con2a) 
con3a = ConnectionPatch(xyA=(242, -20), coordsA=axes[1].transData, xyB=(242, 42.5), coordsB=axes[2].transData, color='gray', alpha=0.65, linestyle='dashed', linewidth=1) 
fig.add_artist(con3a) 
con4a = ConnectionPatch(xyA=(257.5, -10), coordsA=axes[1].transData, xyB=(257.5, 37.5), coordsB=axes[2].transData, color='gray', alpha=0.65, linestyle='dashed', linewidth=1) 
fig.add_artist(con4a) 
 
con2b = ConnectionPatch(xyA=(227, -20), coordsA=axes[2].transData, xyB=(227, 42.5), coordsB=axes[3].transData, color='gray', alpha=0.65, linestyle='dashed', linewidth=1) 
fig.add_artist(con2b) 
con3b = ConnectionPatch(xyA=(242, -20), coordsA=axes[2].transData, xyB=(242, 42.5), coordsB=axes[3].transData, color='gray', alpha=0.65, linestyle='dashed', linewidth=1) 
fig.add_artist(con3b) 
con4b = ConnectionPatch(xyA=(257.5, -10), coordsA=axes[2].transData, xyB=(257.5, 37.5), coordsB=axes[3].transData, color='gray', alpha=0.65, linestyle='dashed', linewidth=1) 
fig.add_artist(con4b) 

axes[0].spines['top'].set_visible(False) 
axes[0].spines['bottom'].set_visible(False) 
axes[1].spines['top'].set_visible(False) 
axes[1].spines['bottom'].set_visible(False) 
axes[2].spines['top'].set_visible(False) 

axes[0].axes.xaxis.set_visible(False)
axes[1].axes.xaxis.set_visible(False)
axes[2].axes.xaxis.set_visible(False)

fig.set_size_inches(6.5, 5) 
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/ordered/N20_vs_hfSEP_paper_test_of_CSP_sensitivity_k003_newest_new.png', dpi=300)     
plt.close('all')