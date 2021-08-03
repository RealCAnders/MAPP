import matplotlib 
matplotlib.rcParams['text.usetex'] = True 
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
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

def calculate_hil_features(transformed):
	hil_dat = scipy.signal.hilbert(transformed, axis=0)
	real_hil_dat = np.real(hil_dat)
	imag_hil_dat = np.imag(hil_dat)
	abs_hil_dat = np.abs(hil_dat)
	angle_hil_dat = np.angle(hil_dat)
	return [real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat]

# hfsep & noise & hilbert-calculations
subject_id = 3
triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % subject_id, allow_pickle=True) 
intrplt_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_kx_data_combined.npy' % subject_id, allow_pickle=True) 
intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % subject_id, allow_pickle=True)  
hfsep_around_artifact_500_900 = meet.epochEEG(intrplt_filt_500_900_kx, triggers_k3_combined, [0, 450])   
a_nb, b_nb, s_nb = meet.spatfilt.CCAvReg(hfsep_around_artifact_500_900[:8,200:350,:]) 
ccar_hfsep = np.tensordot(a_nb[:,0], hfsep_around_artifact_500_900[:8,:,:], axes=(0,0))   
ticks_neg = np.arange(0, 900, 50)   
ticklabels_pos = ['%d' % tick for tick in (ticks_neg / 10)]   
noise_around_artifact_500_900 = meet.epochEEG(intrplt_filt_500_900_kx, triggers_k3_combined, [450, 850])   
ccar_noise = np.tensordot(a_nb[:,0], noise_around_artifact_500_900[:8,:,:], axes=(0,0))   

sta = 20
am_samp = 45
noise_real_hil_dat, noise_imag_hil_dat, noise_abs_hil_dat, noise_angle_hil_dat = calculate_hil_features(ccar_noise[:,sta:sta+am_samp])
hfsep_real_hil_dat, hfsep_imag_hil_dat, hfsep_abs_hil_dat, hfsep_angle_hil_dat = calculate_hil_features(ccar_hfsep[:,sta:sta+am_samp])

# fig_stuff 
fig, axes = plt.subplots(4, 1, sharex=True, sharey=True) 

props = dict(boxstyle='square', facecolor='white', alpha=0.7) 
ftsze = 10
font = FontProperties()    
font.set_family('serif')    
font.set_name('Times New Roman')    
font.set_size(ftsze)

# 100 - 350: hfSEP as input
# 450 - X: noise as input
# in between: black as unused

color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

axes[0].plot(np.arange(0, 100), hfsep_real_hil_dat[:100,:], label='Unused Signal', color='gray', linewidth=0.25)
axes[0].plot(np.arange(100, 350), hfsep_real_hil_dat[100:350,:], label='Signal for hfSEP', color=color1, linewidth=0.25)
axes[0].plot(np.arange(350, 450), hfsep_real_hil_dat[350:450,:], label='Unused Signal', color='gray', linewidth=0.25)
axes[0].plot(np.arange(450, 850), noise_real_hil_dat, label='Signal for noise', color=color2, linewidth=0.25)
axes[0].text(0.972, 1.065, 'real', transform=axes[0].transAxes, fontsize=ftsze, verticalalignment='top', ha='center', bbox=props)
# axes[0].text(0.0175, -0.2555, 'b)', transform=axes[0].transAxes, fontsize=ftsze*0.66, verticalalignment='top', ha='center', bbox=props)
axes[0].set_xticks(ticks=ticks_neg) 
axes[0].set_xticklabels(ticklabels_pos)   

axes[1].plot(np.arange(0, 100), hfsep_imag_hil_dat[:100,:], label='Unused Signal', color='gray', linewidth=0.25)
axes[1].plot(np.arange(100, 350), hfsep_imag_hil_dat[100:350,:], label='Signal for hfSEP', color=color1, linewidth=0.25)
axes[1].plot(np.arange(350, 450), hfsep_imag_hil_dat[350:450,:], label='Unused Signal', color='gray', linewidth=0.25)
axes[1].plot(np.arange(450, 850), noise_imag_hil_dat, label='Signal for noise', color=color2, linewidth=0.25)
axes[1].text(0.937, -0.10, 'imaginary', transform=axes[0].transAxes, fontsize=ftsze, verticalalignment='top', ha='center', bbox=props)
# axes[1].text(0.0175, -1.4580, 'c)', transform=axes[0].transAxes, fontsize=ftsze*0.66, verticalalignment='top', ha='center', bbox=props)

axes[2].plot(np.arange(0, 100), hfsep_abs_hil_dat[:100,:], label='Unused Signal', color='gray', linewidth=0.25)
axes[2].plot(np.arange(100, 350), hfsep_abs_hil_dat[100:350,:], label='Signal for hfSEP', color=color1, linewidth=0.25)
axes[2].plot(np.arange(350, 450), hfsep_abs_hil_dat[350:450,:], label='Unused Signal', color='gray', linewidth=0.25)
axes[2].plot(np.arange(450, 850), noise_abs_hil_dat, label='Signal for noise', color=color2, linewidth=0.25)
axes[2].text(0.936, -1.31, 'amplitude', transform=axes[0].transAxes, fontsize=ftsze, verticalalignment='top', ha='center', bbox=props)
# axes[2].text(0.0175, -2.6609, 'd)', transform=axes[0].transAxes, fontsize=ftsze*0.66, verticalalignment='top', ha='center', bbox=props)

axes[3].plot(np.arange(0, 100), hfsep_angle_hil_dat[:100,:], label='Unused Signal', color='gray', linewidth=0.25)
axes[3].plot(np.arange(100, 350), hfsep_angle_hil_dat[100:350,:], label='Signal for hfSEP', color=color1, linewidth=0.25)
axes[3].plot(np.arange(350, 450), hfsep_angle_hil_dat[350:450,:], label='Unused Signal', color='gray', linewidth=0.25)
axes[3].plot(np.arange(450, 850), noise_angle_hil_dat, label='Signal for noise', color=color2, linewidth=0.25)
axes[3].text(0.96, -2.5125, 'phase', transform=axes[0].transAxes, fontsize=ftsze, verticalalignment='top', ha='center', bbox=props)
# axes[3].text(0.0175, -3.86, 'e)', transform=axes[0].transAxes, fontsize=ftsze*0.66, verticalalignment='top', ha='center', bbox=props)

# fig_stuff 
axes[0].spines['top'].set_visible(False) 
axes[0].spines['bottom'].set_visible(False) 
axes[1].spines['top'].set_visible(False) 
axes[1].spines['bottom'].set_visible(False) 
axes[2].spines['top'].set_visible(False) 
axes[2].spines['bottom'].set_visible(False) 
axes[3].spines['top'].set_visible(False) 

axes[0].axes.xaxis.set_visible(False)
axes[1].axes.xaxis.set_visible(False)
axes[2].axes.xaxis.set_visible(False)

axes[3].set_xlabel('time relative to stimulus (ms)', fontproperties=font) 

blue_patch = mpatches.Patch(color=color1, label='Signal for hfSEP')
red_patch = mpatches.Patch(color=color2, label='Signal for random noise')

plt.legend(handles=[blue_patch, red_patch],
  fontsize=ftsze, loc='upper center', bbox_to_anchor=(0.495, 5.225), fancybox=True, shadow=True, ncol=2)  
plt.text(-88, 43.5, 'B', fontsize=2*ftsze)
plt.ylabel('feature-value (a.u.)', position=(1.0, 2.25), fontproperties=font)   
plt.ylim([-5, 5])
plt.xlim([0,600])
plt.tick_params(labelsize=ftsze) 
fig.set_size_inches(6.5, 5)   
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/ordered/butterfly_compound_newest_k003_right_side.png', dpi=300)