import sys  
from os import listdir
import multiprocessing

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models 
 
#import tensorflow_docs as tfdocs 
#import tensorflow_docs.modeling 
#import tensorflow_docs.plots 
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

import scipy  
from scipy import stats
from scipy import signal    
from scipy.fft import fftshift    
from scipy.ndimage import convolve1d, convolve    

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

import meet    
from meet.spatfilt import CSP

import numpy as np    
from numpy import save
from numpy import load

import pickle

import matplotlib.pyplot as plt    
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties 

# from __future__ import print_function  

 
## ### ### ### ###  
## Definition: utility-function / global vars  
## ### ### ### ###  
offset = 1000  
s_rate = 10000  
stim_per_sec = 4  
out_rej_thresh_fz = [0.45, 0.5, 0.225, 0.6, 0.6, 0.4, 0.45, 0.75, 0.45, 2]  
out_rej_thresh_mean = [0.6, 0.415, 0.12, 0.75, 0.3, 0.3, 0.45, 0.45, 0.3, 1.5]                                                                                      
                                      
## ### ### ### ###  
## Data loading  
## ### ### ### ###  
hfSEP_win = [50, 450]  
noise_win = [-500, -100]  
intrplt_win = [-80, 30] 
rand_stat = 42

def calculate_hil_features(transformed): 
   hil_dat = scipy.signal.hilbert(transformed, axis=0) 
   real_hil_dat = np.real(hil_dat) 
   imag_hil_dat = np.imag(hil_dat) 
   abs_hil_dat = np.abs(hil_dat) 
   angle_hil_dat = np.angle(hil_dat) 
   return np.concatenate((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0) 

def load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_path, noise_path, title, identifier):
# load data in format of (channel x epoch_length x number of epochs)
   title_of_run = title % (identifier)
   hfSEP = load(hfsep_path % (identifier))
   noise = load(noise_path % (identifier))

   ### 01_ToDo: Modify all the 11 base-modalities through: [SSD Y/N] (load respective modifiers therefore)
   # Still will remain open. Only thing to do here however: Load non-epoched data, compute SSD, epoch data
   raw_title = title_of_run + '_raw'

   ### 02_ToDo: Modify all the 22 modalities through: [leave, CSP, CCAr, bCSTP]
   # Compute CSP
   csp_title = title_of_run + '_CSP'
   csp_filters, csp_eigenvals = meet.spatfilt.CSP(hfSEP[:8,:,:].mean(2), noise[:8,:,:].mean(2))
#   hfSEP_CSP_0 = np.tensordot(csp_filters[0].T, hfSEP[:8,:,:], axes=(0 ,0))
#   noise_CSP_0 = np.tensordot(csp_filters[0].T, noise[:8,:,:], axes=(0 ,0))
#   hfSEP_CSP_1 = np.tensordot(csp_filters[1].T, hfSEP[:8,:,:], axes=(0 ,0))
#   noise_CSP_1 = np.tensordot(csp_filters[1].T, noise[:8,:,:], axes=(0 ,0))
#   hfSEP_CSP_2 = np.tensordot(csp_filters[2].T, hfSEP[:8,:,:], axes=(0 ,0))
#   noise_CSP_2 = np.tensordot(csp_filters[2].T, noise[:8,:,:], axes=(0 ,0))

   hfSEP_CSP_0 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_0.npy' % (identifier))
   noise_CSP_0 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_0.npy' % (identifier))
   hfSEP_CSP_1 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_1.npy' % (identifier))
   noise_CSP_1 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_1.npy' % (identifier))
   hfSEP_CSP_2 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_2.npy' % (identifier))
   noise_CSP_2 = np.load('/media/christoph/Volume/paper/prepped_data/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_2.npy' % (identifier))

   # Compute CCAr
   ccar_title = title_of_run + '_CCAr'
   a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(hfSEP[:8,:,:])
   ccar_filt_hfSEP_0 = np.tensordot(a_ccar[:,0], hfSEP[:8,:,:], axes=(0, 0))
   ccar_filt_noise_0 = np.tensordot(a_ccar[:,0], noise[:8,:,:], axes=(0, 0))
   ccar_filt_hfSEP_1 = np.tensordot(a_ccar[:,1], hfSEP[:8,:,:], axes=(0, 0))
   ccar_filt_noise_1 = np.tensordot(a_ccar[:,1], noise[:8,:,:], axes=(0, 0))
   ccar_filt_hfSEP_2 = np.tensordot(a_ccar[:,2], hfSEP[:8,:,:], axes=(0, 0))
   ccar_filt_noise_2 = np.tensordot(a_ccar[:,2], noise[:8,:,:], axes=(0, 0))

   # Compute bCSTP
   # s_bcstp_eigenvals, t_bcstp_eigenvals, W_bcstp, V_bcstp = bCSTP(hfSEP[:8,:,:], noise[:8,:,:], num_iter=15, t_keep=3, s_keep=3)
   # left out as it would also need intrplt.data.... the scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_kx_data_combined_hfsep[-1][:,0], intrplt_kx_data_combined[:8]), V_out_epoched_intrplt_kx_data_combined_hfsep[-1][:,0][::-1], axis=-1)

   ### 03_ToDo: Modify all the 88 modalities through: [hil Y/N]
   hil_csp_title = title_of_run + '_CSP_hil'
   hil_extracted_hfSEP_CSP_0 = calculate_hil_features(hfSEP_CSP_0)
   hil_extracted_noise_CSP_0 = calculate_hil_features(noise_CSP_0)
   hil_extracted_hfSEP_CSP_1 = calculate_hil_features(hfSEP_CSP_1)
   hil_extracted_noise_CSP_1 = calculate_hil_features(noise_CSP_1)
   hil_extracted_hfSEP_CSP_2 = calculate_hil_features(hfSEP_CSP_2)
   hil_extracted_noise_CSP_2 = calculate_hil_features(noise_CSP_2)
   hil_extracted_CSP_hfSEP = np.concatenate((hil_extracted_hfSEP_CSP_0, hil_extracted_hfSEP_CSP_1, hil_extracted_hfSEP_CSP_2), axis=0)
   hil_extracted_CSP_noise = np.concatenate((hil_extracted_noise_CSP_0, hil_extracted_noise_CSP_1, hil_extracted_noise_CSP_2), axis=0)

   hil_ccar_title = title_of_run + '_CCAR_hil'
   hil_extracted_ccar_filt_hfSEP_0 = calculate_hil_features(ccar_filt_hfSEP_0)
   hil_extracted_ccar_filt_noise_0 = calculate_hil_features(ccar_filt_noise_0)
   hil_extracted_ccar_filt_hfSEP_1 = calculate_hil_features(ccar_filt_hfSEP_1)
   hil_extracted_ccar_filt_noise_1 = calculate_hil_features(ccar_filt_noise_1)
   hil_extracted_ccar_filt_hfSEP_2 = calculate_hil_features(ccar_filt_hfSEP_2)
   hil_extracted_ccar_filt_noise_2 = calculate_hil_features(ccar_filt_noise_2)
   hil_extracted_ccar_hfSEP = np.concatenate((hil_extracted_ccar_filt_hfSEP_0, hil_extracted_ccar_filt_hfSEP_1, hil_extracted_ccar_filt_hfSEP_2), axis=0)
   hil_extracted_ccar_noise = np.concatenate((hil_extracted_ccar_filt_noise_0, hil_extracted_ccar_filt_noise_1, hil_extracted_ccar_filt_noise_2), axis=0)

   hfsep_labels = np.ones(len(hfSEP[0,0,:]), dtype=np.int8)
   noise_labels = np.zeros(len(noise[0,0,:]), dtype=np.int8)

   # return the datasets in epoch, channel, time_in_channel - fashion
   return [
      [np.concatenate((hfSEP[5]-hfSEP[0], noise[5]-noise[0]), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), raw_title],
      #   [np.concatenate((hfSEP.reshape(-1, hfSEP.shape[-1]), noise.reshape(-1, noise.shape[-1])), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), title_of_run + '_all_channels_flattened'],// I have no idea why, but this singular error still occurs
      [np.concatenate((hfSEP_CSP_0, noise_CSP_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), csp_title],
      [np.concatenate((ccar_filt_hfSEP_0, ccar_filt_noise_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
      [np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_csp_title],
      [np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
   ]

def load_model_and_predict(subject_id, sample_counter, modality_pair, rand_stat, true_labels_dict, predicted_labels_dict):
   """
   I am ... and therefore I ... I pass results back by means of a shared variable
   """   
   try:
      storage_place = modality_pair[-1]
      model = tf.keras.models.load_model(storage_place % (subject_id - 1, subject_id, modality_pair[0]))
      data_for_prediction = ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy', 'k00%d/epoched_intrplt_filt_500_900_kx']
      workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(data_for_prediction[0], data_for_prediction[1], data_for_prediction[2], subject_id)
      data, labels, run_title = workload[modality_pair[1]]
      print(run_title)
      print("Subject_id == %d" % subject_id)

      ### Shuffle and split data // .T is required to switch back to shape of (trial x feature) 
      shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat) 
      print(shuffled_data.shape) 
      X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat) 
      print(X_train.shape)
      print("Testing_data increase from %d to %d" % (sample_counter, sample_counter + X_test.shape[0]))
      sample_counter = sample_counter + X_test.shape[0]

      X_train = np.expand_dims(X_train, axis=-1)                                                                                                                         
      X_test = np.expand_dims(X_test, axis=-1)                                                                                                                           
      y_train = y_train.reshape((-1,1))                                                                                                                                  
      y_test = y_test.reshape((-1,1))
      y_pred = model.predict(X_test)

   except Exception as ex:
      template = "An exception of type {0} occurred. Arguments:\n{1!r}"
      message = template.format(type(ex).__name__, ex.args)
      print(message)
      print('The work for subject %d failed' % subject_id)
      if storage_place == pl_b:
         print(storage_place % (subject_id - 1, subject_id, modality_pair[0]))
      else:
         print(storage_place % (subject_id, subject_id, modality_pair[0]))
   finally:
      # make results back accessible again at the master or signal error code in array as: -1
      if ('y_test' in locals() or 'y_test' in globals()) and ('y_pred' in locals() or 'y_pred' in globals()):
         true_labels_dict[subject_id] = np.squeeze(y_test)
         predicted_labels_dict[subject_id] = np.squeeze(y_pred)
      else:
         true_labels_dict[subject_id] = [-1]
         predicted_labels_dict[subject_id] = [-1]


### ### ###
### mc-cnn ###
### ### ###
modalities_fpr_mccnn = []
modalities_tpr_mccnn = []
modalities_aucs_mccnn =  []
stds_fpr_mccnn = []
stds_tpr_mccnn = []
stds_aucs_mccnn = []
mean_aucs_mccnn = []
se_aucs_mccnn = []

# model_storage_places
pl_a_mccnn = '/media/christoph/Volume/Masterthesis/mc_cnn/models/%d/mc_cnn_on_k00%d-epoched_intrplt_filt_500_900_kx_%s'
pl_b_mccnn = '/media/christoph/Volume/paper/new_trainings/mc_cnn/models/%d_mc_cnn_on_k00%d-epoched_intrplt_filt_500_900_kx_%s'

for modality_pair in [['raw', 0, pl_b_mccnn], ['CSP', 1, pl_b_mccnn], ['CCAr', 2, pl_b_mccnn], ['CSP_hil', 3, pl_b_mccnn], ['CCAR_hil', 4, pl_b_mccnn]]:
   true_labels = np.asarray([])
   true_labels_per_subject = []
   predicted_labels = np.asarray([])
   predicted_labels_per_subject = []
   sample_counter = 0
   fpr_means = []
   tpr_means = []
   aucs = []

   manager = multiprocessing.Manager()
   true_labels_dict = manager.dict()
   predicted_labels_dict = manager.dict()

   for subject_id in range(1, 11, 1):
      # issues exist with this dataset...
#      if (modality_pair[0] == 'CCAR_hil') and (subject_id == 4):
#         pass
#      else:
      # the reason to manage the model loading and prediciton task by means of processes
      # and then not to execute them in parallel is only that I ran into following issue:
      # https://github.com/tensorflow/tensorflow/issues/36465 after ResourceExhaustedError
      # which occured only because of no proper cleanup of GPU-memory by tensorflow
      # call process and pass required arguments to process / then return the true and predicted labels
      # process calling; then do the concatenation using the proper dict_position
      p = multiprocessing.Process(
         target=load_model_and_predict, 
         args=(subject_id, sample_counter, modality_pair, rand_stat, true_labels_dict, predicted_labels_dict)
      )
      p.start()
      print('Started process w. %s and %s' % (modality_pair, subject_id))
      p.join()
      if len(true_labels_dict[subject_id]) != 1:
         y_test = true_labels_dict[subject_id]
         y_pred = predicted_labels_dict[subject_id]
         true_labels = np.concatenate((true_labels, true_labels_dict[subject_id]), axis=0)
         predicted_labels = np.concatenate((predicted_labels, predicted_labels_dict[subject_id]), axis=0)

      true_labels_per_subject.append(y_test)
      predicted_labels_per_subject.append(y_pred)
      fpr, tpr, thresholds = roc_curve(y_test, y_pred)
      model_auc = auc(fpr, tpr) 

      ## Perform switch of sign, such that the prediction is always better than chance-level
      if model_auc >= 0.5:
         fpr_means.append(np.asarray([x.mean() for x in np.array_split(fpr, 100)])) 
         tpr_means.append(np.asarray([x.mean() for x in np.array_split(tpr, 100)])) 
      else: 
         fpr_means.append(np.asarray([x.mean() for x in np.array_split(tpr, 100)])) 
         tpr_means.append(np.asarray([x.mean() for x in np.array_split(fpr, 100)])) 
         model_auc = auc(tpr, fpr)

      aucs.append(model_auc)
      print("AUC for this run: %f" % model_auc)

   mean_tpr = (np.asarray(tpr_means)).mean(0)
   mean_fpr = (np.asarray(fpr_means)).mean(0)
   mean_auc = (np.asarray(aucs)).mean(0)

   modalities_fpr_mccnn.append(mean_fpr)
   modalities_tpr_mccnn.append(mean_tpr)
   modalities_aucs_mccnn.append(mean_auc)
   stds_fpr_mccnn.append((np.asarray(fpr_means)).std(0))
   stds_tpr_mccnn.append((np.asarray(tpr_means)).std(0))
   stds_aucs_mccnn.append((np.asarray(aucs)).std(0))
   se_aucs_mccnn.append(scipy.stats.sem(aucs))

font = {'family' : 'sans-serif', 
        'weight' : 'light', 
        'size'   : 8}

plt.figure(1)   
plt.plot([0, 1], [0, 1], 'k--')   
base_accessor = 0  
colors = ['#590CF5', '#E8CEC7', '#E6011D', '#90E08F', '#F09523']  
hatches = ['x', 'O', '/', '\\', '+']  
for c, modality in enumerate(['raw', 'CSP', 'CCAr', 'CSP_hil', 'CCAr_hil']):  
   plt.plot(modalities_fpr_mccnn[c], modalities_tpr_mccnn[c], color=colors[c],
      label='%s (area) = %.3f' % (modality, modalities_aucs_mccnn[c]),
      path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], lw=2.5)  
plt.xlabel('False positive rate')   
plt.ylabel('True positive rate')   
plt.xlim([0,1])  
plt.ylim([0,1])  
plt.title('MC-CNN ROC curve over subjects after checking CSP')   
plt.legend(loc='best')  
plt.gcf().set_size_inches(5.5, 4.725)  
plt.rc('font', **font)  
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/new_after_reloading_csp_mc_cnn_roc_all_subjects_without_colorbar.png', dpi=400)  
plt.close('all')

plt.figure(1)   
plt.plot([0, 1], [0, 1], 'k--')   
base_accessor = 0  
colors = ['#590CF5', '#E8CEC7', '#E6011D', '#90E08F', '#F09523']
edgecol = ['#590CF530', '#E8CEC730', '#E6011D30', '#90E08F30', '#F0952330']
facecol = ['#590CF590', '#E8CEC790', '#E6011D90', '#90E08F90', '#F0952390']
# hatches = ['x', 'O', '/', '\\', '+']  
for c, modality in enumerate(['raw', 'CSP', 'CCAr', 'CSP_hil', 'CCAr_hil']):  
   plt.plot(modalities_fpr_mccnn[base_accessor + c], modalities_tpr_mccnn[base_accessor + c], color=colors[c],
      label='%s (area +/- std) = %.3f (+/-%.3f)' % (modality, modalities_aucs_mccnn[c], stds_aucs_mccnn[c]),
      path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], lw=2.5)  
   plt.fill_between(modalities_fpr_mccnn[base_accessor + c], modalities_tpr_mccnn[base_accessor + c] - stds_tpr_mccnn[base_accessor + c],
      modalities_tpr_mccnn[base_accessor + c] + stds_tpr_mccnn[base_accessor + c], facecolor=edgecol[c], edgecolor=facecol[c]) # hatch=hatches[c],    
plt.xlabel('False positive rate')   
plt.ylabel('True positive rate')   
plt.xlim([0,1])  
plt.ylim([0,1])  
plt.title('MC-CNN ROC curve over subjects after checking CSP')   
plt.legend(loc='best')  
plt.gcf().set_size_inches(5.5, 4.75)  
plt.rc('font', **font)  
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/new_after_reloading_csp_mc_cnn_roc_all_subjects_w_colorbar.png', dpi=400)  
plt.close('all') 


### ### ###
### alexnet ###
### ### ###
modalities_fpr_alex = []
modalities_tpr_alex = []
modalities_aucs_alex =  []
stds_fpr_alex = []
stds_tpr_alex = []
stds_aucs_alex = []
mean_aucs_alex = []
se_aucs_alex = []

# model_storage_places
pl_a_alex = '/media/christoph/Volume/Masterthesis/alex_net/models/%d/alex_net_on_k00%d-epoched_intrplt_filt_500_900_kx_%s'
pl_b_alex = '/media/christoph/Volume/paper/new_trainings/alexnet/models/%d_alex_net_on_k00%d-epoched_intrplt_filt_500_900_kx_%s'

for modality_pair in [['raw', 0, pl_b_alex], ['CSP', 1, pl_b_alex], ['CCAr', 2, pl_b_alex], ['CSP_hil', 3, pl_b_alex], ['CCAR_hil', 4, pl_b_alex]]:
   true_labels = np.asarray([])
   true_labels_per_subject = []
   predicted_labels = np.asarray([])
   predicted_labels_per_subject = []
   sample_counter = 0
   fpr_means = []
   tpr_means = []
   aucs = []

   manager = multiprocessing.Manager()
   true_labels_dict = manager.dict()
   predicted_labels_dict = manager.dict()

   for subject_id in range(1, 11, 1):
      # issues exist with this dataset...
      if (modality_pair[0] == 'CCAR_hil') and (subject_id == 4):
         pass
      else:
         # the reason to manage the model loading and prediciton task by means of processes
         # and then not to execute them in parallel is only that I ran into following issue:
         # https://github.com/tensorflow/tensorflow/issues/36465 after ResourceExhaustedError
         # which occured only because of no proper cleanup of GPU-memory by tensorflow
         # call process and pass required arguments to process / then return the true and predicted labels
         # process calling; then do the concatenation using the proper dict_position
         p = multiprocessing.Process(
            target=load_model_and_predict, 
            args=(subject_id, sample_counter, modality_pair, rand_stat, true_labels_dict, predicted_labels_dict)
         )
         p.start()
         p.join()
         if len(true_labels_dict[subject_id]) != 1:
            y_test = true_labels_dict[subject_id]
            y_pred = predicted_labels_dict[subject_id]
            true_labels = np.concatenate((true_labels, true_labels_dict[subject_id]), axis=0)
            predicted_labels = np.concatenate((predicted_labels, predicted_labels_dict[subject_id]), axis=0)

         true_labels_per_subject.append(y_test)
         predicted_labels_per_subject.append(y_pred)
         fpr, tpr, thresholds = roc_curve(y_test, y_pred)
         model_auc = auc(fpr, tpr) 

         ## Perform switch of sign, such that the prediction is always better than chance-level
         if model_auc >= 0.5:
            fpr_means.append(np.asarray([x.mean() for x in np.array_split(fpr, 100)])) 
            tpr_means.append(np.asarray([x.mean() for x in np.array_split(tpr, 100)])) 
         else: 
            fpr_means.append(np.asarray([x.mean() for x in np.array_split(tpr, 100)])) 
            tpr_means.append(np.asarray([x.mean() for x in np.array_split(fpr, 100)])) 
            model_auc = auc(tpr, fpr)

         aucs.append(model_auc)
         print("AUC for this run: %f" % model_auc)

   mean_tpr = (np.asarray(tpr_means)).mean(0)
   mean_fpr = (np.asarray(fpr_means)).mean(0)
   mean_auc = (np.asarray(aucs)).mean(0)

   modalities_fpr_alex.append(mean_fpr)
   modalities_tpr_alex.append(mean_tpr)
   modalities_aucs_alex.append(mean_auc)
   stds_fpr_alex.append((np.asarray(fpr_means)).std(0))
   stds_tpr_alex.append((np.asarray(tpr_means)).std(0))
   stds_aucs_alex.append((np.asarray(aucs)).std(0))
   se_aucs_alex.append(scipy.stats.sem(aucs))

font = {'family' : 'sans-serif', 
        'weight' : 'light', 
        'size'   : 8}

plt.figure(1)   
plt.plot([0, 1], [0, 1], 'k--')   
base_accessor = 0  
colors = ['#590CF5', '#E8CEC7', '#E6011D', '#90E08F', '#F09523']  
hatches = ['x', 'O', '/', '\\', '+']  
for c, modality in enumerate(['raw', 'CSP', 'CCAr', 'CSP_hil', 'CCAr_hil']):  
   plt.plot(modalities_fpr_alex[c], modalities_tpr_alex[c], color=colors[c],
      label='%s (area) = %.3f ' % (modality, modalities_aucs_alex[c]),
      path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], lw=2.5)  
plt.xlabel('False positive rate')   
plt.ylabel('True positive rate')   
plt.xlim([0,1])  
plt.ylim([0,1])  
plt.title('AlexNet ROC curve over subjects after checking CSP')   
plt.legend(loc='best')  
plt.gcf().set_size_inches(5.5, 4.75)  
plt.rc('font', **font)  
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/new_after_reloading_csp_alex_net_roc_all_subjects_without_colorbar.png', dpi=400)  
plt.close('all')

plt.figure(1)   
plt.plot([0, 1], [0, 1], 'k--')   
base_accessor = 0  
colors = ['#590CF5', '#E8CEC7', '#E6011D', '#90E08F', '#F09523']
edgecol = ['#590CF530', '#E8CEC730', '#E6011D30', '#90E08F30', '#F0952330']
facecol = ['#590CF590', '#E8CEC790', '#E6011D90', '#90E08F90', '#F0952390']
# hatches = ['x', 'O', '/', '\\', '+']  
for c, modality in enumerate(['raw', 'CSP', 'CCAr', 'CSP_hil', 'CCAr_hil']):  
   plt.plot(modalities_fpr_alex[base_accessor + c], modalities_tpr_alex[base_accessor + c], color=colors[c],
      label='%s (area +/- std) = %.3f (+/-%.3f)' % (modality, modalities_aucs_alex[c], stds_aucs_alex[c]),
      path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], lw=2.5)  
   plt.fill_between(modalities_fpr_alex[base_accessor + c], modalities_tpr_alex[base_accessor + c] - stds_tpr_alex[base_accessor + c],
      modalities_tpr_alex[base_accessor + c] + stds_tpr_alex[base_accessor + c], facecolor=edgecol[c], edgecolor=facecol[c]) # hatch=hatches[c],    
plt.xlabel('False positive rate')   
plt.ylabel('True positive rate')   
plt.xlim([0,1])  
plt.ylim([0,1])  
plt.title('AlexNet ROC curve over subjects after checking CSP')   
plt.legend(loc='best')  
plt.gcf().set_size_inches(5.5, 4.75)  
plt.rc('font', **font)  
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/new_after_reloading_csp_alex_net_roc_all_subjects_w_colorbar.png', dpi=400)  
plt.close('all') 


### ### ###
### elm ###
### ### ###
modalities_fpr_elm = []
modalities_tpr_elm = []
modalities_aucs_elm =  []
stds_fpr_elm = []
stds_tpr_elm = []
stds_aucs_elm = []
mean_aucs_elm = []
se_aucs_elm = []

# model_storage_places
pl_a_elm = '/media/christoph/Volume/Masterthesis/elm_models_trained/classes_balanced/'
pl_b_elm = '/media/christoph/Volume/paper/new_trainings/elm/models/'

for modality_pair in [['raw', 0, pl_b_elm], ['CSP', 1, pl_b_elm], ['CCAr', 2, pl_b_elm], ['CSP_hil', 3, pl_b_elm], ['CCAR_hil', 4, pl_b_elm]]:
   true_labels = np.asarray([])
   true_labels_per_subject = []
   predicted_labels = np.asarray([])
   predicted_labels_per_subject = []
   sample_counter = 0
   storage_place = modality_pair[-1]
   fpr_means = []
   tpr_means = []
   aucs = []

   for subject_id in range(1, 11, 1):
      print('loading model from: %s' % storage_place)
      elm_model_k00x = load('%selm_k00%d-epoched_intrplt_filt_500_900_kx_%s' % (storage_place, subject_id, modality_pair[0]), allow_pickle=True)
      data_for_prediction = ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy', 'k00%d/epoched_intrplt_filt_500_900_kx']
      workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(data_for_prediction[0], data_for_prediction[1], data_for_prediction[2], subject_id)
      data, labels, run_title = workload[modality_pair[1]]
      print(run_title)
      print("Subject_id == %d" % subject_id)
      print('Subject_data.shape: ')
      print(data.shape)
      ### Shuffle and split data // .T is required to switch back to shape of (trial x feature) 
      shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat) 
      print(shuffled_data.shape) 
      X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat) 
      print(X_train.shape)
      print("Testing_data increase from %d to %d" % (sample_counter, sample_counter + X_test.shape[0]))
      sample_counter = sample_counter + X_test.shape[0]
      y_pred_elmk00x = elm_model_k00x._run(X_test)
      true_labels = np.concatenate((true_labels, y_test), axis=0)
      true_labels_per_subject.append(y_test)
      predicted_labels = np.concatenate((predicted_labels, y_pred_elmk00x), axis=0)
      predicted_labels_per_subject.append(y_pred_elmk00x)
      fpr_elm_k00x, tpr_elm_k00x, thresholds_elm_k00x = roc_curve(y_test, y_pred_elmk00x)
      auc_elmk00x = auc(fpr_elm_k00x, tpr_elm_k00x) 

      ## Perform switch of sign, such that the prediction is always better than chance-level
      if auc_elmk00x >= 0.5:
         fpr_means.append(np.asarray([x.mean() for x in np.array_split(fpr_elm_k00x, 100)])) 
         tpr_means.append(np.asarray([x.mean() for x in np.array_split(tpr_elm_k00x, 100)])) 
      else: 
         fpr_means.append(np.asarray([x.mean() for x in np.array_split(tpr_elm_k00x, 100)])) 
         tpr_means.append(np.asarray([x.mean() for x in np.array_split(fpr_elm_k00x, 100)])) 
         auc_elmk00x = auc(tpr_elm_k00x, fpr_elm_k00x)

      aucs.append(auc_elmk00x)
      print("AUC for this run: %f" % auc_elmk00x)

   mean_elm_tpr = (np.asarray(tpr_means)).mean(0)
   mean_elm_fpr = (np.asarray(fpr_means)).mean(0)
   mean_auc = auc(mean_elm_fpr, mean_elm_tpr)

   modalities_fpr_elm.append(mean_elm_fpr)
   modalities_tpr_elm.append(mean_elm_tpr)
   modalities_aucs_elm.append(mean_auc)
   stds_fpr_elm.append((np.asarray(fpr_means)).std(0))
   stds_tpr_elm.append((np.asarray(tpr_means)).std(0))
   stds_aucs_elm.append((np.asarray(aucs)).std(0))
   se_aucs_elm.append(scipy.stats.sem(aucs))

# figure stuff
font = {'family' : 'sans-serif', 
        'weight' : 'light', 
        'size'   : 8}

plt.figure(1)   
plt.plot([0, 1], [0, 1], 'k--')   
base_accessor = 0
colors = ['#590CF5', '#E8CEC7', '#E6011D', '#90E08F', '#F09523']  
hatches = ['x', 'O', '/', '\\', '+']  
for c, modality in enumerate(['raw', 'CSP', 'CCAr', 'CSP_hil', 'CCAr_hil']):  
   plt.plot(modalities_fpr_elm[base_accessor + c], modalities_tpr_elm[base_accessor + c], color=colors[c],
      label='%s (area) = %.3f' % (modality, modalities_aucs_elm[c]),
      path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], lw=2.5)  
plt.xlabel('False positive rate')   
plt.ylabel('True positive rate')   
plt.xlim([0,1])  
plt.ylim([0,1])  
plt.title('ELM ROC curve over subjects after checking CSP')   
plt.legend(loc='best')  
plt.gcf().set_size_inches(5.5, 4.75)  
plt.rc('font', **font)  
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/after_reloading_csp_elm_roc_all_subjects_without_colorbar.png', dpi=400)  
plt.close('all')

plt.figure(1)   
plt.plot([0, 1], [0, 1], 'k--')   
base_accessor = 0  
colors = ['#590CF5', '#E8CEC7', '#E6011D', '#90E08F', '#F09523']
edgecol = ['#590CF530', '#E8CEC730', '#E6011D30', '#90E08F30', '#F0952330']
facecol = ['#590CF590', '#E8CEC790', '#E6011D90', '#90E08F90', '#F0952390']
# hatches = ['x', 'O', '/', '\\', '+']  
for c, modality in enumerate(['raw', 'CSP', 'CCAr', 'CSP_hil', 'CCAr_hil']):  
   plt.plot(modalities_fpr_elm[base_accessor + c], modalities_tpr_elm[base_accessor + c], color=colors[c],
      label='%s (area +/- std) = %.3f (+/-%.3f)' % (modality, modalities_aucs_elm[c], stds_aucs_elm[c]),
      path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], lw=2.5)  
   plt.fill_between(modalities_fpr_elm[base_accessor + c], modalities_tpr_elm[base_accessor + c] - stds_tpr_elm[base_accessor + c],
      modalities_tpr_elm[base_accessor + c] + stds_tpr_elm[base_accessor + c], facecolor=edgecol[c], edgecolor=facecol[c]) # hatch=hatches[c],    
plt.xlabel('False positive rate')   
plt.ylabel('True positive rate')   
plt.xlim([0,1])  
plt.ylim([0,1])  
plt.title('ELM ROC curve over subjects after checking CSP')   
plt.legend(loc='best')  
plt.gcf().set_size_inches(5.5, 4.75)  
plt.rc('font', **font)  
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/after_reloading_csp_elm_roc_all_subjects_w_colorbar.png', dpi=400)  
plt.close('all') 


### ### ###
### combined figure ###
### ### ###
font = {'family' : 'sans-serif', 
        'weight' : 'light', 
        'size'   : 10}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)
# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True)

for c, modality in enumerate(['raw', 'CSP', 'CCAr', 'CSP_hil', 'CCAr_hil']):  
   ### elm
   ax1.plot(modalities_fpr_elm[base_accessor + c], modalities_tpr_elm[base_accessor + c], color=colors[c],
      label='%s = %.2f (%.2f)' % (modality, modalities_aucs_elm[c], se_aucs_elm[c]),
      path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], lw=2)
   ### mccnn
   ax2.plot(modalities_fpr_mccnn[c], modalities_tpr_mccnn[c], color=colors[c],
      label='%s = %.2f (%.2f)' % (modality, modalities_aucs_mccnn[c], se_aucs_mccnn[c]),
      path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], lw=2)  
   ### alexnet
   ax3.plot(modalities_fpr_alex[c], modalities_tpr_alex[c], color=colors[c],
      label='%s = %.2f (%.2f)' % (modality, modalities_aucs_alex[c], se_aucs_alex[c]),
      path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], lw=2)  

for axee in (ax1, ax2, ax3):
   axee.plot([0, 1], [0, 1], 'k--')
   axee.legend(loc=4)
   axee.set_xlim([0,1])
   axee.set_ylim([0,1])  

plt.xlim([0,1])  
plt.ylim([0,1])  
ax1.set_title('ELM')
ax2.set_title('MC-CNN')
ax3.set_title('AlexNet')

ax1.set_xlabel('false positive rate')
ax2.set_xlabel('false positive rate')
ax3.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
# ax4.set_ylabel('True positive rate')
fig.set_size_inches(10, 3)  
plt.rc('font', **font) 

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 9

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.1, wspace = 0.1)
plt.margins(0.1,0.1)
#plt.gca().xaxis.set_major_locator(plt.NullLocator())
#plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_27_04/for_paper/ordered/fig_03_compound_newest.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
