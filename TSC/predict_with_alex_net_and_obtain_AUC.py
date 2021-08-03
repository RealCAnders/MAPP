import sys  
import time
from os import listdir
import multiprocessing

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models 
 
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

import scipy  
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
from matplotlib.font_manager import FontProperties 

# starting time
start = time.time()
 
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
   # For the transformed data, calclulate respective Hilbert Transform and return
   # real, imaginary, absolute, and angle thereof
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

   raw_title = title_of_run + '_raw'

   # Compute CSP
   csp_title = title_of_run + '_CSP'
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

   # Compute CSP_hil
   hil_csp_title = title_of_run + '_CSP_hil'
   hil_extracted_hfSEP_CSP_0 = calculate_hil_features(hfSEP_CSP_0)
   hil_extracted_noise_CSP_0 = calculate_hil_features(noise_CSP_0)
   hil_extracted_hfSEP_CSP_1 = calculate_hil_features(hfSEP_CSP_1)
   hil_extracted_noise_CSP_1 = calculate_hil_features(noise_CSP_1)
   hil_extracted_hfSEP_CSP_2 = calculate_hil_features(hfSEP_CSP_2)
   hil_extracted_noise_CSP_2 = calculate_hil_features(noise_CSP_2)
   hil_extracted_CSP_hfSEP = np.concatenate((hil_extracted_hfSEP_CSP_0, hil_extracted_hfSEP_CSP_1, hil_extracted_hfSEP_CSP_2), axis=0)
   hil_extracted_CSP_noise = np.concatenate((hil_extracted_noise_CSP_0, hil_extracted_noise_CSP_1, hil_extracted_noise_CSP_2), axis=0)

   # Compute CCAr_hil
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
      [np.concatenate((hfSEP_CSP_0, noise_CSP_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), csp_title],
      [np.concatenate((ccar_filt_hfSEP_0, ccar_filt_noise_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
      [np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_csp_title],
      [np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
   ]


def load_model_and_predict(subject_id, sample_counter, modality_pair, rand_stat, true_labels_dict, predicted_labels_dict):
   """
   It was possible to run multiple predictions and models in parallel.
   So, as to speed up the TSC, multiprocessing was used. Utilizing shared
   variables, the results get passed from the worker to the master.
   """  
   try:
      storage_place = modality_pair[-1]
      if storage_place == pl_b:
         model = tf.keras.models.load_model(storage_place % (subject_id - 1, subject_id, modality_pair[0]))
      else:
         model = tf.keras.models.load_model(storage_place % (subject_id, subject_id, modality_pair[0]))
      alex_net = model
      data_for_prediction = ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy', 'k00%d/epoched_intrplt_filt_500_900_kx']
      workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(data_for_prediction[0], data_for_prediction[1], data_for_prediction[2], subject_id)
      data, labels, run_title = workload[modality_pair[1]]
      print(run_title)
      print("Subject_id == %d" % subject_id)

      ### Shuffle and split data // .T is required to switch back to shape of (trial x feature) 
      shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat) 
      print(shuffled_data.shape) 
      X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat) 
      print(X_test.shape)
      print("Testing_data increase from %d to %d" % (sample_counter, sample_counter + X_test.shape[0]))
      sample_counter = sample_counter + X_test.shape[0]

      X_train = np.expand_dims(X_train, axis=-1)                                                                                                                         
      X_test = np.expand_dims(X_test, axis=-1)                                                                                                                           
      y_train = y_train.reshape((-1,1))                                                                                                                                  
      y_test = y_test.reshape((-1,1))
      y_pred_alexnetk00x = alex_net.predict(X_test)
      
      fpr_alexnet, tpf_alexnet, thresholds_alexnet = roc_curve(y_test, y_pred_alexnetk00x)
      auc_alexnet = auc(fpr_alexnet, tpf_alexnet)
      print("AUC for this run: %f" % auc_alexnet)

   except Exception as ex:
      template = "An exception of type {0} occurred. Arguments:\n{1!r}"
      message = template.format(type(ex).__name__, ex.args)
      print(message)
      print('The work for subject %d failed' % subject_id)
   finally:
      # make results back accessible again at the master or signal error code -1
      if ('y_test' in locals() or 'y_test' in globals()) and ('y_pred_alexnetk00x' in locals() or 'y_pred_alexnetk00x' in globals()):
         true_labels_dict[subject_id] = np.squeeze(y_test)
         predicted_labels_dict[subject_id] = np.squeeze(y_pred_alexnetk00x)
      else:
         true_labels_dict[subject_id] = [-1]
         predicted_labels_dict[subject_id] = [-1]  



modalities_fpr = []
modalities_tpr = []
modalities_thresholds = []
modalities_aucs =  []

# model_storage_places
pl_a = '/media/christoph/Volume/Masterthesis/alex_net/models/%d/alex_net_on_k00%d-epoched_intrplt_filt_500_900_kx_%s'
pl_b = '/media/christoph/Volume/paper/new_trainings/alexnet/models/%d_alex_net_on_k00%d-epoched_intrplt_filt_500_900_kx_%s'

# for subject_id in range(4, 5, 1):
for subject_id in range(1, 11, 1):
   for modality_pair in [['raw', 0, pl_b], ['CSP', 1, pl_b], ['CCAr', 2, pl_b], ['CSP_hil', 3, pl_b], ['CCAR_hil', 4, pl_b]]:
      true_labels = np.asarray([])
      predicted_labels = np.asarray([])
      sample_counter = 0
      manager = multiprocessing.Manager()
      true_labels_dict = manager.dict()
      predicted_labels_dict = manager.dict()

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
      else: 
         print('Something went wrong! Ran into "len(true_labels_dict) == 1" for subject %d' % subject_id)
      fpr, tpr, thresholds = roc_curve(y_test, y_pred)
      model_auc = auc(fpr, tpr)
      
      ## Perform switch of sign, such that the prediction is always better than chance-level
      if model_auc >= 0.5:
         modalities_fpr.append(np.asarray([x.mean() for x in np.array_split(fpr, 100)])) 
         modalities_tpr.append(np.asarray([x.mean() for x in np.array_split(tpr, 100)])) 
      else: 
         modalities_fpr.append(np.asarray([x.mean() for x in np.array_split(tpr, 100)])) 
         modalities_tpr.append(np.asarray([x.mean() for x in np.array_split(fpr, 100)])) 
         model_auc = auc(tpr, fpr)

      modalities_thresholds.append(thresholds)
      modalities_aucs.append(model_auc)

   font = {'family' : 'sans-serif', 
     'weight' : 'light', 
     'size'   : 10}

   plt.figure(1) 
   plt.rc('font', **font)  
   plt.plot([0, 1], [0, 1], 'k--') 
   colors = ['#590CF5', '#E8CEC7', '#E6011D', '#90E08F', '#F09523']

   # for single-subject repetitions, base_accessor has to be 0 obviously
   base_accessor = (subject_id - 1) * 5

   available_modalities = ['raw', 'CSP', 'CCAr', 'CSP_hil', 'CCAR_hil']

   for c, modality in enumerate(available_modalities):
      plt.plot(modalities_fpr[base_accessor + c], modalities_tpr[base_accessor + c], color=colors[c], label='%s (area = %.3f)' % (modality, modalities_aucs[base_accessor + c]), lw=2.5)
   plt.xlabel('False positive rate') 
   plt.ylabel('True positive rate') 
   plt.title('AlexNet ROC curve for SO%d' % subject_id) 
   plt.legend(loc='best')
   plt.gcf().set_size_inches(5.5, 4.75)   
   plt.xlim([0,1])  
   plt.ylim([0,1])  
   plt.savefig('/home/christoph/Desktop/paper_presentation/temp/meeting_20_04/alex_net_roc_s%d.png' % subject_id, dpi=200)
   plt.close('all')

# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")