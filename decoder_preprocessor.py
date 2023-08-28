import numpy as np
import pandas as pd
import random

class preprocess_single_trial_data:
    def __init__(self):
        self.pmap = {
            'sf': [],
            'ori': [],
            'phi': [],
            'id': []
        }
        
    def reset_pmap(self):
        self.pmap = {
            'sf': [],
            'ori': [],
            'phi': [],
            'id': []
        }
    
    def expand_pmap(self, row, n_trials):
        """
        Generate an ordered list of categorical variables
        corresponding to the order of appearance and number
        of trials in the single trial data.
        """
        self.pmap['sf'] += [row['Spatial Freq']] * n_trials
        self.pmap['ori'] += [row['Orientation']] * n_trials
        self.pmap['phi'] += [row['Phase']] * n_trials
        self.pmap['id'] += [row['stim_condition_ids']] * n_trials
    
    def generate_class_labels(self, parameter_map, label, n_trials, slc = None):
        Y = []
        if slc == None: 
            for sf in parameter_map[label].unique():
                Y+=[sf]*n_trials
            Y = np.array(Y)
        else:
            for sf in parameter_map[label].unique()[slc]:
                Y+=[sf]*n_trials
            Y = np.array(Y)         
        return (Y*100).astype(int)
            
    def unpack_pref_data(self, l):
        unpacked = []
        for i in l:
            for key, val in i.items():
                unpacked.append(val)
        return unpacked
    
    def collect_max_indices(self, row): 
        sf = round(row['sf'], 2)
        pref_ori = row['pref_ori']
        pref_phase = row['pref_phase']
        sf_map = self.pmap.loc[
            (self.pmap.sf == sf)
            & (self.pmap.ori == pref_ori)
            & (self.pmap.phi == pref_phase)
        ]
        self.map_list.append(sf_map)

    def select_max_data(self, trial_data, parameter_map, pref_phase_ori, n_trials):    
        pref_phase_ori = self.unpack_pref_data(pref_phase_ori)
        self.reset_pmap()
        parameter_map.apply(self.expand_pmap, n_trials = n_trials, axis = 1)
        self.pmap = pd.DataFrame(self.pmap)
        
        unit_best_trial_indices = []
        for unit_pref in pref_phase_ori:
            self.map_list = []
            unit_pref.iloc[1:].apply(self.collect_max_indices, axis = 1)
            unit_best_trial_indices.append(pd.concat(self.map_list))
        std_maxonly = []
        for i0, unit in enumerate(unit_best_trial_indices):
            i1 = unit.index.values
            std_maxonly.append(trial_data[i0][i1])
        self.reset_pmap()
        return np.array(std_maxonly)
    
    def get_trial_averaged_samples(self, trial_data, n_stim = 10, n_trials = 250, n_avg = 5):
        sf_indices = list(
            zip(
                list(range(0, int(n_stim*n_trials), n_trials)),
                list(range(n_trials, int((n_stim*n_trials)+n_trials), n_trials))
            )
        )
        n_samples = n_trials/n_avg
        if n_samples%2 != 0:
            return "n_trials/n_avg must be a whole number"

        trial_averaged_samples = []
        for sf in sf_indices:
            sf_slice = trial_data[:,sf[0]:sf[1],:]
            averages = []
            choices = list(range(n_trials))
            random.shuffle(choices)
            choices = np.array(choices).reshape(int(n_samples), n_avg)

            for choice in choices:
                averages.append(sf_slice[:,choice,:].mean(1))
            averages = np.stack(averages, axis = 1)
            trial_averaged_samples.append(averages)

        return np.concatenate(trial_averaged_samples, axis = 1)
    
    def apply_temporal_smoothing(self, trial_data, kernel, t_axis):
        return np.apply_along_axis(
                        lambda m: np.convolve(
                            m, kernel, 
                            mode='same'
                        ), axis=t_axis, arr=trial_data
                    )