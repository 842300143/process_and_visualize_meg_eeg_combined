import numpy as np
import mne
from mne.datasets import sample
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from mne.minimum_norm import apply_inverse, make_inverse_operator
import matplotlib.pyplot as plt
import numpy as np
import os
# 数据模拟函数
def simulate_source_activity():
    data_path = sample.data_path()
    subjects_dir = data_path / 'subjects'
    subject = 'sample'

    # 使用sample数据集中的文件
    evoked_fname = data_path / 'MEG' / subject / 'sample_audvis-ave.fif'
    fwd_fname = data_path / 'MEG' / subject / 'sample_audvis-meg-eeg-oct-6-fwd.fif'

    info = mne.io.read_info(evoked_fname)
    fwd = mne.read_forward_solution(fwd_fname)
    tstep = 1.0 / info['sfreq']
    src = fwd['src']

    selected_label = mne.read_labels_from_annot(subject, regexp='caudalmiddlefrontal-lh', subjects_dir=subjects_dir)[0]
    location = 'center'
    extent = 10.0
    label = mne.label.select_sources(subject, selected_label, location=location, extent=extent, subjects_dir=subjects_dir)

    source_time_series = np.sin(2.0 * np.pi * 18.0 * np.arange(100) * tstep) * 10e-9
    n_events = 50
    events = np.zeros((n_events, 3), int)
    events[:, 0] = 100 + 200 * np.arange(n_events)
    events[:, 2] = 1

    source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
    source_simulator.add_data(label, source_time_series, events)

    fwd_eeg = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=[])
    info_eeg = mne.pick_info(info, mne.pick_types(info, meg=False, eeg=True, exclude=[]))
    raw_eeg = mne.simulation.simulate_raw(info_eeg, source_simulator, forward=fwd_eeg)
    raw_eeg.pick_types(meg=False, eeg=True)
    V_EEG = raw_eeg.get_data()

    meg_info = mne.pick_info(info, mne.pick_types(info, meg=True, stim=True, exclude=[]))
    raw_meg = mne.simulation.simulate_raw(meg_info, source_simulator, forward=fwd)
    raw_meg.pick_types(meg=True, eeg=False)
    B_MEG = raw_meg.get_data()
    info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=True, exclude=[]))

    return source_time_series, V_EEG, B_MEG, label, events, info, fwd

# 损失函数和正则化
def loss_function(V_EEG, B_MEG, weights_eeg, weights_meg, eeg_forward_matrix, meg_forward_matrix, source_time_series, info, label, events, combine_fwd):
    if eeg_forward_matrix.shape[1] != meg_forward_matrix.shape[1]:
        raise ValueError("EEG and MEG forward matrices must have the same number of columns.")
    combined_forward_matrix = np.vstack((weights_eeg[:, None] * eeg_forward_matrix, weights_meg[:, None] * meg_forward_matrix))
    src = combine_fwd['src']
    source_simulator = mne.simulation.SourceSimulator(src, tstep=1.0 / info['sfreq'])
    source_simulator.add_data(label, source_time_series, events)

    combine_fwd['sol']['data'] = combined_forward_matrix
    info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=True, exclude=[]))
    if not all(a == b for a, b in zip(info['ch_names'], combine_fwd['info']['ch_names'])):
        raise ValueError("Channel names in info and fwd do not match.")
    # 确保EEG和MEG的前向矩阵尺寸匹配
    if eeg_forward_matrix.shape[0] != len(mne.pick_types(info, meg=False, eeg=True, exclude=[])):
        raise ValueError("EEG forward matrix size does not match the number of EEG channels.")
    if meg_forward_matrix.shape[0] != len(mne.pick_types(info, meg=True, eeg=False, exclude=[])):
        raise ValueError("MEG forward matrix size does not match the number of MEG channels.")

    raw_simulated = mne.simulation.simulate_raw(info, source_simulator, forward=combine_fwd)
    raw_simulated.pick_types(meg=True, eeg=True)
    simulated_data = raw_simulated.get_data()

    eeg_diff = np.sum((V_EEG - simulated_data[:V_EEG.shape[0]]) ** 2)
    meg_diff = np.sum((B_MEG - simulated_data[-B_MEG.shape[0]:]) ** 2)
    total_loss = eeg_diff + meg_diff

    return total_loss

# 性能评估函数
def combined_model_performance(weights_eeg, weights_meg, eeg_forward_matrix, meg_forward_matrix, V_EEG, B_MEG, source_time_series, info, label, events, combine_fwd):
    performance_metric = loss_function(V_EEG, B_MEG, weights_eeg, weights_meg, eeg_forward_matrix, meg_forward_matrix, source_time_series, info, label, events, combine_fwd)
    return performance_metric





def optimize_and_save_combined_forward_model(source_time_series, V_EEG, B_MEG, label, events, info, fwd,
                                             combined_fwd_fname, n_calls=50):
    """
    Optimize EEG and MEG forward matrices weights using Bayesian optimization and save the combined forward model.

    Parameters:
    - source_time_series, V_EEG, B_MEG, label, events, info, fwd: Simulation results and data structures.
    - combined_fwd_fname: File path to save the combined forward model.
    - n_calls: Number of calls for the Bayesian optimization. Default is 50.

    Returns:
    - None. The combined forward model is saved to the specified file.
    """
    # Determine the number of EEG and MEG channels
    eeg_forward_matrix = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=[])['sol']['data']
    meg_forward_matrix = mne.pick_types_forward(fwd, meg=True, eeg=False, exclude=[])['sol']['data']
    n_eeg_channels = eeg_forward_matrix.shape[0]
    n_meg_channels = meg_forward_matrix.shape[0]

    # Create the parameter space
    param_space = [Real(0, 1, name='weight_eeg_{}'.format(i)) for i in range(n_eeg_channels)] + \
                  [Real(0, 1, name='weight_meg_{}'.format(i)) for i in range(n_meg_channels)]

    # Define the objective function
    @use_named_args(param_space)
    def objective(**params):
        weights_eeg = np.array([params['weight_eeg_{}'.format(i)] for i in range(n_eeg_channels)])
        weights_meg = np.array([params['weight_meg_{}'.format(i)] for i in range(n_meg_channels)])

        combined_forward_matrix = np.vstack((weights_eeg[:, None] * eeg_forward_matrix,
                                             weights_meg[:, None] * meg_forward_matrix))
        fwd['sol']['data'] = combined_forward_matrix

        return combined_model_performance(weights_eeg, weights_meg, eeg_forward_matrix, meg_forward_matrix, V_EEG,
                                          B_MEG, source_time_series, info, label, events, fwd)

    # Perform Bayesian optimization
    optimal_result = gp_minimize(objective, param_space, n_calls=n_calls, random_state=0)

    # Apply the optimal weights
    optimal_weights_eeg = np.array(optimal_result.x[:n_eeg_channels])
    optimal_weights_meg = np.array(optimal_result.x[n_eeg_channels:])

    combined_forward_matrix = np.vstack((optimal_weights_eeg[:, None] * eeg_forward_matrix,
                                         optimal_weights_meg[:, None] * meg_forward_matrix))
    fwd['sol']['data'] = combined_forward_matrix

    # Save the model
    mne.write_forward_solution(combined_fwd_fname, fwd, overwrite=True)
    return fwd


def process_and_visualize_meg_eeg_combined(data_path, raw_fname, event_id, tmin, tmax, baseline, reject, fwd_fname,
                                           combined_fwd_fname, subjects_dir, screenshots_dir, method, snr, loose, depth,
                                           ch_types):
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.info["bads"] += ["MEG 2443", "EEG 053"]

    # 对原始数据设置平均参考投影器
    if 'eeg' in raw:
        raw.set_eeg_reference('average', projection=True).apply_proj()

    events = mne.find_events(raw, stim_channel="STI 014")
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=("meg", "eeg", "eog"), baseline=baseline,
                        reject=reject, preload=True)
    noise_cov = mne.compute_covariance(epochs, tmax=0, method=["shrunk", "empirical"], rank=None)

    for ch_type in ch_types:
        evoked = epochs.average()
        if ch_type != 'combined':
            evoked.pick_types(meg=ch_type == 'meg', eeg=ch_type == 'eeg', eog=False, exclude='bads')

        # 现在不需要对evoked对象进行平均参考设置，因为已经在原始数据上应用了
        fwd_fname_use = combined_fwd_fname if ch_type == 'combined' else fwd_fname
        fwd = mne.read_forward_solution(fwd_fname_use)
        fwd = mne.pick_types_forward(fwd, meg=ch_type != 'eeg', eeg=ch_type != 'meg', exclude=[])
        inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov, loose=loose, depth=depth)
        lambda2 = 1.0 / snr ** 2
        stc, residual = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None,
                                      return_residual=True, verbose=True)

        vertno_max, time_max = stc.get_peak(hemi='rh')
        brain = stc.plot(
            hemi='rh', subjects_dir=subjects_dir,
            clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
            initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10, 
        )
        brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue', scale_factor=0.6, alpha=0.5)
        screenshot_path = f"{screenshots_dir}/{ch_type}_brain.png"
        brain.save_image(screenshot_path)
        print(f"Saved {ch_type} brain screenshot to {screenshot_path}")

    print("Processing and visualization complete.")
# 读取模拟数据
from get_params import  method,n_calls,snr,depth,screenshots_dir,combined_fwd_fname
source_time_series, V_EEG, B_MEG, label, events, info, fwd = simulate_source_activity()
# 调用函数，其中'data/b1-combine-fwd.fif'是输出路径，n_calls=50是数字
combined_fwd = optimize_and_save_combined_forward_model(source_time_series, V_EEG, B_MEG, label, events, info, fwd, combined_fwd_fname, n_calls)
# Example usage:
data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
event_id = {'aud_l': 1}  # 例如，用于听觉左侧刺激的事件ID
tmin, tmax = -0.2, 0.5  # 定义每个事件的时间窗口（秒）
baseline = (None, 0)  # 基线校正窗口（开始，结束）
reject = {'grad': 4000e-13, 'mag': 4e-12, 'eog': 150e-6}  # 拒绝标准
fwd_fname = data_path / 'MEG' / 'sample' / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
combined_dir = data_path / 'MEG' / 'sample' / 'sample_audvis-meg-eeg-oct-6-fwd.fif'  # 假设使用相同的正向模型文件
subjects_dir = data_path / 'subjects'
loose = 0.2
ch_types = ['eeg', 'meg', 'combined']
os.makedirs(screenshots_dir, exist_ok=True)
#下面是活的
# screenshots_dir = 'F:/docker_workspace/test/image'  # 输出路径
# method = 'dSPM'#['dSPM',"MNE","sLORETA"]三选一
# snr = 3.0
# loose = 0.2
# depth = 0.8
# ch_types = ['eeg', 'meg', 'combined']#这三个可以三选任意

# 调用函数
process_and_visualize_meg_eeg_combined(data_path, raw_fname, event_id, tmin, tmax, baseline, reject, fwd_fname,
                                    combined_dir, subjects_dir, screenshots_dir, method, snr, loose, depth,
                                    ch_types)