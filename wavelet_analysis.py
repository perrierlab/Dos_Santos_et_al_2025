import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os
import xlsxwriter
from scipy.ndimage import uniform_filter, generic_filter
from scipy.signal import firls, convolve, iirnotch, lfilter
import pyabf
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

np.set_printoptions(threshold = 100)


def detect_seizures(file,journal, high_freq_start = -1, high_freq_end = -1,high_freq_threshold = -1, width = -1,
low_w = 'cmor20-1', high_w = 'cmor20-1', low_freq_threshold = 1.5,
low_freq_start = 6, low_freq_end = 14, 
overlap = 2,baseline_width = 10000):
    """
    input_folder: the path of the folder which contains the files to be analyzed
    output_excel: the path of the excel sheat to write to(works whether or not it already exists)
    area_size: Area that is integrated over (in seconds)
    area_threshold: threshold for integrated function to cross to be detected as signal
    wave_threshold: threshold to be crossed 3 times within 1 second to be detected as signal
    diff_threshold: differnce between valley and peak for wave to be considered as signal
    seizure_gap: Minimum gap between signals before new seizure can be considered (in seconds)
    overlap: How close area detection and seizure detection must be to be considered as signal(in seconds)
    mode: Whether to use just wave threshold, just area threshold or both. 0 = just area, 1 = just wave, 2 = both.
    plot: Whether or not to plot each sweep with detected seizures(bool)
    """
    mice = {
        "Box 1":"",
        "Box 2":"",
        "Box 3":"",
        "Box 4":"",
    }
    with open(journal) as f:
        lines = f.readlines()
        important_lines = [line for line in lines if len(line)>5]
        for idx,line in enumerate(important_lines):
            splits = line.split(":")
            mouse_segment = splits[1].lstrip()
            mouse = mouse_segment.split(" ")[0].strip()
            mice[list(mice.keys())[idx]] = mouse
            # try:
            #     mouse
            # except:
            #     raise(ValueError("Journal.txt wrongly formatted"))
            # if len(mouse) != 10:
            #     raise(ValueError("Mouse name is not 10 characters long or journal.txt wrongly formatted. If you would like to remove this check it can safely be deleted"))



    file = file.replace("\\","/")
    input_name_chunks = file.split("/")
    channel_idx = {
        0:"IN4",
        1:"IN7",
        2:"IN14",
        3:"IN15" 
    }
    channel_name= {
        "IN4":"Box 1",
        "IN7":"Box 2",
        "IN14":"Box 3",
        "IN15":"Box 4",
        }
    df = pd.DataFrame(columns = ["Week","Day","Pre/Post","Mouse","Box","Seizure frequency (per min)","Sweeps usable"])
    # if file != "2024_01_09_0000.abf": ##insert file name and uncomment these two lines to only run script on that file
    #         continue                  ##
    abf = pyabf.ABF(file)
    duration = abf.sweepLengthSec/60
    fs = abf.sampleRate
    notch_b, notch_a = iirnotch(50, 50, fs)
    bands = np.array([0,1,5,35,60,fs/2]) ##this one the filter
    desired = np.array([0,0,1,1,0,0])
    taps = firls(51, bands, desired, fs=fs, weight= [1,1000,1])
    for channel in abf.channelList:
        seizure_frequencies = []
        for sweep in range(abf.sweepCount):
            abf.setSweep(sweep,channel = channel)
            if np.std(abf.sweepY) > 0.3 or np.abs(np.quantile(abf.sweepY,.75))>1: ###these conditions skip sweeps
                continue
            # if sweep+1 != 8 or channel != 3:
            #     continue
            filtered = lfilter(notch_b,notch_a,abf.sweepY)
            filtered = convolve(filtered,taps,mode ='same')
            filtered[np.where(filtered>0.5)[0]] = 0.5
            filtered[np.where(filtered<-0.5)[0]] = -0.5
            baseline_spot = np.argmin(generic_filter(filtered,np.std, baseline_width)[baseline_width:-baseline_width])+baseline_width
            low_freqs_to_scale = np.arange(low_freq_start,low_freq_end,1)/abf.sampleRate
            high_freqs_to_scale = np.arange(high_freq_start,high_freq_end,1)/abf.sampleRate
            low_scales = pywt.frequency2scale(low_w,low_freqs_to_scale)
            high_scales = pywt.frequency2scale(high_w,high_freqs_to_scale)

            a,b = pywt.cwt(filtered,low_scales,wavelet = low_w,method = "fft")
            c,d = pywt.cwt(filtered,high_scales,wavelet = high_w,method = "fft")
            
            low_freqs = np.abs(a)
            low_freqs = low_freqs/np.mean(low_freqs[:,baseline_spot-baseline_width//2:baseline_spot+baseline_width//2],axis = 1)[:,np.newaxis]
            low_freq_sum = np.mean(low_freqs,axis = 0)-1
            low_freq_sum = uniform_filter(low_freq_sum,size = width)


            high_freqs = np.abs(c)
            high_freqs = high_freqs/np.mean(high_freqs[:,baseline_spot-baseline_width//2:baseline_spot+baseline_width//2],axis = 1)[:,np.newaxis]
            high_freq_sum = np.mean(high_freqs,axis = 0)-1
            high_freq_sum = uniform_filter(high_freq_sum,size = width)

            peaks = high_freq_sum > high_freq_threshold
            diff_array = np.diff(peaks.astype(int))
            switch_indices = np.where(diff_array == 1)[0]
            if high_freq_sum[0]> high_freq_threshold:
                switch_indices = np.concatenate((np.array([0]),switch_indices))

            reset = high_freq_sum > high_freq_threshold*0.8
            reset_diff = np.diff(reset.astype(int))
            fall_indices = np.where(reset_diff == -1)[0]
            high_peak_idxs = []
            if len(switch_indices) > 0:
                signals = [switch_indices[0]]
            else:
                signals = []
            for idx in range(1,len(switch_indices)):
                if len(np.where((fall_indices < switch_indices[idx]) & (fall_indices > switch_indices[idx-1]))[0]) > 0:
                    signals.append(switch_indices[idx])
            for idx,signal in enumerate(signals):
                if idx == len(signals)-1:
                    high_peak_idxs.append(np.argmax(high_freq_sum[signal:])+signal)
                else:
                    high_peak_idxs.append(np.argmax(high_freq_sum[signal:signals[idx+1]])+signal)
            high_peak_idxs = np.array(high_peak_idxs)
            low_freq_periods = np.where(low_freq_sum > low_freq_threshold)
            seizures = np.array([seizure for seizure in high_peak_idxs if len(np.where(np.abs(low_freq_periods-seizure)<overlap*fs)[0]) > 0]).astype(int)
            seizure_frequencies.append(len(seizures)/duration)
        df.loc[input_name_chunks[-1] + " " + channel_name[channel_idx[channel]]] = input_name_chunks[-4:-1]+[mice[channel_name[channel_idx[channel]]],channel_name[channel_idx[channel]],np.mean(seizure_frequencies),len(seizure_frequencies)]

    return df
    
data_folder_name = r"E:\data\EEG\l_serine_temp"
experiment_name = "l_serine"
output_folder_name = r"E:\data\EEG"

def process_condition(args):
    condition_path, journal,start,end,thresh,width = args
    print(condition_path)
    return detect_seizures(condition_path, journal,start,end,thresh,width= width)
    


if __name__ == '__main__':
    start_time = time.time()
    for params in [
    (16,25,2.0,4000),
    ]:
        tasks = []
        for week in os.listdir(data_folder_name):
            week_path = os.path.join(data_folder_name, week)
            for condition in [condition for condition in os.listdir(week_path) if os.path.isdir(os.path.join(week_path, condition))]:
                journal_path = [os.path.join(week_path, file) for file in os.listdir(week_path) if file.endswith(".txt")]
                journal = journal_path[0]
                for session in ['pre',"post"]:
                    file_folder = os.path.join(data_folder_name, week, condition, session)
                    for file in os.listdir(file_folder):
                        tasks.append((os.path.join(file_folder,file), journal)+params)
        with mp.Pool(processes=20) as pool: ## change depending on core count and RAM available
            dfs = pool.map(process_condition,tasks)
        df = pd.concat(dfs)
        output_file_name = os.path.join(output_folder_name,f"{experiment_name}_{params[0]}_{params[1]}_{params[2]}_{params[3]}.xlsx")
        writer = pd.ExcelWriter(output_file_name, engine='xlsxwriter')
        df.sort_index()
        df.sort_values(by=['Week', 'Box','Day','Pre/Post'],ascending = [True,True,True,False],inplace = True)
        df.to_excel(writer, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:F', 15)
        worksheet.set_column('G:H', 25)
        writer.close()
        del writer
    
        print(time.time()-start_time)