import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchaudio
import scipy.signal

import torch.utils.data
import sounddevice as sd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPS = np.finfo(float).eps

def custom_collate(batch):
    """
    pads data in batch to equal length
    :param batch: data in list
    :return:
    """

    lengths = sorted([np.shape(x)[0] for x in batch], reverse=True)
    unsrt_lengths = [np.shape(x)[0] for x in batch]

    s_ind = np.argsort(unsrt_lengths)[::-1]
    max_len = max(lengths)
    batch = [batch[b_id] for b_id in s_ind ]
    batch= [torch.Tensor(b) for b in batch]

    batch = torch.nn.utils.rnn.pad_sequence(batch,batch_first=True)
    #batch = torch.nn.utils.rnn.pack_sequence(batch)

    return batch

def custom_collate_cp(batch):
    """
    pads data in batch to equal length
    :param batch: data in list
    :return:
    """

    lengths = sorted([np.shape(x)[2] for x in batch], reverse=True)
    unsrt_lengths = [np.shape(x)[2] for x in batch]

    s_ind = np.argsort(unsrt_lengths)[::-1]
    max_len = max(lengths)
    batch = [batch[b_id] for b_id in s_ind ]

    pad_batch = []
    for b_ in batch:
        b_shape = b_.shape
        tmp = torch.cat((b_,torch.zeros((b_shape[0],b_shape[1],max_len-b_shape[2],2))),axis=2)
        pad_batch.append(tmp)

    return pad_batch



class Dataset(Dataset):

    def __init__(self, input_dir, data_scp, transform=None):
        self.input_dir = input_dir
        self.data_scp = pd.read_csv(os.path.join(input_dir,data_scp), delimiter=' ')
        self.transform = transform
    def __len__(self):
        return len(self.data_scp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_filename = os.path.join(self.input_dir,self.data_scp.iloc[idx][-1])
        waveform, sample_rate = torchaudio.load(audio_filename)#, num_frames=3*16000)
        self.samplerate = sample_rate

        if self.transform:
            waveform = self.transform(waveform)

        #return waveform, audio_filename
        print("Read file {}".format(audio_filename))
        return waveform


class STFT(object):

    def __init__(self, n_fft, hop_length=None, win_length=None, window=None, onesided=True, abs_val = True, log_val = True, transpose=True):
        self.n_fft = n_fft
        if not hop_length:
            self.hop_length = n_fft // 2
        else:
            self.hop_length=hop_length
        self.win_length=win_length
        self.window = window
        self.onesided = onesided
        self.abs_val = abs_val
        self.log_val = log_val
        self.transpose = transpose


    def __call__(self, waveform):
        _, _, stft_matrix = scipy.signal.stft(waveform, window=self.window, nperseg=self.win_length, noverlap=self.hop_length,
                          nfft=self.n_fft, return_onesided=True)
        stft_matrix = stft_matrix[0]
        if self.abs_val:
            stft_matrix = np.abs(stft_matrix)
        if self.log_val:
            stft_matrix = np.log10(stft_matrix + EPS)
        if self.transpose:
            stft_matrix = stft_matrix.T

        #return np.float32(stft_matrix)
        return stft_matrix
        #return torch.stft(waveform, n_fft=self.n_fft,hop_length=self.hop_length, win_length=self.win_length, window=self.window, onesided=self.onesided)

class STFT_PT(object):

    def __init__(self, n_fft, hop_length=None, win_length=None, window=None, onesided=True, abs_val = True, log_val = True, transpose=True):
        self.n_fft = n_fft
        if not hop_length:
            self.hop_length = n_fft // 2
        else:
            self.hop_length=hop_length
        self.win_length=win_length
        self.window = window
        self.onesided = onesided
        self.abs_val = abs_val
        self.log_val = log_val
        self.transpose = transpose


    def __call__(self, waveform):
        stft_matrix = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                   window=self.window, onesided=self.onesided)
        if self.abs_val:

            stft_matrix = torch.sqrt(stft_matrix[:, :, :, 0] ** 2 + stft_matrix[:, :, :, 1] ** 2)

        if self.log_val:
            stft_matrix = np.log10(stft_matrix + EPS)
        if self.transpose:
            stft_matrix = stft_matrix.T

        #return np.float32(stft_matrix)
        return stft_matrix
        #return torch.stft(waveform, n_fft=self.n_fft,hop_length=self.hop_length, win_length=self.win_length, window=self.window, onesided=self.onesided)



class Specgram(object):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None, onesided=True):
        self.n_fft = n_fft
        if not hop_length:
            self.hop_length = n_fft // 2
        else:
            self.hop_length=hop_length
        self.win_length=win_length
        if window is None:
            import numpy as np
            import scipy.signal
            window = np.sqrt(scipy.signal.get_window('hann', win_length))
            window = torch.from_numpy(window)#.to(device)


        self.window = window
        self.onesided = onesided


    def __call__(self, waveform):
        return torchaudio.transforms.Spectrogram(n_fft=self.n_fft,hop_length=self.hop_length, win_length=self.win_length, window_fn=self.window)(waveform)





#torchaudio.transforms.Spectrogram(n_fft=400, win_length=None, hop_length=None, pad=0, window_fn=None, power=2, normalized=False, wkwargs=None)
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

if __name__ == '__main__':
    import numpy as np
    win_length = 128
    window = np.sqrt(scipy.signal.get_window('hann', win_length))
    window = torch.tensor(window).float()
    window_fn = torch.hann_window
    def window_fn():
        return torch.sqrt(torch.hann_window)

    input_dir = "/Volumes/LC/WSJ0_mixtures/tr"
    input_dir = '/Users/parviaim/Data/WSJ0_mixtures/tr'
    ds_mix = Dataset(input_dir=input_dir, data_scp='mix.scp', transform=STFT(128, window=window))
    ds_mix_PT = Dataset(input_dir=input_dir, data_scp='mix.scp', transform=STFT_PT(128, window=window, abs_val=False, log_val=False, transpose=False))
    ds_mix_sm = Dataset(input_dir=input_dir, data_scp='mix_4000.scp', transform=Specgram(128, window=window_fn))
    ds_spk1 = Dataset(input_dir=input_dir, data_scp='spk1_4000.scp')
    ds_spk2 = Dataset(input_dir=input_dir, data_scp='spk2_4000.scp')
    bs = 4
    #dl_mix = DataLoader(ds_mix,batch_size=bs,shuffle=False,num_workers=0, collate_fn=custom_collate)
    dl_mix = DataLoader(ds_mix_PT, batch_size=bs, shuffle=False, num_workers=32, collate_fn=custom_collate_cp)

    #for i_batch, sample_batched in enumerate(dl_mix):
    for i_batch, sample_batched in enumerate(dl_mix):
        print("{} d.type, {} d.shape".format(sample_batched[0].type,sample_batched[0].shape))
        print("Batch index: {} (batch size {})".format(i_batch, bs))
        if i_batch == -1:
            break
        if True: #i_batch == 3:
            for b in sample_batched:

                stft_b = b.float()

                #reconstructed = torchaudio.functional.istft(torch.reshape(stft_b,(1,stft_b.shape[2],stft_b.shape[3],stft_b.shape[4])), n_fft =win_length, win_length=win_length, hop_length=int(win_length/2), window=window)
                reconstructed = torchaudio.functional.istft(
                    torch.reshape(stft_b, stft_b.shape), n_fft=win_length,
                    win_length=win_length, hop_length=int(win_length / 2), window=window)
                reconstructed = reconstructed.numpy().reshape(-1,1)
                # fs, original = read(sample_batched[1][0])
                # if 'int' in str(original.dtype):
                #     original = original / 2 ** 16
                #     original = original /max(abs(original))
                #     original = original.reshape(-1,1)
            # #reconstructed = torchaudio.functional.istft(stft_b,
            # #                                            n_fft=win_length, win_length=win_length)
                reconstructed = reconstructed / max(abs(reconstructed))
                # min_len = min(len(original), len(reconstructed))

                # rms = np.sqrt(np.mean((original[:min_len] - reconstructed[:min_len]) ** 2))
                # print("Reconstruction error is {:.10f}".format(rms))
                #sd.play(reconstructed,16000)
                #plt.figure()
                #plt.plot(reconstructed)
                # plt.title(r"Reconstruction error is {}".format(rms))
                #plt.show(block=False)
                #plt.pause(3)
                #plt.close()


    # for s in range(len(ds_mix)):
    #     print("{}, {}, {}".format(ds_mix[s][-1], ds_spk1[s][-1], ds_spk2[s][-1]))
    #     print("Shape is {}:".format(ds_mix_sm[s][0].shape))