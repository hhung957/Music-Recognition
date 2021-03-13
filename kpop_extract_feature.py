import os
import glob
import librosa
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
from pydub import AudioSegment
from collections import Counter
from annoy import AnnoyIndex

def extract_features(y, sr=16000, nfilt=10, winsteps=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winsteps)
        return feat
    except:
        raise Exception("Extraction feature error")
def crop_feature(feat, i = 0, nb_step=10, maxlen=100):
    crop_feat = np.array(feat[i : i + nb_step]).flatten()
    print(crop_feat.shape)
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    return crop_feat
def pre_train(path, sr = 16000):
	features = []
	songs = []
	for song in tqdm(os.listdir(path)):
		song = os.path.join(wav_path, song)
		y, sr = librosa.load(song, sr = sr)
		feat = extract_features(y)
		for i in range(0, feat.shape[0] - 10, 5):
			features.append(crop_feature(feat, i, nb_step=10))
			songs.append(song)
	return features, songs
def train(features, f = 100):
	t = AnnoyIndex(f)
	for i in range (len(features)):
		v = features[i]
		t.add_item(i, v)
	t.build(100) #trees
	t.save('music.ann')