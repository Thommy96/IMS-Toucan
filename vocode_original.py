from Preprocessing.AudioPreprocessor import AudioPreprocessor
import soundfile as sf
import torch
from numpy import trim_zeros
from InferenceInterfaces.InferenceArchitectures.InferenceBigVGAN import BigVGAN
from InferenceInterfaces.InferenceArchitectures.InferenceAvocodo import HiFiGANGenerator
import soundfile
from Utility.utils import float2pcm
import os
from Utility.storage_config import MODELS_DIR

if __name__ == '__main__':
    paths_female = ["/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Angry/0015_000605.wav",
                    "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Happy/0015_000814.wav",
                    "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Neutral/0015_000148.wav",
                    "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Sad/0015_001088.wav",
                    "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Surprise/0015_001604.wav"]
    ids_female = [0, 0, 0, 0, 1]

    paths_male = ["/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0014/Angry/0014_000479.wav",
                  "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0014/Happy/0014_001048.wav",
                  "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0014/Neutral/0014_000061.wav",
                  "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0014/Sad/0014_001169.wav",
                  "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0014/Surprise/0014_001639.wav"]   
    ids_male = [1, 1, 1, 1, 0]

    emotions = ["anger", "joy", "neutral", "sadness", "surprise"]

    vocoder_model_path = os.path.join(MODELS_DIR, "Avocodo", "best.pt")
    mel2wav = HiFiGANGenerator(path_to_weights=vocoder_model_path).to(torch.device('cpu'))
    mel2wav.remove_weight_norm()
    mel2wav.eval()

    for i, path in enumerate(paths_male):
        wave, sr = sf.read(path)
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=True, device='cpu')
        norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
        norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
        spec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).cpu()

        wave = mel2wav(spec)
        silence = torch.zeros([10600])
        wav = silence.clone()
        wav = torch.cat((wav, wave, silence), 0)

        wav = [val for val in wav.detach().numpy() for _ in (0, 1)]  # doubling the sampling rate for better compatibility (24kHz is not as standard as 48kHz)
        soundfile.write(file=f"./audios/Original/male/orig_{emotions[i]}_{ids_male[i]}.flac", data=float2pcm(wav), samplerate=48000, subtype="PCM_16")