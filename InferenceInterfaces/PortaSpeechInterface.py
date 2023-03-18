import itertools
import os

import librosa.display as lbd
import matplotlib.pyplot as plt
import sounddevice
import soundfile
import torch
from pedalboard import Compressor
from pedalboard import HighShelfFilter
from pedalboard import HighpassFilter
from pedalboard import LowpassFilter
from pedalboard import NoiseGate
from pedalboard import PeakFilter
from pedalboard import Pedalboard

from InferenceInterfaces.InferenceArchitectures.InferenceAvocodo import HiFiGANGenerator
from InferenceInterfaces.InferenceArchitectures.InferenceBigVGAN import BigVGAN
from InferenceInterfaces.InferenceArchitectures.InferencePortaSpeech import PortaSpeech
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.storage_config import MODELS_DIR


class PortaSpeechInterface(torch.nn.Module):

    def __init__(self,
                 device="cpu",
                 # device that everything computes on. If a cuda device is available, this can speed things up by an order of magnitude.
                 tts_model_path=os.path.join(MODELS_DIR, f"PortaSpeech_Meta", "best.pt"),
                 # path to the PortaSpeech checkpoint or just a shorthand if run standalone
                 vocoder_model_path=None,
                 # path to the hifigan/avocodo/bigvgan checkpoint
                 faster_vocoder=True,
                 # whether to use the quicker HiFiGAN or the better BigVGAN
                 language="en",
                 # initial language of the model, can be changed later with the setter methods
                 use_signalprocessing=False,
                 # some subtle effects that are frequently used in podcasting
                 ):
        super().__init__()
        self.device = device
        if not tts_model_path.endswith(".pt"):
            # default to shorthand system
            tts_model_path = os.path.join(MODELS_DIR, tts_model_path, "best.pt")
        if vocoder_model_path is None:
            if faster_vocoder:
                vocoder_model_path = os.path.join(MODELS_DIR, "Avocodo", "best.pt")
            else:
                vocoder_model_path = os.path.join(MODELS_DIR, "BigVGAN", "best.pt")
        self.use_signalprocessing = use_signalprocessing
        if self.use_signalprocessing:
            self.effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=60),
                                               HighShelfFilter(cutoff_frequency_hz=8000, gain_db=5.0),
                                               LowpassFilter(cutoff_frequency_hz=17000),
                                               PeakFilter(cutoff_frequency_hz=150, gain_db=2.0),
                                               PeakFilter(cutoff_frequency_hz=220, gain_db=-2.0),
                                               PeakFilter(cutoff_frequency_hz=900, gain_db=-2.0),
                                               PeakFilter(cutoff_frequency_hz=3200, gain_db=-2.0),
                                               PeakFilter(cutoff_frequency_hz=7500, gain_db=-2.0),
                                               NoiseGate(),
                                               Compressor(ratio=2.0)])

        ################################
        #   build text to phone        #
        ################################
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True)

        ################################
        #   load weights               #
        ################################
        checkpoint = torch.load(tts_model_path, map_location='cpu')

        ################################
        #   load phone to mel model    #
        ################################
        self.use_lang_id = True
        try:
            self.phone2mel = PortaSpeech(weights=checkpoint["model"])  # multi speaker multi language
        except (RuntimeError, TypeError):
            try:
                self.use_lang_id = False
                self.phone2mel = PortaSpeech(weights=checkpoint["model"],
                                             lang_embs=None)  # multi speaker single language
            except (RuntimeError, TypeError):
                try:
                    self.phone2mel = PortaSpeech(weights=checkpoint["model"],
                                                lang_embs=None,
                                                utt_embed_dim=None)  # single speaker
                except (RuntimeError, TypeError):
                    sent_embed_dim=None
                    sent_embed_encoder=False
                    sent_embed_decoder=False
                    sent_embed_each=False
                    concat_sent_style=False
                    use_concat_projection=False
                    sent_embed_postnet=False
                    if "a01" in tts_model_path:
                        sent_embed_dim=768
                        sent_embed_encoder=True
                    if "a02" in tts_model_path:
                        sent_embed_dim=768
                        sent_embed_encoder=True
                        sent_embed_decoder=True
                    if "a03" in tts_model_path:
                        sent_embed_dim=768
                        sent_embed_encoder=True
                        sent_embed_decoder=True
                        sent_embed_postnet=True
                    if "a04" in tts_model_path:
                        sent_embed_dim=768
                        sent_embed_encoder=True
                        sent_embed_each=True
                    if "a05" in tts_model_path:
                        sent_embed_dim=768
                        sent_embed_encoder=True
                        sent_embed_decoder=True
                        sent_embed_each=True
                    if "a06" in tts_model_path:
                        sent_embed_dim=768
                        sent_embed_encoder=True
                        sent_embed_decoder=True
                        sent_embed_each=True
                        sent_embed_postnet=True
                    if "a07" in tts_model_path:
                        sent_embed_dim=768
                        concat_sent_style=True
                        use_concat_projection=True
                    if "a08" in tts_model_path:
                        sent_embed_dim=768
                        concat_sent_style=True
                    self.phone2mel = PortaSpeech(weights=checkpoint["model"],
                                                                    lang_embs=None,
                                                                    sent_embed_dim=sent_embed_dim,
                                                                    sent_embed_encoder=sent_embed_encoder,
                                                                    sent_embed_decoder=sent_embed_decoder,
                                                                    sent_embed_each=sent_embed_each,
                                                                    concat_sent_style=concat_sent_style,
                                                                    use_concat_projection=use_concat_projection,
                                                                    sent_embed_postnet=sent_embed_postnet)
        with torch.no_grad():
            self.phone2mel.store_inverse_all()
        self.phone2mel = self.phone2mel.to(torch.device(device))

        #################################
        #  load mel to style models     #
        #################################
        self.style_embedding_function = StyleEmbedding()
        check_dict = torch.load(os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"), map_location="cpu")
        self.style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        self.style_embedding_function.to(self.device)

        ################################
        #  load mel to wave model      #
        ################################
        if faster_vocoder:
            self.mel2wav = HiFiGANGenerator(path_to_weights=vocoder_model_path).to(torch.device(device))
        else:
            self.mel2wav = BigVGAN(path_to_weights=vocoder_model_path).to(torch.device(device))
        self.mel2wav.remove_weight_norm()

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        try:
            self.default_sentence_embedding = checkpoint["default_sent_emb"].to(self.device)
        except KeyError:
            self.default_sentence_embedding = None
        self.audio_preprocessor = AudioPreprocessor(input_sr=16000, output_sr=16000, cut_silence=True, device=self.device)
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.style_embedding_function.eval()
        if self.use_lang_id:
            self.lang_id = get_language_id(language)
        else:
            self.lang_id = None
        self.to(torch.device(device))
        self.eval()

    def set_utterance_embedding(self, path_to_reference_audio="", embedding=None):
        if embedding is not None:
            self.default_utterance_embedding = embedding.squeeze().to(self.device)
            return
        assert os.path.exists(path_to_reference_audio)
        wave, sr = soundfile.read(path_to_reference_audio)
        if sr != self.audio_preprocessor.sr:
            self.audio_preprocessor = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True, device=self.device)
        spec = self.audio_preprocessor.audio_to_mel_spec_tensor(wave).transpose(0, 1)
        spec_len = torch.LongTensor([len(spec)])
        self.default_utterance_embedding = self.style_embedding_function(spec.unsqueeze(0).to(self.device),
                                                                         spec_len.unsqueeze(0).to(self.device)).squeeze()
        
    def set_sentence_embedding(self, prompt:str, sentence_embedding_extractor):
        prompt_embedding = sentence_embedding_extractor.encode([prompt]).squeeze().to(self.device)
        self.default_sentence_embedding = prompt_embedding

    def set_language(self, lang_id):
        """
        The id parameter actually refers to the shorthand. This has become ambiguous with the introduction of the actual language IDs
        """
        self.set_phonemizer_language(lang_id=lang_id)
        self.set_accent_language(lang_id=lang_id)

    def set_phonemizer_language(self, lang_id):
        self.text2phone = ArticulatoryCombinedTextFrontend(language=lang_id, add_silence_to_end=True)

    def set_accent_language(self, lang_id):
        if self.use_lang_id:
            self.lang_id = get_language_id(lang_id).to(self.device)
        else:
            self.lang_id = None

    def forward(self,
                text,
                view=False,
                duration_scaling_factor=1.0,
                pitch_variance_scale=1.0,
                energy_variance_scale=1.0,
                pause_duration_scaling_factor=1.0,
                durations=None,
                pitch=None,
                energy=None,
                input_is_phones=False,
                return_plot_as_filepath=False):
        """
        duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
        pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
        energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        with torch.inference_mode():
            phones = self.text2phone.string_to_tensor(text, input_phonemes=input_is_phones).to(torch.device(self.device))
            mel, durations, pitch, energy = self.phone2mel(phones,
                                                           return_duration_pitch_energy=True,
                                                           utterance_embedding=self.default_utterance_embedding,
                                                           sentence_embedding=self.default_sentence_embedding,
                                                           durations=durations,
                                                           pitch=pitch,
                                                           energy=energy,
                                                           lang_id=self.lang_id,
                                                           duration_scaling_factor=duration_scaling_factor,
                                                           pitch_variance_scale=pitch_variance_scale,
                                                           energy_variance_scale=energy_variance_scale,
                                                           pause_duration_scaling_factor=pause_duration_scaling_factor,
                                                           device=self.device)

            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel)
            if self.use_signalprocessing:
                try:
                    wave = torch.Tensor(self.effects(wave.cpu().numpy(), 24000))
                except ValueError:
                    # if the audio is too short, a value error might arise
                    pass

        if view or return_plot_as_filepath:
            from Utility.utils import cumsum_durations
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(),
                         ax=ax[1],
                         sr=16000,
                         cmap='GnBu',
                         y_axis='mel',
                         x_axis=None,
                         hop_length=256)
            ax[0].yaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
            ax[1].xaxis.grid(True, which='minor')
            ax[1].set_xticks(label_positions, minor=False)
            if input_is_phones:
                phones = text.replace(" ", "|")
            else:
                phones = self.text2phone.get_phone_string(text, for_plot_labels=True)
            ax[1].set_xticklabels(phones)
            word_boundaries = list()
            for label_index, phone in enumerate(phones):
                if phone == "|":
                    word_boundaries.append(label_positions[label_index])

            try:
                prev_word_boundary = 0
                word_label_positions = list()
                for word_boundary in word_boundaries:
                    word_label_positions.append((word_boundary + prev_word_boundary) / 2)
                    prev_word_boundary = word_boundary
                word_label_positions.append((duration_splits[-1] + prev_word_boundary) / 2)

                secondary_ax = ax[1].secondary_xaxis('bottom')
                secondary_ax.tick_params(axis="x", direction="out", pad=24)
                secondary_ax.set_xticks(word_label_positions, minor=False)
                secondary_ax.set_xticklabels(text.split())
                secondary_ax.tick_params(axis='x', colors='orange')
                secondary_ax.xaxis.label.set_color('orange')
            except ValueError:
                ax[0].set_title(text)
            except IndexError:
                ax[0].set_title(text)

            ax[1].vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
            ax[1].vlines(x=word_boundaries, colors="orange", linestyles="solid", ymin=0.0, ymax=8000, linewidth=1.2)
            pitch_array = pitch.cpu().numpy()
            for pitch_index, xrange in enumerate(zip(duration_splits[:-1], duration_splits[1:])):
                if pitch_array[pitch_index] != 0:
                    ax[1].hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="blue",
                                 linestyles="solid", linewidth=0.5)
            plt.subplots_adjust(left=0.05, bottom=0.12, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            if not return_plot_as_filepath:
                plt.show()
            else:
                plt.savefig("tmp.png")
                return wave, "tmp.png"
        return wave

    def read_to_file(self,
                     text_list,
                     file_location,
                     duration_scaling_factor=1.0,
                     pitch_variance_scale=1.0,
                     energy_variance_scale=1.0,
                     silent=False,
                     dur_list=None,
                     pitch_list=None,
                     energy_list=None):
        """
        Args:
            silent: Whether to be verbose about the process
            text_list: A list of strings to be read
            file_location: The path and name of the file it should be saved to
            energy_list: list of energy tensors to be used for the texts
            pitch_list: list of pitch tensors to be used for the texts
            dur_list: list of duration tensors to be used for the texts
            duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
            pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
            energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        if not dur_list:
            dur_list = []
        if not pitch_list:
            pitch_list = []
        if not energy_list:
            energy_list = []
        wav = None
        silence = torch.zeros([24000])
        for (text, durations, pitch, energy) in itertools.zip_longest(text_list, dur_list, pitch_list, energy_list):
            if text.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(text))
                if wav is None:
                    wav = self(text,
                               durations=durations.to(self.device) if durations is not None else None,
                               pitch=pitch.to(self.device) if pitch is not None else None,
                               energy=energy.to(self.device) if energy is not None else None,
                               duration_scaling_factor=duration_scaling_factor,
                               pitch_variance_scale=pitch_variance_scale,
                               energy_variance_scale=energy_variance_scale).cpu()
                    wav = torch.cat((wav, silence), 0)
                else:
                    wav = torch.cat((wav, self(text,
                                               durations=durations.to(self.device) if durations is not None else None,
                                               pitch=pitch.to(self.device) if pitch is not None else None,
                                               energy=energy.to(self.device) if energy is not None else None,
                                               duration_scaling_factor=duration_scaling_factor,
                                               pitch_variance_scale=pitch_variance_scale,
                                               energy_variance_scale=energy_variance_scale).cpu()), 0)
                    wav = torch.cat((wav, silence), 0)
        soundfile.write(file=file_location, data=wav.cpu().numpy(), samplerate=24000)

    def read_aloud(self,
                   text,
                   view=False,
                   duration_scaling_factor=1.0,
                   pitch_variance_scale=1.0,
                   energy_variance_scale=1.0,
                   blocking=False):
        if text.strip() == "":
            return
        wav = self(text,
                   view,
                   duration_scaling_factor=duration_scaling_factor,
                   pitch_variance_scale=pitch_variance_scale,
                   energy_variance_scale=energy_variance_scale).cpu()
        wav = torch.cat((wav, torch.zeros([12000])), 0)
        if not blocking:
            sounddevice.play(wav.numpy(), samplerate=24000)
        else:
            sounddevice.play(torch.cat((wav, torch.zeros([6000])), 0).numpy(), samplerate=24000)
            sounddevice.wait()
