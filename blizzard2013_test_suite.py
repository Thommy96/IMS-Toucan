import os

import torch

from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface
from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor
from Preprocessing.sentence_embeddings.LEALLASentenceEmbeddingExtractor import LEALLASentenceEmbeddingExtractor


def le_corbeau_et_le_renard(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_integration=False):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = PortaSpeechInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if sent_emb_integration:
        #sentence_embedding_extractor = STSentenceEmbeddingExtractor(model='camembert')
        sentence_embedding_extractor = LEALLASentenceEmbeddingExtractor()

    for i, sentence in enumerate(["Master Raven, on a perched tree, was holding a cheese in his beak.",
                                  "Master Fox, attracted by the smell, told him more or less this language:",
                                  "And hello, Master Raven, how pretty you are! How beautiful you seem to me!",
                                  "No lie, if your bearing is anything like your plumage, you are the Phoenix of the hosts in these woods.",
                                  "At these words the Raven does not feel happy, and to show his beautiful voice, he opens a wide beak, drops his prey.",
                                  "The Fox seized it, and said: My good Sir, learn that every flatterer lives at the expense of the one who listens to him.",
                                  "This lesson is worth a cheese without doubt.",
                                  "The ashamed and confused Raven swore, but a little late, that he would not be caught again.",
                                  "Master Raven, on a perched tree, was holding a cheese in his beak. Master Fox, attracted by the smell, told him more or less this language: And hello, Master Raven, how pretty you are! How beautiful you seem to me! No lie, if your bearing is anything like your plumage, you are the Phoenix of the hosts in these woods. At these words the Raven does not feel happy, and to show his beautiful voice, he opens a wide beak, drops his prey. The Fox seized it, and said: My good Sir, learn that every flatterer lives at the expense of the one who listens to him. This lesson is worth a cheese without doubt. The ashamed and confused Raven swore, but a little late, that he would not be caught again."
                                  ]):
        if sent_emb_integration:
            tts.set_sentence_embedding(sentence, sentence_embedding_extractor)
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/The_raven_and_the_fox_{i}.wav")

def test_sentence(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_integration=False):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = PortaSpeechInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("en")
    sentence = "Well, she said, if I had had your bringing up I might have had as good a temper as you, but now I don't believe I ever shall."
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if sent_emb_integration:
        #sentence_embedding_extractor = STSentenceEmbeddingExtractor(model='camembert')
        sentence_embedding_extractor = LEALLASentenceEmbeddingExtractor()
        prompt = sentence
        tts.set_sentence_embedding(prompt, sentence_embedding_extractor)
    tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/test_sentence.wav")


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"2"
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    exec_device = "cpu"
    print(f"running on {exec_device}")

    speaker_reference = "/mount/resources/speech/corpora/Blizzard2013/train/segmented/wavn/CA-BB-07-04.wav"

    #le_corbeau_et_le_renard(version="04_PortaSpeech_Blizzard2013", model_id="04_PortaSpeech_Blizzard2013", exec_device=exec_device, vocoder_model_path=None, biggan=False, speaker_reference=speaker_reference)
    #le_corbeau_et_le_renard(version="04_PortaSpeech_Blizzard2013_sent_emb_a01", model_id="04_PortaSpeech_Blizzard2013_sent_emb_a01", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #le_corbeau_et_le_renard(version="04_PortaSpeech_Blizzard2013_sent_emb_a01_loss", model_id="04_PortaSpeech_Blizzard2013_sent_emb_a01_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #le_corbeau_et_le_renard(version="04_PortaSpeech_Blizzard2013_sent_emb_a07", model_id="04_PortaSpeech_Blizzard2013_sent_emb_a07", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #le_corbeau_et_le_renard(version="04_PortaSpeech_Blizzard2013_sent_emb_a07_loss", model_id="04_PortaSpeech_Blizzard2013_sent_emb_a07_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)

    test_sentence(version="04_PortaSpeech_Blizzard2013", model_id="04_PortaSpeech_Blizzard2013", exec_device=exec_device, vocoder_model_path=None, biggan=False, speaker_reference=speaker_reference)
    test_sentence(version="04_PortaSpeech_Blizzard2013_sent_emb_a01", model_id="04_PortaSpeech_Blizzard2013_sent_emb_a01", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    test_sentence(version="04_PortaSpeech_Blizzard2013_sent_emb_a01_loss", model_id="04_PortaSpeech_Blizzard2013_sent_emb_a01_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    test_sentence(version="04_PortaSpeech_Blizzard2013_sent_emb_a07", model_id="04_PortaSpeech_Blizzard2013_sent_emb_a07", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    test_sentence(version="04_PortaSpeech_Blizzard2013_sent_emb_a07_loss", model_id="04_PortaSpeech_Blizzard2013_sent_emb_a07_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
