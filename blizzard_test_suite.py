import os

import torch

from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface
from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor
from Preprocessing.sentence_embeddings.LEALLASentenceEmbeddingExtractor import LEALLASentenceEmbeddingExtractor


def le_corbeau_et_le_renard(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_integration=False):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = PortaSpeechInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("fr")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if sent_emb_integration:
        #sentence_embedding_extractor = STSentenceEmbeddingExtractor(model='camembert')
        sentence_embedding_extractor = LEALLASentenceEmbeddingExtractor()

    for i, sentence in enumerate(["Maître Corbeau, sur un arbre perché, tenait en son bec un fromage.",
                                  "Maître Renard, par l’odeur alléché, lui tint à peu près ce langage:",
                                  "Et bonjour, Monsieur du Corbeau, que vous êtes joli! que vous me semblez beau!",
                                  "Sans mentir, si votre ramage se rapporte à votre plumage, vous êtes le Phénix des hôtes de ces bois.",
                                  "À ces mots le Corbeau ne se sent pas de joie, et pour montrer sa belle voix, il ouvre un large bec, laisse tomber sa proie.",
                                  "Le Renard s’en saisit, et dit: Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute.",
                                  "Cette leçon vaut bien un fromage sans doute.",
                                  "Le Corbeau honteux et confus jura, mais un peu tard, qu’on ne l’y prendrait plus.",
                                  "Maître Corbeau, sur un arbre perché, tenait en son bec un fromage. Maître Renard, par l’odeur alléché, lui tint à peu près ce langage: Et bonjour, Monsieur du Corbeau, que vous êtes joli! que vous me semblez beau! Sans mentir, si votre ramage se rapporte à votre plumage, vous êtes le Phénix des hôtes de ces bois. À ces mots le Corbeau ne se sent pas de joie, et pour montrer sa belle voix, il ouvre un large bec, laisse tomber sa proie. Le Renard s’en saisit, et dit: Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute. Cette leçon vaut bien un fromage sans doute. Le Corbeau honteux et confus jura, mais un peu tard, qu’on ne l’y prendrait plus."
                                  ]):
        if sent_emb_integration:
            tts.set_sentence_embedding(sentence, sentence_embedding_extractor)
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/Le_corbeau_et_le_renard_{i}.wav")

def phonetically_interesting_sentences_seen(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_integration=False):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = PortaSpeechInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("fr")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if sent_emb_integration:
        #sentence_embedding_extractor = STSentenceEmbeddingExtractor(model='camembert')
        sentence_embedding_extractor = LEALLASentenceEmbeddingExtractor()

    for i, sentence in enumerate(["Les passagers, s'aidant les uns les autres, parvinrent à se dégager des mailles du filet.",
                                  "On peut le faire si ce principe, vient en contrepartie d'un autre principe de même niveau.",
                                  "Ce manuscrit, signé de mon nom, complété par l'histoire de ma vie, sera renfermé dans un petit appareil insubmersible.",
                                  "On vous inviterait à chasser l'ours dans les montagnes de la Suisse, que vous diriez: «Très bien!»",
                                  "Mais de là à monter cette histoire en épingle",
                                  "La France mérite un tout autre projet.",
                                  "Mon maître, dit alors nab, j'ai l'idée que nous pouvons chercher tant que nous voudrons le monsieur dont il s'agit, mais que nous ne le découvrirons que quand il lui plaira.",
                                  "Pendant la première semaine du mois d'août, les rafales s'apaisèrent peu à peu, et l'atmosphère recouvra un calme qu'elle semblait avoir à jamais perdu.",
                                  ]):
        if sent_emb_integration:
            tts.set_sentence_embedding(sentence, sentence_embedding_extractor)
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/seen_sentences_{i}.wav")

def phonetically_interesting_sentences_unseen(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_integration=False):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = PortaSpeechInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("fr")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if sent_emb_integration:
        #sentence_embedding_extractor = STSentenceEmbeddingExtractor(model='camembert')
        sentence_embedding_extractor = LEALLASentenceEmbeddingExtractor()

    for i, sentence in enumerate(["Les amis ont vu un ancien ami en avril, dit-on.",
                                  "Des amis ont vu en avril un vieil ami qui était très aimable, dit-on.",
                                  "C'est une maison où l'on peut aller quand il pleut.",
                                  "Après un tour de présentation, ils sont allés."
                                  ]):
        if sent_emb_integration:
            tts.set_sentence_embedding(sentence, sentence_embedding_extractor)
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/unseen_sentences_{i}.wav")

def test_sentence(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_integration=False):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = PortaSpeechInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("fr")
    sentence = "Je suis en colère, c'est injuste."
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
    #exec_device = "cpu"
    print(f"running on {exec_device}")

    #le_corbeau_et_le_renard(version="03_NEB_pretrain", model_id="NEB", exec_device=exec_device, vocoder_model_path=None, biggan=True)
    #le_corbeau_et_le_renard(version="03_AD_pretrain", model_id="AD", exec_device=exec_device, vocoder_model_path=None, biggan=True)

    #le_corbeau_et_le_renard(version="02_French_reference", model_id="French", exec_device=exec_device, vocoder_model_path=None, biggan=False, speaker_reference='/mount/resources/speech/corpora/Blizzard2023/NEB/EC_LFDP_NEB_00_0003_15.wav')
    #le_corbeau_et_le_renard(version="02_French_concat_STCamembert", model_id="French_concat_STCamembert", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration='concat')
    #le_corbeau_et_le_renard(version="02_French_encoder_STCamembert", model_id="French_encoder_STCamembert", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration='encoder')
    #le_corbeau_et_le_renard(version="02_French_concat_norm_STCamembert", model_id="French_concat_norm_STCamembert", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration='concat')

    #le_corbeau_et_le_renard(version="02_NEB", model_id="NEB", exec_device=exec_device, vocoder_model_path=None, biggan=False)
    #le_corbeau_et_le_renard(version="02_NEB_concat_STCamembert", model_id="NEB_concat_STCamembert", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration='concat')
    #le_corbeau_et_le_renard(version="02_NEB_encoder_STCamembert", model_id="NEB_encoder_STCamembert", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration='encoder')
    #le_corbeau_et_le_renard(version="02_NEB_encoder_single_STCamembert", model_id="NEB_encoder_single_STCamembert", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration='encoder')

    #le_corbeau_et_le_renard(version="04_PortaSpeech_AD", model_id="04_PortaSpeech_AD", exec_device=exec_device, vocoder_model_path=None, biggan=False)

    #le_corbeau_et_le_renard(version="03_PortaSpeech_NEB", model_id="03_PortaSpeech_NEB", exec_device=exec_device, vocoder_model_path=None, biggan=False)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_NEB_sent_emb_a05", model_id="03_PortaSpeech_NEB_sent_emb_a05", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_NEB_sent_emb_a05_loss", model_id="03_PortaSpeech_NEB_sent_emb_a05_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_NEB_sent_emb_a07", model_id="03_PortaSpeech_NEB_sent_emb_a07", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_NEB_sent_emb_a08", model_id="03_PortaSpeech_NEB_sent_emb_a08", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)

    #speaker_reference = "/mount/resources/speech/corpora/Blizzard2023/AD/DIVERS_BOOK_AD_04_0002_25.wav"
    #speaker_reference = "/mount/resources/speech/corpora/Blizzard2023/NEB/ES_LMP_NEB_04_0014_1.wav"
    speaker_reference = None

    #le_corbeau_et_le_renard(version="03_PortaSpeech_French_ref", model_id="03_PortaSpeech_French", exec_device=exec_device, vocoder_model_path=None, biggan=False, speaker_reference=speaker_reference)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_French_sent_emb_a01", model_id="03_PortaSpeech_French_sent_emb_a01", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_French_sent_emb_a01_loss", model_id="03_PortaSpeech_French_sent_emb_a01_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_French_sent_emb_a03_loss", model_id="03_PortaSpeech_French_sent_emb_a03_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_French_sent_emb_a06_loss", model_id="03_PortaSpeech_French_sent_emb_a06_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_French_sent_emb_a07", model_id="03_PortaSpeech_French_sent_emb_a07", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)
    #le_corbeau_et_le_renard(version="03_PortaSpeech_French_sent_emb_a07_loss", model_id="03_PortaSpeech_French_sent_emb_a07_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True)

    #test_sentence(version="03_PortaSpeech_French", model_id="03_PortaSpeech_French", exec_device=exec_device, vocoder_model_path=None, biggan=False, speaker_reference=speaker_reference)
    #test_sentence(version="03_PortaSpeech_French_sent_emb_a01", model_id="03_PortaSpeech_French_sent_emb_a01", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #test_sentence(version="03_PortaSpeech_French_sent_emb_a01_loss", model_id="03_PortaSpeech_French_sent_emb_a01_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #test_sentence(version="03_PortaSpeech_French_sent_emb_a03_loss", model_id="03_PortaSpeech_French_sent_emb_a03_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #test_sentence(version="03_PortaSpeech_French_sent_emb_a06_loss", model_id="03_PortaSpeech_French_sent_emb_a06_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #test_sentence(version="03_PortaSpeech_French_sent_emb_a07", model_id="03_PortaSpeech_French_sent_emb_a07", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #test_sentence(version="03_PortaSpeech_French_sent_emb_a07_loss", model_id="03_PortaSpeech_French_sent_emb_a07_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)

    #le_corbeau_et_le_renard(version="04_PortaSpeech_French_NEB", model_id="04_PortaSpeech_French", exec_device=exec_device, vocoder_model_path=None, biggan=False, speaker_reference=speaker_reference)
    #le_corbeau_et_le_renard(version="04_PortaSpeech_French_sent_emb_a01_NEB", model_id="04_PortaSpeech_French_sent_emb_a01", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #le_corbeau_et_le_renard(version="04_PortaSpeech_French_sent_emb_a01_loss_NEB", model_id="04_PortaSpeech_French_sent_emb_a01_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #le_corbeau_et_le_renard(version="04_PortaSpeech_French_sent_emb_a07_NEB", model_id="04_PortaSpeech_French_sent_emb_a07", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    #le_corbeau_et_le_renard(version="04_PortaSpeech_French_sent_emb_a07_loss_NEB", model_id="04_PortaSpeech_French_sent_emb_a07_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)

    phonetically_interesting_sentences_unseen(version="04_PortaSpeech_French", model_id="04_PortaSpeech_French", exec_device=exec_device, vocoder_model_path=None, biggan=False, speaker_reference=speaker_reference)
    phonetically_interesting_sentences_unseen(version="04_PortaSpeech_French_sent_emb_a01", model_id="04_PortaSpeech_French_sent_emb_a01", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    phonetically_interesting_sentences_unseen(version="04_PortaSpeech_French_sent_emb_a01_loss", model_id="04_PortaSpeech_French_sent_emb_a01_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    phonetically_interesting_sentences_unseen(version="04_PortaSpeech_French_sent_emb_a07", model_id="04_PortaSpeech_French_sent_emb_a07", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
    phonetically_interesting_sentences_unseen(version="04_PortaSpeech_French_sent_emb_a07_loss", model_id="04_PortaSpeech_French_sent_emb_a07_loss", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration=True, speaker_reference=speaker_reference)
