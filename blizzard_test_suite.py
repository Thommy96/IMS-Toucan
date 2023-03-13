import os

import torch

from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface
from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor


def le_corbeau_et_le_renard(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_integration=''):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = PortaSpeechInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("fr")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if sent_emb_integration:
        sentence_embedding_extractor = STSentenceEmbeddingExtractor(model='camembert')

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
            tts.set_sentence_embedding(sentence, sentence_embedding_extractor, sent_emb_integration=sent_emb_integration)
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/Le_corbeau_et_le_renard_{i}.wav")


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    exec_device = "cpu"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"2"
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
    le_corbeau_et_le_renard(version="02_NEB_encoder_single_STCamembert", model_id="NEB_encoder_single_STCamembert", exec_device=exec_device, vocoder_model_path=None, biggan=False, sent_emb_integration='encoder')
