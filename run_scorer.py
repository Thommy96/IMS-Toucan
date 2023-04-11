"""
Example use of the scorer utility to inspect data.

(pre-)trained models and already cache files with extracted features are required.
"""
import os

import torch

from Utility.Scorer import AlignmentScorer
from Utility.Scorer import TTSScorer
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"5"
exec_device = "cuda" if torch.cuda.is_available() else "cpu"

#alignment_scorer = AlignmentScorer(path_to_aligner_model=os.path.join(MODELS_DIR, "Aligner", "aligner.pt"), device=exec_device)
#alignment_scorer.score(path_to_aligner_dataset=os.path.join(PREPROCESSING_DIR, "blizzard2013", "aligner_train_cache.pt"))
#alignment_scorer.save_scores()
#alignment_scorer.show_samples_with_highest_loss(n=50)
#alignment_scorer.remove_samples_with_highest_loss(path_to_aligner_dataset=os.path.join(PREPROCESSING_DIR, "blizzard2013", "aligner_train_cache.pt"), n=20)

tts_scorer = TTSScorer(path_to_portaspeech_model=os.path.join(MODELS_DIR, "05_PortaSpeech_PromptSpeech", "best.pt"), device=exec_device)
tts_scorer.score(path_to_portaspeech_dataset=os.path.join(PREPROCESSING_DIR, "promptspeech"), lang_id="en")
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(20)
