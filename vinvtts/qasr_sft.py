"""Wrapper around the official Qwen3-ASR finetuning/qwen3_asr_sft.py with one fix: the official collator derives the
label mask from a BATCHED prefix-only processor call, and with batch size > 1 those prefix lengths do not line up with
each row (probe 09-27: rows 1-2 of a batch had the whole system prompt + hundreds of <|audio_pad|> tokens labelled ->
loss ~786, grad_norm nan). Here each row's prefix length is measured on its own. Same CLI as the official script.
"""
import sys
sys.path.insert(0, "/mnt/vinvtts/src/qwen3asr/finetuning")
import qwen3_asr_sft as S  # noqa: E402


def _call(self, features):
    audio_paths = [f["audio"] for f in features]; prefix_texts = [f["prefix_text"] for f in features]
    targets = [f["target"] for f in features]; eos = self.processor.tokenizer.eos_token or ""
    audios = [S.load_audio(p, sr=self.sampling_rate) for p in audio_paths]
    full = self.processor(text=[p + t + eos for p, t in zip(prefix_texts, targets)], audio=audios, return_tensors="pt", padding=True, truncation=False)
    labels = full["input_ids"].clone()
    for i, (pt, a) in enumerate(zip(prefix_texts, audios)):
        pl = int(self.processor(text=[pt], audio=[a], return_tensors="pt", padding=False, truncation=False)["attention_mask"].sum())
        real = full["attention_mask"][i].nonzero().flatten()   # the batched processor LEFT-pads (probe 09-27) -> index real tokens
        labels[i, real[:pl]] = -100
    labels[full["attention_mask"] == 0] = -100
    full["labels"] = labels
    return full


S.DataCollatorForQwen3ASRFinetuning.__call__ = _call

# Second fix (diag 09-27): 2.5% of clips give a finite loss but NaN gradients; all had 100-fps feature lengths with
# length % 100 >= 78 (1478 / 1480 / 490 / 496) -> a near-empty tail window in the audio encoder. Pad trailing silence so
# the frame count lands at length % 100 == 50 (<= .8 s of zeros; silence after the utterance does not change the label).
import numpy as _np
_orig_load = S.load_audio


def _load_padded(path, sr=16000):
    w = _orig_load(path, sr=sr); hop = sr // 100; n = len(w) // hop
    k = (50 - n % 100) % 100
    return _np.concatenate([w, _np.zeros(k * hop, dtype=w.dtype)]) if k else w


S.load_audio = _load_padded

if __name__ == "__main__":
    S.main()
