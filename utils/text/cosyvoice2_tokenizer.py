import base64
import os
from functools import lru_cache
from typing import Optional, List, Tuple
import torch
from whisper.tokenizer import Tokenizer
import tiktoken
from utils.nn.seq_utils import sequence_mask

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
    "minnan": "minnan",
    "wuyu": "wuyu",
    "dialect": "dialect",
    "zh/en": "zh/en",
    "en/zh": "en/zh",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

AUDIO_EVENT = {
    "ASR": "ASR",
    "AED": "AED",
    "SER": "SER",
    "Speech": "Speech",
    "/Speech": "/Speech",
    "BGM": "BGM",
    "/BGM": "/BGM",
    "Laughter": "Laughter",
    "/Laughter": "/Laughter",
    "Applause": "Applause",
    "/Applause": "/Applause",
}

EMOTION = {
    "HAPPY": "HAPPY",
    "SAD": "SAD",
    "ANGRY": "ANGRY",
    "NEUTRAL": "NEUTRAL",
}

TTS_Vocal_Token = {
    "TTS/B": "TTS/B",
    "TTS/O": "TTS/O",
    "TTS/Q": "TTS/Q",
    "TTS/A": "TTS/A",
    "TTS/CO": "TTS/CO",
    "TTS/CL": "TTS/CL",
    "TTS/H": "TTS/H",
    **{f"TTS/SP{i:02d}": f"TTS/SP{i:02d}" for i in range(1, 14)}
}


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99, special_symbols: Optional[List[str]] = None):
    vocab_path = os.path.join("pretrained_models/cosyvoice2", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        *[f"<|{audio_event}|>" for audio_event in list(AUDIO_EVENT.keys())],
        *[f"<|{emotion}|>" for emotion in list(EMOTION.keys())],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],        # register special tokens for ASR
        *[f"<|{tts}|>" for tts in list(TTS_Vocal_Token.keys())],  # register special tokens for TTS
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
        '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
        '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>', 
        '<S1>', '</S1>','<S2>', '</S2>', 
        '<Audio>', '</Audio>','<BGM>','</BGM>', 
        '<W>', '</W>', '<FILL>', '<PAD>',
        '<S3>', '</S3>','<S4>', '</S4>',
        '<|laughter|>', '<|breathe|>', '<|cry|>',
    ]

    # add pinyin
    with open('egs/tts/pinyin.txt') as f:
        pinyins = f.read().split('\n')
    pinyins_ = set()
    for pinyin in pinyins:
        if pinyin[-1].isdigit():
            pinyins_.add(''.join(list(pinyin)[:-1]))
    pinyins = sorted(list(pinyins_))
    for pinyin in pinyins:
        for tone in ['', '1', '2', '3', '4']:
            specials.append(f'<|py_{pinyin}{tone}|>')

    # other specials
    specials = specials + ['<|sp|>', '<|wbd|>']
    if special_symbols is not None:
        specials = specials + list(special_symbols)
    
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


class CosyVoice2Tokenizer(Tokenizer):
    def encode(self, text, **kwargs):
        return self.encoding.encode(text, allowed_special='all', **kwargs)
    
    def decode(self, token_ids, **kwargs) -> str:
        return self.encoding.decode(token_ids, **kwargs)

    def __call__(self, text, padding=True, return_tensors='pt', **kwargs):
        if isinstance(text, str):
            text = [text]
        token_ids = self.encoding.encode_batch(text, allowed_special='all')

        max_len = max(len(ids) for ids in token_ids)
        token_lens = [len(ids) for ids in token_ids]
        token_ids = [ids + [self.encoding.eot_token] * (max_len - len(ids)) for ids in token_ids]

        token_ids = torch.tensor(token_ids)

        return {
            "input_ids": token_ids,
            "attention_mask": sequence_mask(torch.LongTensor(token_lens), max_len)
        }

@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool = True,
    num_languages: int = 100,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
    special_symbols: Optional[Tuple[str]] = None
) -> CosyVoice2Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual_zh_ja_yue_char_del"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = get_encoding(name=encoding_name, num_languages=num_languages, special_symbols=special_symbols)

    return CosyVoice2Tokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )


class QwenTokenizer():
    def __init__(self, token_path, skip_special_tokens=True):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
        }
        from transformers import AutoTokenizer
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text


@lru_cache(maxsize=None)
def get_qwen_tokenizer(
    token_path: str,
    skip_special_tokens: bool
) -> QwenTokenizer:
    return QwenTokenizer(token_path=token_path, skip_special_tokens=skip_special_tokens)


if __name__ == '__main__':
    tokenizer = get_tokenizer(multilingual=True)

    print(f"{tokenizer.encoding.n_vocab = }")
    # print(f"{tokenizer.special_tokens = }")

    # tokens = tokenizer.encode("IndexTTS 正式发布1.0版本了，效果666")
    # tokens = tokenizer.encode("<MASK><FILL>它富含高浓度的 omega3和各种维生素，口感很好！可以直接喂食！也可以拌在猫粮或喜欢的零食里！", allowed_special='all')
    # tokens = tokenizer.encode("<MASK><FILL>ยินดีที่ได้รู้จัก")
    tokens = tokenizer.encode("")
    
    print(f"{tokens = }")

    for token in tokens:
        print(f"{token} {tokenizer.decode([token])}")

    print(f"{tokenizer.decode(tokens) = }")


    # tokens = tokenizer.encode("IndexTTS 正式发布1.0版本了，效果666")
    # print(f"{tokens = }")
    # tokens = tokenizer(["IndexTTS 正式发布1.0版本了，效果666", "IndexTTS 正式发布1.0版本了"])['input_ids']
    # print(f"{tokens = }")


    # from transformers import AutoTokenizer, Qwen2Tokenizer
    # text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # text_tokenizer.add_tokens([
    #     '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
    #     '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>', 
    #     '<S1>', '</S1>','<S2>', '</S2>', 
    #     '<Audio>', '</Audio>','<BGM>','</BGM>', 
    #     '<W>', '</W>', '<FILL>', '<PAD>',
    #     '<S3>', '</S3>','<S4>', '</S4>', 
    #     '<|laughter|>', '<|breathe|>', '<|cry|>'
    # ], special_tokens=True)
    # vocab_size = len(text_tokenizer)

    # # tokens = text_tokenizer.encode("IndexTTS 正式发布1.0版本了，效果666")
    # tokens = text_tokenizer.encode("它富含高浓度的 omega3和各种维生素，口感很好！可以直接喂食！也可以拌在猫粮或喜欢的零食里！")

    # print(f"{tokens = }")

    # for token in tokens:
    #     print(f"{token} {text_tokenizer.decode([token])}")


