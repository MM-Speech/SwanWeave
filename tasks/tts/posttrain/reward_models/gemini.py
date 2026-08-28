import random
import time
import json

from utils.commons.os_utils import handle_exacption

PROMPT = """

# Role: Senior TTS Voice Quality Evaluator & Audio Engineer

You are a highly experienced TTS (Text-to-Speech) voice quality evaluator and audio engineer.  
Your standards are exceptionally high, equivalent to those of a top-tier animation studio, a major audiobook production house, or a leading commercial TTS provider.

Your task is to meticulously evaluate **TTS-generated audio** to determine if it meets the criteria for a high-quality, production-ready synthetic voice.

## Core Task

Analyze the provided **TTS audio output** and the corresponding input text to evaluate it based on the dimensions below. The input text is mainly English and Chinese, where some Chinese characters could be replaced with the corresponding pinyins. There could also be some speaker indicaters (such as <S1></S1>). 
Your evaluation must be critical, precise, and professional, identifying both strengths and subtle flaws specifically from a TTS perspective (not a human actor recording).

---

## Evaluation Dimensions & Criteria (1-10 Scale)

### 1. Pronunciation Accuracy (发音准确度) - 20% Weight
*Does the TTS engine pronounce everything correctly and consistently?*
- **Word-Level Accuracy (词级发音):** Are all words pronounced correctly (no misreadings, no wrong syllables, no incorrect initials/finals/tones in Chinese, no wrong stress in English or other languages)?
- **Syllable/Phoneme Accuracy (音节/音素准确):** Are there any unnatural substitutions, deletions, or insertions of sounds?  
- **Named Entities & Numbers (专有名词与数字):** Are names, technical terms, and numbers (dates, amounts, times) pronounced correctly and consistently?
- **Missing/Extra Words (漏读与添字):** Are there any omissions or insertions compared to the reference text?
- **Reordering (顺序错误):** Are any segments spoken in the wrong order?

### 2. Pausing & Flow Naturalness (停顿与流畅度) - 15% Weight
*Does the timing of pauses sound like real human speech when reading this text?*
- **Phrase Boundaries (语义分段):** Are pauses inserted at logical phrase/semantic boundaries, or are they missing/placed in awkward positions?
- **Pause Duration (停顿时长):** Are pauses too short (causing the speech to feel rushed) or too long (breaking the flow), or generally natural and balanced?
- **Continuity & Smoothness (连贯性):** Does the speech flow smoothly without abrupt starts/stops, glitches, or timing artifacts?
- **Punctuation Interpretation (标点理解):** Does the TTS handle punctuation in a way that matches the intended structure of the text?

### 3. Prosody & Rhythm Naturalness (韵律与节奏自然度) - 20% Weight
*Does the TTS have human-like prosody, or does it sound robotic?*
- **Intonation Patterns (语调变化):** Does the pitch contour follow natural patterns for statements, questions, emphasis, etc., or does it sound flat, mechanical, or randomly oscillating?
- **Stress & Emphasis (重音与强调):** Are important words appropriately emphasized? Does the model capture contrastive stress or focal points where needed?
- **Rhythm & Tempo (节奏与语速):** Is the speaking rate appropriate to the content (not too fast/too slow)? Is the rhythm consistent and human-like rather than choppy or overly uniform?

### 4. Expressiveness & Stylistic Fit (表现力与风格贴合度) - 15% Weight
*Even as a synthetic voice, does it convey suitable emotion and style for the text?*
- **Emotional Coloring (情感色彩):** Does the voice carry appropriate emotional undertones (e.g., calm, formal, cheerful, serious) without being exaggerated or awkward?
- **Style Consistency (风格一致性):** Is the speaking style (formal/informal, narrative/dialogue, promotional/neutral) consistent and aligned with the content and scenario?
- **Engagement & Presence (吸引力与存在感):** Does the voice feel engaging and pleasant to listen to, or dull and lifeless?

### 5. Overall Naturalness & Human-likeness (整体自然度) - 20% Weight
*Does the TTS sound like a plausible human speaker?*
- **Artifact-Free Naturalness (无明显合成感):** Are there audible TTS artifacts (buzzing, glitching, robotic timbre, unnatural pitch jumps, “unit” boundaries, vocoder artifacts)?
- **Voice Consistency (音色稳定性):** Is the voice timbre stable across the clip (no random changes in tone, harshness, or brightness)?
- **Human-Like Impression (类人程度):** If the listener did not know it was TTS, could they momentarily mistake it for a human in some contexts?

### 6. Technical Audio Quality (技术音质) - 10% Weight
*This assesses the technical quality of the rendered audio file itself, independent of the TTS model’s linguistic/prosodic behavior.*
- **Clarity & Fidelity (清晰度与保真度):** Is the audio clean, crisp, and full-bandwidth? Is there enough detail in the high frequencies without sounding harsh?
- **Noise & Artifacts (噪声与技术伪影):** Is the audio free of hiss, hum, background noise, crackles, or digital glitches introduced by the synthesis or post-processing?
- **Rendering Quality (渲染质量):** Is there any clipping, distortion, aliasing, pumping, or overly aggressive compression/limiting?

---

## Scoring & Output Format

For each of the core dimensions above, give a score from 1 to 10 with a concise but detailed justification.

- **1-3:** Unacceptable for production use; severe issues.
- **4-6:** Clearly suboptimal; usable for low-stakes or internal use only; significant room for improvement.
- **7-8:** Good; meets most production standards but lacks polish or exhibits noticeable synthetic traits.
- **9-10:** Excellent; near or at top-tier TTS quality, highly natural and pleasant with minimal detectable flaws.

### Weights for Final Score (out of 10)
Use the following weights to compute the final score:

- Pronunciation Accuracy - **20%**
- Pausing & Flow Naturalness - **15%**
- Prosody & Rhythm Naturalness - **20%**
- Expressiveness & Stylistic Fit - **15%**
- Overall Naturalness & Human-likeness - **20%**
- Technical Audio Quality - **10%**

**Final Weighted Score** is computed as:
```text
Final_Score = 
    (Pronunciation_Score * 0.20) +
    (Pausing_Flow_Score * 0.15) +
    (Prosody_Score * 0.20) +
    (Expressiveness_Score * 0.15) +
    (Naturalness_Score * 0.20) +
    (Audio_Quality_Score * 0.10)
```

---

## Output Format

Please provide your evaluation in the following JSON format.  
Be concise but specific in all textual explanations.

```json
{
  "Overall_Impression": "A brief, one-sentence summary of your overall feeling about the TTS audio.",
  
  "Pronunciation": "Describe pronunciation accuracy: any mispronunciations, tonal errors, word-level or phoneme-level issues, entity/number handling.",
  "Pronunciation_Score": 0-10,
  
  "Pausing_and_Flow": "Describe whether pauses, phrasing, and overall flow are natural, logical, and smooth.",
  "Pausing_and_Flow_Score": 0-10,

  "Prosody_and_Rhythm": "Describe intonation, stress, rhythm, and speaking rate; note whether it sounds human-like or robotic.",
  "Prosody_and_Rhythm_Score": 0-10,

  "Expressiveness_and_Style": "Describe emotional coloring, style consistency, and engagement level of the voice.",
  "Expressiveness_and_Style_Score": 0-10,

  "Overall_Naturalness": "Describe how human-like the TTS sounds and whether there are noticeable synthetic artifacts.",
  "Overall_Naturalness_Score": 0-10,

  "Audio_Quality": "Describe technical sound quality: clarity, noise, artifacts, rendering issues.",
  "Audio_Quality_Score": 0-10,

  "Final_Weighted_Score": 0-10,
  "Final_Recommendation": "Choose one: **Highly Recommended / Recommended with Reservations / Not Recommended**"
}
```

---

Now, given the input text:

<|text|>

please evaluate the following TTS-generated audio file:

"""


PROMPT2 = """

# Role: Senior TTS Voice Prosody Quality Evaluator & Audio Engineer

You are a highly experienced TTS (Text-to-Speech) voice prosody quality evaluator and audio engineer.  
Your standards are exceptionally high, equivalent to those of a top-tier animation studio, a major audiobook production house, or a leading commercial TTS provider.

Your task is to meticulously evaluate **TTS-generated audio** to determine if it meets the criteria for a high-quality, production-ready synthetic voice, especially focusing on pausing, punctuation, and pronunciation.

## Core Task

Analyze the provided **TTS audio output** and the corresponding input text to evaluate it based on the dimensions below. The input text is mainly English and Chinese, where some Chinese characters could be replaced with the corresponding pinyins. There could also be some speaker indicaters (such as <S1></S1>). 
Your evaluation must be critical, precise, and professional, identifying both strengths and subtle flaws specifically from a TTS perspective (not a human actor recording).

---

## Evaluation Dimensions & Criteria (1-10 Scale)

### 1. Pronunciation Accuracy (发音准确度) - 20% Weight
*Does the TTS engine pronounce everything correctly and consistently?*
- **Word-Level Accuracy (词级发音):** Are all words pronounced correctly (no misreadings, no wrong syllables, no incorrect initials/finals/tones in Chinese, no wrong stress in English or other languages)?
- **Syllable/Phoneme Accuracy (音节/音素准确):** Are there any unnatural substitutions, deletions, or insertions of sounds?  
- **Named Entities & Numbers (专有名词与数字):** Are names, technical terms, and numbers (dates, amounts, times) pronounced correctly and consistently?
- **Missing/Extra Words (漏读与添字):** Are there any omissions or insertions compared to the reference text?
- **Reordering (顺序错误):** Are any segments spoken in the wrong order?

### 2. Pausing & Flow Naturalness (停顿与流畅度) - 50% Weight
*Does the timing of pauses sound like real human speech when reading this text?*
- **Phrase Boundaries (语义分段):** Are pauses inserted at logical phrase/semantic boundaries, or are they missing/placed in awkward positions?
- **Pause Duration (停顿时长):** Are pauses too short (causing the speech to feel rushed) or too long (breaking the flow), or generally natural and balanced?
- **Continuity & Smoothness (连贯性):** Does the speech flow smoothly without abrupt starts/stops, glitches, or timing artifacts?
- **Punctuation Interpretation (标点理解):** Does the TTS handle punctuation in a way that matches the intended structure of the text?

### 3. Prosody & Rhythm Naturalness (韵律与节奏自然度) - 30% Weight
*Does the TTS have human-like prosody, or does it sound robotic?*
- **Intonation Patterns (语调变化):** Does the pitch contour follow natural patterns for statements, questions, emphasis, etc., or does it sound flat, mechanical, or randomly oscillating?
- **Stress & Emphasis (重音与强调):** Are important words appropriately emphasized? Does the model capture contrastive stress or focal points where needed?
- **Rhythm & Tempo (节奏与语速):** Is the speaking rate appropriate to the content (not too fast/too slow)? Is the rhythm consistent and human-like rather than choppy or overly uniform?

---

## Scoring & Output Format

For each of the core dimensions above, give a score from 1 to 10 with a concise but detailed justification.

- **1-3:** Unacceptable for production use; severe issues.
- **4-6:** Clearly suboptimal; usable for low-stakes or internal use only; significant room for improvement.
- **7-8:** Good; meets most production standards but lacks polish or exhibits noticeable synthetic traits.
- **9-10:** Excellent; near or at top-tier TTS quality, highly natural and pleasant with minimal detectable flaws.

### Weights for Final Score (out of 10)
Use the following weights to compute the final score:

- Pronunciation Accuracy - **20%**
- Pausing & Flow Naturalness - **50%**
- Prosody & Rhythm Naturalness - **30%**

**Final Weighted Score** is computed as:
```text
Final_Score = 
    (Pronunciation_Score * 0.20) +
    (Pausing_Flow_Score * 0.50) +
    (Prosody_Score * 0.30)
```

---

## Output Format

Please provide your evaluation in the following JSON format.  
Be concise but specific in all textual explanations.

```json
{
  "Overall_Impression": "A brief, one-sentence summary of your overall feeling about the TTS audio.",
  
  "Pronunciation": "Describe pronunciation accuracy: any mispronunciations, tonal errors, word-level or phoneme-level issues, entity/number handling.",
  "Pronunciation_Score": 0-10,
  
  "Pausing_and_Flow": "Describe whether pauses, phrasing, and overall flow are natural, logical, and smooth.",
  "Pausing_and_Flow_Score": 0-10,

  "Prosody_and_Rhythm": "Describe intonation, stress, rhythm, and speaking rate; note whether it sounds human-like or robotic.",
  "Prosody_and_Rhythm_Score": 0-10,

  "Final_Weighted_Score": 0-10,
  "Final_Recommendation": "Choose one: **Highly Recommended / Recommended with Reservations / Not Recommended**"
}
```

---

Now, given the input text:

<|text|>

please evaluate the following TTS-generated audio file:

"""


from pydantic import BaseModel
class Caption(BaseModel):
    Overall_Impression: str
    Pronunciation: str
    Pronunciation_Score: float
    Pausing_and_Flow: str
    Pausing_and_Flow_Score: float
    Prosody_and_Rhythm: str
    Prosody_and_Rhythm_Score: float
    Expressiveness_and_Style: str
    Expressiveness_and_Style_Score: float
    Overall_Naturalness: str
    Overall_Naturalness_Score: float
    Audio_Quality: str
    Audio_Quality_Score: float
    Final_Weighted_Score: float
    Final_Recommendation: str


from pydantic import BaseModel
class Caption2(BaseModel):
    Overall_Impression: str
    Pronunciation: str
    Pronunciation_Score: float
    Pausing_and_Flow: str
    Pausing_and_Flow_Score: float
    Prosody_and_Rhythm: str
    Prosody_and_Rhythm_Score: float
    Final_Weighted_Score: float
    Final_Recommendation: str


import openai
from data_gen.openai.gemini import GeminiModel
class GeminiRewardModel:
    def __init__(self, **kwargs):
        self.models = {
            'TdCbJzQpvZvTIEpIyQFmmojZdhxSDJiQ': [
                (GeminiModel('gemini-3-pro-preview-new', ak='TdCbJzQpvZvTIEpIyQFmmojZdhxSDJiQ'), 'gemini-3-pro-preview-new'),
            ],
            '8pfD4ynPpyZzvMqQwXmVeKTfG06iJXUg': [
                (GeminiModel('gemini-3-pro-preview-new', ak='8pfD4ynPpyZzvMqQwXmVeKTfG06iJXUg'), 'gemini-3-pro-preview-new'),
            ],
            '3T1CBABxtfQT16JpGeduuiJJI5u02SAq': [
                (GeminiModel('gemini-3-pro-preview-new', ak='3T1CBABxtfQT16JpGeduuiJJI5u02SAq'), 'gemini-3-pro-preview-new'),
            ],
            '19awYFaZsmasIY7uucrpKiF65rNUK8b2_GPT_AK': [
                (GeminiModel('gemini-3-pro-preview-new', ak='19awYFaZsmasIY7uucrpKiF65rNUK8b2_GPT_AK'), 'gemini-3-pro-preview-new'),
            ],
            'UWdoI4zYfhZBuejU0jLe03ew8lkCClDt': [
                (GeminiModel('gemini-3-pro-preview-new', ak='UWdoI4zYfhZBuejU0jLe03ew8lkCClDt'), 'gemini-3-pro-preview-new'),
            ],
            '1ktpxY3UqT76RiyX4PbfhhiKScX7NluH': [
                (GeminiModel('gemini-3-pro-preview-new', ak='1ktpxY3UqT76RiyX4PbfhhiKScX7NluH'), 'gemini-3-pro-preview-new'),
            ],
            'P8go7N6wPAN4eU7DtmbTm9uF57exWCkO_GPT_AK': [
                (GeminiModel('gemini-3-pro-preview-new', ak='P8go7N6wPAN4eU7DtmbTm9uF57exWCkO_GPT_AK'), 'gemini-3-pro-preview-new'),
            ],
            'NNL85HUL8LVG5PqTtBMIofR0mYxC76WR_GPT_AK': [
                (GeminiModel('gemini-3-pro-preview-new', ak='NNL85HUL8LVG5PqTtBMIofR0mYxC76WR_GPT_AK'), 'gemini-3-pro-preview-new'),
            ],
        }

    def process(
            self,
            wav_path,
            text
    ):
        try:
            max_retry = 50

            retry_cnt = 0
            rate_limit_cnt = 0
            result = None
            while retry_cnt < max_retry:
                try:
                    user = random.choice(list(self.models.keys()))
                    model, model_name = random.choice(self.models[user])
                    result = model(
                        wav_path, 
                        PROMPT2.replace('<|text|>', text), 
                        to_base64=True, is_base64=False, modality='audio', max_tokens=8192, 
                        response_format={
                        'type': 'json_schema',
                        'json_schema': 
                            {
                                "name":"Caption2", 
                                "schema": Caption2.model_json_schema()
                            }
                        },
                        thinking_budget=1024
                    ).strip()
                    # print('result', result)
                    # print('type(result)', type(result))
                    if result.lower().startswith('```json'):
                        result = result[7:]
                    if result.lower().endswith('```'):
                        result = result[:-3]
                    result = json.loads(result.strip())
                    result['model_name'] = model_name
                except openai.RateLimitError as err:
                    handle_exacption(err, f'wait for {(rate_limit_cnt+1)*10} seconds ({model_name})')
                    rate_limit_cnt += 1
                    time.sleep(10 * rate_limit_cnt)
                    continue
                except json.JSONDecodeError as err:
                    handle_exacption(err)
                    print(f"{result = }")
                    retry_cnt += 1
                    time.sleep(0.5)
                    continue
                except Exception as err:
                    handle_exacption(err)
                    retry_cnt += 1
                    time.sleep(0.5)
                    continue
                break
            
            # print('gemini_results', result)

            return result
        
        except Exception as err:
            handle_exacption(err)
            return



if __name__ == '__main__':
    gemini = GeminiRewardModel()

    wav_path = '/mnt/bn/sa-ag-data/liruiqi/data/speech/robust_mega3/ref_251211/zh/1-干练白领.wav'
    text = '要买学区房嘛，直接回答我不建议。因为上海的重点初中附近普通的这个小房子都要好几百万，我们家长的压力真的很大呀。但是有一个方法能够让你的孩子即便不在学区里面去买房，也能够大大的提升。他进入重点初中的概率，而且全国都有这个规则，那就是大家要去填好这张表格，它叫做。'

    result = gemini.process(wav_path, text)
