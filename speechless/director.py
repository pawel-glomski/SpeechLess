from typing import List

from speechless.nlp import sentence_segmentation, TimedToken
from speechless.nlp.method import TEXT_ANALYSIS_METHOD, AnalysisMethod


class Director:

  def __init__(self, analysis_method):

    self.analysis_method = analysis_method
    ...

  def direct(self, recording_path, transcript: List[TimedToken] = None):
    sentences = None
    # if TEXT_ANALYSIS_METHOD in self.analysis_method.type:
    sentences = sentence_segmentation(transcript)
    scores = self.analysis_method.analyze(recording_path, sentences)
    assert len(scores) == len(transcript)
    return scores


import webvtt

vtt = webvtt.read(
    '/mnt/Shared/Code/Python/LectureCutter/data/hidden/Stanford_CS224N_-_NLP_with_Deep_Learning_Winter_2019_Lecture_1_Introduction_and_Word_Vectors-8rXD5-xhemo/subs.en.vtt'
)
timed_transcript = [TimedToken(c.text, c.start_in_seconds, c.end_in_seconds) for c in vtt.captions]
timed_transcript = [t for t in timed_transcript if len(t.text) > 0]

director = Director(AnalysisMethod())
director.direct(
    '/mnt/Shared/Code/Python/LectureCutter/data/hidden/Stanford_CS224N_-_NLP_with_Deep_Learning_Winter_2019_Lecture_1_Introduction_and_Word_Vectors-8rXD5-xhemo/video.mp4',
    timed_transcript)
