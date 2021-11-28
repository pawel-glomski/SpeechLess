class CfgID:
  VIDEO_STREAM = 'video'  # compatible with PyAV
  AUDIO_STREAM = 'audio'  # compatible with PyAV
  CODEC = 'codec'
  CODEC_OPTIONS = 'codec-options'
  BITRATE = 'bitrate'
  RESOLUTION = 'resolution'
  MAX_FPS = 'max-fps'
  SAMPLE_RATE = 'sample-rate'
  MONO = 'mono'
  TIMELINE_CHANGES = 'timeline-changes'
  METHODS = 'methods'

  @staticmethod
  def has_value(value):
    return value in CfgID.__dict__.values()
