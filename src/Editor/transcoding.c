/*
 * Copyright (c) 2010 Nicolas George
 * Copyright (c) 2011 Stefano Sabatini
 * Copyright (c) 2014 Andrey Utkin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * @file
 * API example for demuxing, decoding, encoding and muxing
 * @example transcoding.c
 */

#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>

#define MakeVec(VecName, Type)                                                                     \
  typedef struct VecName {                                                                         \
    Type *data;                                                                                    \
    int reserve, size;                                                                             \
  } VecName;                                                                                       \
                                                                                                   \
  Type *push##Type(VecName *vec, Type element) {                                                   \
    if (!vec)                                                                                      \
      return NULL;                                                                                 \
    if (vec->size >= vec->reserve) {                                                               \
      Type *old = vec->data;                                                                       \
      int old_reserve = vec->reserve;                                                              \
      vec->reserve = 2 * old_reserve + (old_reserve == 0) * 32;                                    \
      vec->data = av_mallocz_array(vec->reserve, sizeof(Type));                                    \
      if (old) {                                                                                   \
        memcpy(vec->data, old, old_reserve * sizeof(Type));                                        \
        av_free(old);                                                                              \
      }                                                                                            \
    }                                                                                              \
    vec->data[vec->size++] = element;                                                              \
  }

typedef struct EditRange {
  double beg, end, ratio;
} EditRange;

typedef struct FrameChange {
  int64_t dur_old;
  int64_t dur_new;
} FrameChange;

MakeVec(EditVector, EditRange);
MakeVec(FrameChangeVector, FrameChange);
MakeVec(PTSVector, int64_t);

typedef struct StreamContext {
  AVCodecContext *dec_ctx;
  AVCodecContext *enc_ctx;
  AVFrame *dec_frame;

  int64_t end_pts;
  int64_t end_dur;
  StreamEditVector sev;
} StreamContext;

static int get_delim(char *buffer, int b_size, int delim, FILE *file);
static int get_line(char *buffer, int b_size, FILE *file);
static int fill_edit_vec(EditVector *edit_vec, char const *filename);
static PTSVector *get_pts_vec();
static int edit();

static int open_input_file(const char *filename);
static int open_output_file(const char *filename);
static int encode_write_frame(AVFrame *frame, AVPacket *enc_pkt, StreamContext *str, int str_idx);
static int flush_encoder(AVPacket *pkt, StreamContext *str, int str_idx);

#define STREAM_DISCARDED -1
#define TS_ACCURACY (1 << 24)

static AVFormatContext *ifmt_ctx;
static AVFormatContext *ofmt_ctx;
static StreamContext *stream_ctx;
static int *stream_mapping = NULL;

int main(int argc, char **argv) {
  EditVector edit_vec = {.data = NULL, .reserve = 0, .size = 0};
  PTSVector *pts_vecs = NULL;
  AVPacket *packet = NULL;
  int ret;

  if (argc != 4) {
    av_log(NULL, AV_LOG_ERROR, "Usage: %s <input file> <output file> <speed ranges file>\n",
           argv[0]);
    return 1;
  }
  if ((ret = open_input_file(argv[1])) < 0)
    goto end;
  if ((ret = open_output_file(argv[2])) < 0)
    goto end;
  if (fill_edit_vec(&edit_vec, argv[3]) < 0)
    goto end;
  if (!(packet = av_packet_alloc()))
    goto end;
  if (!(pts_vecs = get_pts_vec(&edit_vec)))
    goto end;

  while ((ret = av_read_frame(ifmt_ctx, packet)) >= 0) {
    int in_s_idx = packet->stream_index;
    int out_s_idx = stream_mapping[in_s_idx];
    if (out_s_idx == STREAM_DISCARDED)
      continue;

    StreamContext *stream = &stream_ctx[in_s_idx];
    AVRational in_s_timebase = ifmt_ctx->streams[in_s_idx]->time_base;
    AVRational out_s_timebase = ofmt_ctx->streams[out_s_idx]->time_base;

    if (ifmt_ctx->streams[in_s_idx]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      av_packet_rescale_ts(packet, in_s_timebase, stream->dec_ctx->time_base);
      if ((avcodec_send_packet(stream->dec_ctx, packet)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Decoding failed\n");
        break;
      }

      while (ret >= 0) {
        ret = avcodec_receive_frame(stream->dec_ctx, stream->dec_frame);
        if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
          break;
        else if (ret < 0) {
          av_log(NULL, AV_LOG_ERROR, "Decoding failed\n");
          goto end;
        }

        ret = encode_write_frame(stream->dec_frame, packet, stream, out_s_idx);
        av_frame_unref(stream->dec_frame);
        if (ret < 0)
          goto end;
      }
    } else {
      /* remux this frame without reencoding */
      av_packet_rescale_ts(packet, in_s_timebase, out_s_timebase);
      if ((ret = av_interleaved_write_frame(ofmt_ctx, packet)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Muxing error\n");
        goto end;
      }
    }
    av_packet_unref(packet);
  }

  /* flush encoders */
  for (unsigned int i = 0; i < ifmt_ctx->nb_streams; i++)
    if (stream_mapping[i] != STREAM_DISCARDED)
      if ((ret = flush_encoder(packet, &stream_ctx[i], stream_mapping[i])) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Flushing encoder failed\n");
        goto end;
      }

  av_write_trailer(ofmt_ctx);
end:
  av_packet_free(&packet);
  for (unsigned int i = 0; i < ifmt_ctx->nb_streams; i++) {
    avcodec_free_context(&stream_ctx[i].dec_ctx);
    if (ofmt_ctx && i < ofmt_ctx->nb_streams && ofmt_ctx->streams[i] && stream_ctx[i].enc_ctx)
      avcodec_free_context(&stream_ctx[i].enc_ctx);
    av_frame_free(&stream_ctx[i].dec_frame);
  }

  av_free(edit_vec.data);
  for (int i = 0; i < ofmt_ctx->nb_streams; ++i)
    av_free(stream_ctx[i].sev.data);
  av_free(stream_mapping);
  av_free(stream_ctx);
  avformat_close_input(&ifmt_ctx);
  if (ofmt_ctx && !(ofmt_ctx->oformat->flags & AVFMT_NOFILE))
    avio_closep(&ofmt_ctx->pb);
  avformat_free_context(ofmt_ctx);

  if (ret < 0)
    av_log(NULL, AV_LOG_ERROR, "Error occurred: %s\n", av_err2str(ret));

  return ret ? 1 : 0;
}

static int get_delim(char *buffer, int b_size, int delim, FILE *file) {
  int size, c;
  for (size = 0; ((c = fgetc(file)) != EOF && c != delim); ++size)
    if (size < b_size)
      buffer[size] = c;
  if (size < b_size)
    buffer[size] = 0;
  // return_value > b_size when delim wasn't found
  return (size == 0 && c == EOF) ? EOF : size + 1;
}

static int get_line(char *buffer, int b_size, FILE *file) {
  return get_delim(buffer, b_size, '\n', file);
}

static int fill_edit_vec(EditVector *edit_vec, char const *filename) {
  FILE *file = fopen(filename, "r");
  if (!file)
    return -1;

  static char buffer[64];
  int read;

  EditRange range;
  float last_end = 0;
  for (int i = 1; (read = get_line(buffer, sizeof(buffer), file)) != EOF; ++i)
    if (read <= sizeof(buffer) &&
        sscanf(buffer, "%lf %lf %lf", &range.beg, &range.end, &range.ratio) == 3 &&
        range.beg >= 0 && (range.beg < range.end || range.end < 0) && range.beg >= last_end) {
      if (last_end < 0) {
        av_log(NULL, AV_LOG_WARNING, "[%s] Discarding entries after #%u\n", filename, i - 1);
        break;
      }
      pushEditRange(edit_vec, range);
      last_end = range.end;
    } else
      av_log(NULL, AV_LOG_WARNING, "[%s] Bad #%u line\n", filename, i);
  return edit_vec->size ? 0 : -1;
}

static int compare_PTS(void const *l, void const *r) {
  int64_t const *_l = l;
  int64_t const *_r = r;
  return (_l > _r) - (_l < _r);
}

static PTSVector *get_pts_vec(EditVector *edit_vec) {
  int ret;
  AVPacket pkt;
  PTSVector *pts_vecs = av_mallocz_array(ofmt_ctx->nb_streams, sizeof(PTSVector));
  int64_t *last_durs = av_mallocz_array(ofmt_ctx->nb_streams, sizeof(PTSVector));
  if (!pts_vecs)
    return NULL;

  // seek begining
  for (int i = 0; i < ifmt_ctx->nb_streams; ++i)
    if (stream_mapping[i] != STREAM_DISCARDED) {
      av_seek_frame(
          ifmt_ctx, i,
          ifmt_ctx->streams[i]->start_time != AV_NOPTS_VALUE ? ifmt_ctx->streams[i]->start_time : 0,
          AVSEEK_FLAG_BACKWARD);
      break;
    }

  // fill pts vectors of each output stream
  while ((ret = av_read_frame(ifmt_ctx, &pkt)) >= 0) {
    int str_idx = stream_mapping[pkt.stream_index];
    if (str_idx != STREAM_DISCARDED)
      pushint64_t(&pts_vecs[str_idx], pkt.pts);
    av_packet_unref(&pkt);
  }

  // sort PTSs
  for (int i = 0; i < ofmt_ctx->nb_streams; ++i) {
    av_assert0(pts_vecs[i].size > 1);
    qsort(pts_vecs[i].data, pts_vecs[i].size, sizeof(int64_t), compare_PTS);
  }

  // create edit vectors for each output stream
  StreamEditVector *sevs = av_mallocz_array(ofmt_ctx->nb_streams, sizeof(StreamEditVector));
  for (int in_str_idx = 0; in_str_idx < ifmt_ctx->nb_streams; ++in_str_idx) {
    int const str_idx = stream_mapping[in_str_idx];
    if (str_idx == STREAM_DISCARDED)
      continue;

    PTSVector const pv = pts_vecs[str_idx];
    int64_t const tb_num = ifmt_ctx->streams[in_str_idx]->time_base.num;
    int64_t const tb_den = ifmt_ctx->streams[in_str_idx]->time_base.den;
    int64_t const avg_dur = (pv.data[pv.size - 1] - pv.data[0]) / pv.size;

    // calculate durations
    for (int i = pts_vecs[str_idx].size; i >= 1; --i)
      pts_vecs[str_idx].data[i] -= pts_vecs[str_idx].data[i - 1];

    for (int ev_idx = 0, pk_idx = 0; ev_idx < edit_vec->size; ++ev_idx) {
      EditRange const er = edit_vec->data[ev_idx];
      // from time [s] to this stream's timebase units
      AVRational const beg_q = av_d2q(er.beg, TS_ACCURACY);
      AVRational const end_q = av_d2q(er.end, TS_ACCURACY);
      int64_t const beg_ts = av_rescale(beg_q.num, tb_den, beg_q.den * tb_num);
      int64_t const end_ts =
          er.end >= 0 ? av_rescale(end_q.num, tb_den, end_q.den * tb_num) : INT64_MAX;
      StreamEditRange ser;
      // find edit range begin
      for (pk_idx; pk_idx < pv.size; ++pk_idx) {
        int64_t pts = pv.data[pk_idx];
        if (pts >= beg_ts) {
          ser.beg = (beg_ts == pts || pk_idx - 1 < 0) ? pts : pv.data[pk_idx - 1];
          break;
        }
      }
      // find edit range end
      for (pk_idx; pk_idx < pv.size; ++pk_idx) {
        int64_t pts = pv.data[pk_idx];
        if (pts >= end_ts) {
          ser.end = (pts == beg_ts || pk_idx + 1 >= pv.size) ? pts : pv.data[pk_idx + 1];
          break;
        }
      }
      // correct the ratio to account for extended time
      //    1   2   3   4   5   6   7   8
      // ---|---|---|---|---|---|---|---|---
      //      |           |
      //      B           E
      //    |---------------|
      // co jeśli rozszerzone zakresy nachodzą na siebie - trzeba update'ować tutaj na bierząco ptsy
      AVRational ratio_q = av_d2q(er.ratio, TS_ACCURACY);
      int64_t dur_q = av_rescale_rnd(end_ts - beg_ts, ratio_q.num, ratio_q.den, AV_ROUND_NEAR_INF);
      int64_t ser_dur_now = ser.end - ser.beg;
      int64_t ser_dur_after = (beg_ts - ser.beg) + dur_q + (ser.end - end_ts);
      // ser_dur_after = ser_dur_now * ser.ratio
      ser.ratio = av_d2q(ser_dur_after / (double)ser_dur_now, TS_ACCURACY);
      pushStreamEditRange(&sevs[stream_mapping[str_idx]], ser);
    }
  }

  av_free(pts_vecs);
}

static int open_input_file(const char *filename) {
  int ret;
  unsigned int i;

  ifmt_ctx = NULL;
  if ((ret = avformat_open_input(&ifmt_ctx, filename, NULL, NULL)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot open input file\n");
    return ret;
  }

  if ((ret = avformat_find_stream_info(ifmt_ctx, NULL)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot find stream information\n");
    return ret;
  }

  if (!(stream_ctx = av_mallocz_array(ifmt_ctx->nb_streams, sizeof(*stream_ctx)))) {
    av_log(NULL, AV_LOG_ERROR, "Failed to allocate stream contexts\n");
    return AVERROR(ENOMEM);
  }

  for (i = 0; i < ifmt_ctx->nb_streams; i++) {
    if (ifmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
      continue;

    AVStream *stream = ifmt_ctx->streams[i];
    AVCodec *dec;
    AVCodecContext *codec_ctx;

    if (!(dec = avcodec_find_decoder(stream->codecpar->codec_id))) {
      av_log(NULL, AV_LOG_ERROR, "Failed to find decoder for stream #%u\n", i);
      return AVERROR_DECODER_NOT_FOUND;
    }
    if (!(codec_ctx = avcodec_alloc_context3(dec))) {
      av_log(NULL, AV_LOG_ERROR, "Failed to allocate the decoder context for stream #%u\n", i);
      return AVERROR(ENOMEM);
    }
    if ((ret = avcodec_parameters_to_context(codec_ctx, stream->codecpar)) < 0) {
      av_log(NULL, AV_LOG_ERROR,
             "Failed to copy decoder parameters to input decoder context for stream #%u\n", i);
      return ret;
    }

    if (codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
      codec_ctx->framerate = av_guess_frame_rate(ifmt_ctx, stream, NULL);
      if ((ret = avcodec_open2(codec_ctx, dec, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Failed to open decoder for stream #%u\n", i);
        return ret;
      }
    }
    stream_ctx[i].dec_ctx = codec_ctx;
    if (!(stream_ctx[i].dec_frame = av_frame_alloc())) {
      av_log(NULL, AV_LOG_ERROR, "Failed to allocate memory for decoded frame for stream #%u\n", i);
      return AVERROR(ENOMEM);
    }
  }

  av_dump_format(ifmt_ctx, 0, filename, 0);
  return 0;
}

static int open_output_file(const char *filename) {
  AVStream *out_stream;
  AVStream *in_stream;
  AVCodecContext *dec_ctx, *enc_ctx;
  AVCodec *encoder;
  int ret;
  unsigned int out_streams_num = 0;

  ofmt_ctx = NULL;
  avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, filename);
  if (!ofmt_ctx) {
    av_log(NULL, AV_LOG_ERROR, "Could not create output context\n");
    return AVERROR_UNKNOWN;
  }

  if (!(stream_mapping = av_mallocz_array(ifmt_ctx->nb_streams, sizeof(*stream_mapping)))) {
    av_log(NULL, AV_LOG_ERROR, "Failed to allocate input to output streams mapping\n");
    return AVERROR(ENOMEM);
  }

  for (unsigned int i = 0; i < ifmt_ctx->nb_streams; i++) {
    if (ifmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      stream_mapping[i] = STREAM_DISCARDED;
      continue;
    }
    stream_mapping[i] = out_streams_num++;

    if (!(out_stream = avformat_new_stream(ofmt_ctx, NULL))) {
      av_log(NULL, AV_LOG_ERROR, "Failed allocating output stream\n");
      return AVERROR_UNKNOWN;
    }

    in_stream = ifmt_ctx->streams[i];
    dec_ctx = stream_ctx[i].dec_ctx;

    if (dec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
      if (!(encoder = avcodec_find_encoder(dec_ctx->codec_id))) { // transcode to same codec
        av_log(NULL, AV_LOG_FATAL, "Necessary encoder not found\n");
        return AVERROR_INVALIDDATA;
      }
      if (!(enc_ctx = avcodec_alloc_context3(encoder))) {
        av_log(NULL, AV_LOG_FATAL, "Failed to allocate the encoder context\n");
        return AVERROR(ENOMEM);
      }

      // enc_ctx->framerate = dec_ctx->framerate;
      enc_ctx->height = dec_ctx->height;
      enc_ctx->width = dec_ctx->width;
      enc_ctx->sample_aspect_ratio = dec_ctx->sample_aspect_ratio;
      if (encoder->pix_fmts)
        enc_ctx->pix_fmt = encoder->pix_fmts[0]; // first format from list of supported formats
      else
        enc_ctx->pix_fmt = dec_ctx->pix_fmt;
      enc_ctx->time_base = dec_ctx->time_base;

      if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

      if ((ret = avcodec_open2(enc_ctx, encoder, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open video encoder for stream #%u\n", i);
        return ret;
      }

      if ((ret = avcodec_parameters_from_context(out_stream->codecpar, enc_ctx)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Failed to copy encoder parameters to output stream #%u\n", i);
        return ret;
      }

      out_stream->time_base = enc_ctx->time_base;
      stream_ctx[i].enc_ctx = enc_ctx;
    } else if (dec_ctx->codec_type == AVMEDIA_TYPE_UNKNOWN)
      av_log(NULL, AV_LOG_WARNING, "Elementary stream #%d is of unknown type, skipping\n", i);
    else {
      if ((avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Copying parameters for stream #%u failed\n", i);
        return ret;
      }
      out_stream->time_base = in_stream->time_base;
    }
  }

  av_dump_format(ofmt_ctx, 0, filename, 1);
  if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE))
    if ((avio_open(&ofmt_ctx->pb, filename, AVIO_FLAG_WRITE)) < 0) {
      av_log(NULL, AV_LOG_ERROR, "Could not open output file '%s'", filename);
      return ret;
    }

  if ((ret = avformat_write_header(ofmt_ctx, NULL)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Error occurred when opening output file\n");
    return ret;
  }

  return 0;
}

static int encode_write_frame(AVFrame *frame, AVPacket *enc_pkt, StreamContext *str, int str_idx) {
  int ret;

  av_packet_unref(enc_pkt);
  if (frame) {
    frame->pict_type = AV_PICTURE_TYPE_NONE;
    frame->pts = av_rescale_q_rnd(frame->pts, str->dec_ctx->time_base, str->enc_ctx->time_base,
                                  AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
  }
  if ((ret = avcodec_send_frame(str->enc_ctx, frame)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Encoding error\n");
    return ret;
  }

  while (ret >= 0) {
    ret = avcodec_receive_packet(str->enc_ctx, enc_pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return 0;

    /* prepare packet for muxing */
    enc_pkt->stream_index = str_idx;
    av_packet_rescale_ts(enc_pkt, str->enc_ctx->time_base, ofmt_ctx->streams[str_idx]->time_base);
    // enc_pkt->pts /= 3;
    // enc_pkt->dts /= 3;
    // enc_pkt->duration /= 3;

    av_log(NULL, AV_LOG_DEBUG, "Muxing frame\n");
    ret = av_interleaved_write_frame(ofmt_ctx, enc_pkt);
  }

  return ret;
}

static int flush_encoder(AVPacket *pkt, StreamContext *str, int str_idx) {
  if (!(str->enc_ctx->codec->capabilities & AV_CODEC_CAP_DELAY))
    return 0;

  av_log(NULL, AV_LOG_INFO, "Flushing stream #%u encoder\n", str_idx);
  return encode_write_frame(NULL, pkt, str, str_idx);
}
