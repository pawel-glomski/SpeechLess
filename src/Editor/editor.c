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

#include <ctype.h>
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/opt.h>

#define MakeVec(VecName, Type)                                                                     \
  typedef struct VecName {                                                                         \
    Type* data;                                                                                    \
    int reserve, size;                                                                             \
  } VecName;                                                                                       \
                                                                                                   \
  void pushTo##VecName(VecName* vec, Type element);                                                \
  void pushTo##VecName(VecName* vec, Type element)                                                 \
  {                                                                                                \
    if (!vec)                                                                                      \
      return;                                                                                      \
    if (vec->size >= vec->reserve) {                                                               \
      Type* old = vec->data;                                                                       \
      int old_reserve = vec->reserve;                                                              \
      vec->reserve = 2 * old_reserve + (old_reserve == 0) * 8;                                     \
      vec->data = av_malloc_array(vec->reserve, sizeof(Type));                                     \
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

MakeVec(EditRanges, EditRange);
MakeVec(Times, double);

typedef struct InData {
  AVCodecContext* dec_ctx;

  double time_base;
  double avg_dur;
  double end_t;
  Times p_times;
} InData;

typedef struct OutData {
  AVCodecContext* enc_ctx;

  uint32_t width, height;
  AVRational max_fps;
  enum AVCodecID codec_id;
} OutData;

typedef struct EditingData {
  AVFrame* frame;

  Times durs;
  double last_t;
  int64_t pkt_counter;
  int done;
} EditingData;

typedef struct StreamContext {
  InData in;
  OutData out;
  EditingData editing;
} StreamContext;

/*************************************************************************************************/

static char const EDIT_SPEC_DELIM = ';';
static char const* const EDIT_KEY_DELIM = ":";
static char const EDIT_WHITESPACE = ' ';
static int const EDIT_RANGE_TO_END = -1;

static char const* const RES_KEY = "resolution";
static char const* const CODEC_KEY = "codec";
static char const* const MAX_FPS_KEY = "max_fps";

static int const STREAM_DISCARDED = -1;
static double const DROP_REST_DURATION = -1;
static double const DEFAULT_MAX_FPS = 120.0;
static size_t const ACCURACY = 1 << 24;

static AVFormatContext* ifmt_ctx;
static AVFormatContext* ofmt_ctx;
static StreamContext* stream_ctx;
static int* stream_mapping = NULL;

static EditRanges edit_ranges = { .data = NULL, .reserve = 0, .size = 0 };

/*************************************************************************************************/

// Consumes and stores characters from `file` up to `delim`, which is consumed, but not stored.
// `buffer` should be able to store `b_size` - 1 characters from the file and '\0' at the end.
// Only one whitespace character in a row is stored, the rest is discarded.
// `return value` >= b_size when delim wasn't found and buffer is full
static int get_delim(char* buffer, int b_size, int delim, FILE* file)
{
  av_assert1(b_size > 0);

  int size = 0, c;
  // consume to delim, even when cannot store, store only 1 whitespace in a row
  while ((c = fgetc(file)) != EOF && c != delim)
    if (size < b_size && (!isspace(c) || (size > 0 && !isspace(buffer[size - 1]))))
      buffer[size++] = isspace(c) ? EDIT_WHITESPACE : c;
  buffer[size < b_size ? size : b_size - 1] = '\0';
  int test = isspace('\n');
  return (size == 0 && c == EOF) ? EOF : size;
}

static int get_line(char* buffer, int b_size, FILE* file)
{
  return get_delim(buffer, b_size, '\n', file);
}

static int parse_edit_resolution(char const* value)
{
  size_t stream_idx;
  size_t width, height;
  int pos;

  // update specified
  while (sscanf(value, "%zu : %zu x %zu %n", &stream_idx, &width, &height, &pos) == 3) {
    if (width * height > 0 && stream_idx < ifmt_ctx->nb_streams
        && stream_mapping[stream_idx] != STREAM_DISCARDED) {
      stream_ctx[stream_idx].out.width = width;
      stream_ctx[stream_idx].out.height = height;
    } else
      av_log(NULL, AV_LOG_WARNING, "SPEC: Bad '%s' specification: '%zu: %zu x %zu' \n", RES_KEY,
          stream_idx, width, height);
    value += pos;
  }
  return -strlen(value);
}

static int parse_edit_codec(char const* value)
{
  size_t stream_idx;
  char name[64];
  int pos;

  // update specified
  while (sscanf(value, "%zu : %63s %n", &stream_idx, name, &pos) == 2) {
    AVCodecDescriptor const* desc = avcodec_descriptor_get_by_name(name);
    if (desc && stream_idx < ifmt_ctx->nb_streams)
      stream_ctx[stream_idx].out.codec_id = desc->id;
    else
      av_log(NULL, AV_LOG_WARNING, "SPEC: Bad '%s' specification: '%zu: %s' \n", CODEC_KEY,
          stream_idx, name);
    value += pos;
  }
  return -strlen(value);
}

static int parse_edit_fps(char const* value)
{
  size_t stream_idx;
  double max_fps;
  int pos;

  // update specified
  while (sscanf(value, "%zu : %lf %n", &stream_idx, &max_fps, &pos) == 2) {
    if (max_fps > 0 && stream_idx < ifmt_ctx->nb_streams)
      stream_ctx[stream_idx].out.max_fps = av_d2q(max_fps, ACCURACY);
    else
      av_log(NULL, AV_LOG_WARNING, "SPEC: Bad '%s' specification: (%zu: %lf) \n", MAX_FPS_KEY,
          stream_idx, max_fps);
    value += pos;
  }
  return -strlen(value);
}

static int fill_edit_specs(char const* filename)
{
  FILE* file = fopen(filename, "r");
  if (!file)
    return -1;

  static char buffer[256];
  char const* value;
  char const* key;
  int read;

  // resolution: 'same' by default
  for (size_t i = 0; i < ifmt_ctx->nb_streams; ++i)
    if (stream_mapping[i] != STREAM_DISCARDED) {
      stream_ctx[i].out.width = stream_ctx[i].in.dec_ctx->width;
      stream_ctx[i].out.height = stream_ctx[i].in.dec_ctx->height;
    }

  // codec: 'same' by default
  for (size_t i = 0; i < ifmt_ctx->nb_streams; ++i)
    if (stream_mapping[i] != STREAM_DISCARDED)
      stream_ctx[i].out.codec_id = stream_ctx[i].in.dec_ctx->codec_id;

  // max_fps: set default
  for (size_t i = 0; i < ifmt_ctx->nb_streams; ++i)
    if (stream_mapping[i] != STREAM_DISCARDED)
      stream_ctx[i].out.max_fps = av_d2q(DEFAULT_MAX_FPS, ACCURACY);

  // read specs
  while ((read = get_delim(buffer, sizeof(buffer), EDIT_SPEC_DELIM, file)) > 0)
    if (read < sizeof(buffer)) {
      key = buffer;
      value = strpbrk(buffer, EDIT_KEY_DELIM);
      if (value) {
        buffer[value++ - buffer] = '\0';

        int ret = 0;
        if (strcmp(key, RES_KEY) == 0)
          ret = parse_edit_resolution(value);
        else if (strcmp(key, CODEC_KEY) == 0)
          ret = parse_edit_codec(value);
        else if (strcmp(key, MAX_FPS_KEY) == 0)
          ret = parse_edit_fps(value);
        else
          av_log(NULL, AV_LOG_WARNING, "SPEC: unknown key - '%s' \n", key);
        if (ret < 0)
          av_log(NULL, AV_LOG_WARNING, "SPEC: value error '%s:%s'\n", key, value);
      } else
        av_log(NULL, AV_LOG_WARNING, "SPEC: format error '%s'\n", buffer);
    } else
      av_log(NULL, AV_LOG_WARNING, "SPEC: too long '%s...'\n", buffer);

  // read ranges
  EditRange range;
  float last_end = 0;
  while ((read = get_line(buffer, sizeof(buffer), file)) != EOF)
    if (read < sizeof(buffer)
        && sscanf(buffer, "%lf %lf %lf", &range.beg, &range.end, &range.ratio) == 3
        && range.beg >= 0 && (range.beg < range.end || range.end < 0) && range.beg >= last_end) {
      pushToEditRanges(&edit_ranges, range);
      last_end = range.end;
      if (last_end == EDIT_RANGE_TO_END) {
        av_log(NULL, AV_LOG_WARNING, "RANGE: ending at '%s'\n", buffer);
        break;
      }
    } else if (read > 0)
      av_log(NULL, AV_LOG_WARNING, "RANGE: bad format for '%s' \n", buffer);

  fclose(file);
  return edit_ranges.size ? 0 : -1;
}

static int compare_time(void const* l, void const* r)
{
  double const _l = *(double*)l;
  double const _r = *(double*)r;
  return (_l > _r) - (_l < _r);
}

static EditRange fixEditRange(EditRange er, StreamContext* ctx)
{
  er.end = er.end > 0 ? er.end : ctx->in.end_t;
  return er;
}

static int fill_editing_durs()
{
  AVPacket* pkt;
  int ret = 0;
  if (!(pkt = av_packet_alloc()))
    return -1;

  // seek begining
  for (int i = 0; i < ifmt_ctx->nb_streams; ++i)
    if (stream_mapping[i] != STREAM_DISCARDED) {
      av_seek_frame(ifmt_ctx, i,
          ifmt_ctx->streams[i]->start_time != AV_NOPTS_VALUE ? ifmt_ctx->streams[i]->start_time : 0,
          AVSEEK_FLAG_BACKWARD);
      break;
    }

  // fill pts vectors of each output stream
  while ((ret = av_read_frame(ifmt_ctx, pkt)) >= 0) {
    if (stream_mapping[pkt->stream_index] != STREAM_DISCARDED) {
      av_assert0(pkt->pts != AV_NOPTS_VALUE);
      StreamContext* ctx = &stream_ctx[pkt->stream_index];
      double pkt_t = ctx->in.time_base * pkt->pts;
      double pkt_end_t = pkt_t + ctx->in.time_base * pkt->duration;
      pushToTimes(&ctx->in.p_times, pkt_t);
      if (pkt_end_t > ctx->in.end_t)
        ctx->in.end_t = pkt_end_t;
    }
    av_packet_unref(pkt);
  }

  // sort times
  for (int i = 0; i < ifmt_ctx->nb_streams; ++i)
    if (stream_mapping[i] != STREAM_DISCARDED) {
      Times const p_times = stream_ctx[i].in.p_times;
      av_assert0(p_times.size > 1);
      qsort(p_times.data, p_times.size, sizeof(double), compare_time);
      stream_ctx[i].in.avg_dur = (p_times.data[p_times.size - 1] - p_times.data[0]) / p_times.size;

      // fix in.end_t
      if (p_times.data[p_times.size - 1] >= stream_ctx[i].in.end_t)
        stream_ctx[i].in.end_t = p_times.data[p_times.size - 1] + stream_ctx[i].in.avg_dur;
    }

  // calculate new durations of packets for each output stream
  for (int in_str_idx = 0; in_str_idx < ifmt_ctx->nb_streams; ++in_str_idx) {
    if (stream_mapping[in_str_idx] == STREAM_DISCARDED)
      continue;
    StreamContext* ctx = &stream_ctx[in_str_idx];
    Times const p_times = ctx->in.p_times;
    EditRange range = fixEditRange(edit_ranges.data[0], ctx);

    for (int idx_ev = 0, idx_t = 0; idx_t < p_times.size; ++idx_t) {
      double const time = p_times.data[idx_t];
      double const next_t = idx_t + 1 < p_times.size ? p_times.data[idx_t + 1] : ctx->in.end_t;
      double old_dur = next_t - time;
      double new_dur = 0;

      while (range.beg < next_t && idx_ev < edit_ranges.size) {
        av_assert0(range.beg >= time);
        double clamped_end = range.end < next_t ? range.end : next_t;
        double dur_part = clamped_end - range.beg;
        av_assert0(old_dur >= dur_part);
        old_dur -= dur_part;
        new_dur += dur_part * range.ratio;

        if (clamped_end < range.end)
          range.beg = next_t;
        else
          range = fixEditRange(edit_ranges.data[++idx_ev], ctx);
      }
      new_dur += old_dur;
      if (new_dur == 0 && range.end == ctx->in.end_t) {
        pushToTimes(&ctx->editing.durs, DROP_REST_DURATION);
        break;
      }
      pushToTimes(&ctx->editing.durs, new_dur);
    }

    // collapse too small frames
    double const minimal_dur = av_q2d(av_inv_q(ctx->out.max_fps));
    int64_t last_nz_idx = -1;
    for (int i = 0; i < ctx->editing.durs.size; ++i) {
      if (ctx->editing.durs.data[i] <= 0)
        continue;
      if (last_nz_idx < 0)
        last_nz_idx = i;
      else if (ctx->editing.durs.data[i] < minimal_dur) {
        ctx->editing.durs.data[last_nz_idx] += ctx->editing.durs.data[i];
        ctx->editing.durs.data[i] = 0;
        if (ctx->editing.durs.data[last_nz_idx] >= minimal_dur) {
          double surplus = ctx->editing.durs.data[last_nz_idx] - minimal_dur;
          // move the surplus duration to the current frame
          ctx->editing.durs.data[last_nz_idx] = minimal_dur;
          ctx->editing.durs.data[i] = surplus;
          last_nz_idx = surplus > 0.0 ? i : -1;
        }
      }
    }
    if (last_nz_idx != -1)
      if (ctx->editing.durs.data[last_nz_idx] / minimal_dur < 0.5)
        ctx->editing.durs.data[last_nz_idx] = 0;
      else
        ctx->editing.durs.data[last_nz_idx] = minimal_dur;
  }
  av_packet_free(&pkt);
  return ret;
}

static int open_input_file(const char* filename)
{
  int ret;

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

  if (!(stream_mapping = av_mallocz_array(ifmt_ctx->nb_streams, sizeof(*stream_mapping)))) {
    av_log(NULL, AV_LOG_ERROR, "Failed to allocate input to output streams mapping\n");
    return AVERROR(ENOMEM);
  }

  for (size_t out_streams_num = 0, i = 0; i < ifmt_ctx->nb_streams; i++) {
    if (ifmt_ctx->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
      av_log(NULL, AV_LOG_WARNING, "Skipping stream #%zu\n", i);
      stream_mapping[i] = STREAM_DISCARDED;
      continue;
    }
    stream_mapping[i] = out_streams_num++;

    AVStream* stream = ifmt_ctx->streams[i];
    AVCodec* dec;
    AVCodecContext* codec_ctx;

    if (!(dec = avcodec_find_decoder(stream->codecpar->codec_id))) {
      av_log(NULL, AV_LOG_ERROR, "Failed to find decoder for stream %zu\n", i);
      return AVERROR_DECODER_NOT_FOUND;
    }
    if (!(codec_ctx = avcodec_alloc_context3(dec))) {
      av_log(NULL, AV_LOG_ERROR, "Failed to allocate the decoder context for stream #%zu\n", i);
      return AVERROR(ENOMEM);
    }
    if ((ret = avcodec_parameters_to_context(codec_ctx, stream->codecpar)) < 0) {
      av_log(NULL, AV_LOG_ERROR,
          "Failed to copy decoder parameters to input decoder context for stream %zu\n", i);
      return ret;
    }

    codec_ctx->framerate = av_guess_frame_rate(ifmt_ctx, stream, NULL);
    if ((ret = avcodec_open2(codec_ctx, dec, NULL)) < 0) {
      av_log(NULL, AV_LOG_ERROR, "Failed to open decoder for stream %zu\n", i);
      return ret;
    }

    stream_ctx[i].in.dec_ctx = codec_ctx;
    stream_ctx[i].in.time_base = av_q2d(stream->time_base);
  }

  av_dump_format(ifmt_ctx, 0, filename, 0);
  return 0;
}

// must be called after open_input_file and fill_edit_specs
static int open_output_file(const char* filename)
{
  int ret;

  ofmt_ctx = NULL;
  if ((ret = avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, filename)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Could not create output context\n");
    return ret;
  }

  for (size_t i = 0; i < ifmt_ctx->nb_streams; i++) {
    if (stream_mapping[i] == STREAM_DISCARDED)
      continue;

    StreamContext const* ctx = &stream_ctx[i];
    AVStream* out_stream;
    AVCodec* encoder;
    AVCodecContext* enc_ctx;

    if (!(stream_ctx[i].editing.frame = av_frame_alloc())) {
      av_log(NULL, AV_LOG_ERROR, "Failed to allocate memory frame for stream %zu\n", i);
      return AVERROR(ENOMEM);
    }
    if (!(out_stream = avformat_new_stream(ofmt_ctx, NULL))) {
      av_log(NULL, AV_LOG_ERROR, "Failed allocating output stream\n");
      return AVERROR_UNKNOWN;
    }

    ctx->out.codec_id;

    if (!(encoder = avcodec_find_encoder(ctx->out.codec_id))) {
      av_log(NULL, AV_LOG_FATAL, "Necessary encoder not found\n");
      return AVERROR_INVALIDDATA;
    }
    if (!(enc_ctx = avcodec_alloc_context3(encoder))) {
      av_log(NULL, AV_LOG_FATAL, "Failed to allocate the encoder context\n");
      return AVERROR(ENOMEM);
    }

    // TODO: add rescaling
    // enc_ctx->width = ctx->out.width;
    // enc_ctx->height = ctx->out.height;
    // enc_ctx->sample_aspect_ratio = (AVRational) { enc_ctx->width, enc_ctx->height };
    enc_ctx->width = ctx->in.dec_ctx->width;
    enc_ctx->height = ctx->in.dec_ctx->height;
    enc_ctx->sample_aspect_ratio = ctx->in.dec_ctx->sample_aspect_ratio;
    if (encoder->pix_fmts)
      enc_ctx->pix_fmt = encoder->pix_fmts[0]; // first format from list of supported formats
    else
      enc_ctx->pix_fmt = ctx->in.dec_ctx->pix_fmt;

    enc_ctx->time_base = (AVRational) { 1, 60000 }; // as high resolution as possible
    out_stream->time_base = (AVRational) { 1, 60000 };
    // enc_ctx->editing.framerate = (AVRational) { 200, 1 }; do not set
    // out_stream->avg_frame_rate = ctx->out.max_fps;

    // same quality for now
    // enc_ctx->compression_level = x;
    // enc_ctx->global_quality = x;
    // enc_ctx->level = x;
    // av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);

    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
      enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if ((ret = avcodec_open2(enc_ctx, encoder, NULL)) < 0) {
      av_log(NULL, AV_LOG_ERROR, "Cannot open video encoder for stream %zu\n", i);
      return ret;
    }
    if ((ret = avcodec_parameters_from_context(out_stream->codecpar, enc_ctx)) < 0) {
      av_log(NULL, AV_LOG_ERROR, "Failed to copy encoder parameters to output stream %zu\n", i);
      return ret;
    }

    stream_ctx[i].out.enc_ctx = enc_ctx;
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

static int encode_write_frame(AVFrame* frame, AVPacket* enc_pkt, StreamContext* ctx, int str_idx)
{
  int ret;

  av_packet_unref(enc_pkt);
  if (frame) {
    av_assert0(ctx->editing.pkt_counter < ctx->editing.durs.size);
    int64_t pkt_idx = ctx->editing.pkt_counter++;
    if (ctx->editing.durs.data[pkt_idx] <= 0) { // drop
      if (ctx->editing.durs.data[pkt_idx] == DROP_REST_DURATION) // end encoding
        ctx->editing.done = 1;
      return 0;
    }
    frame->pict_type = AV_PICTURE_TYPE_NONE;
    frame->pts = (int64_t)(ctx->editing.last_t * ACCURACY + 0.5);
    frame->pts = av_rescale_q_rnd(frame->pts, (AVRational) { 1, ACCURACY },
        ctx->out.enc_ctx->time_base, AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
    ctx->editing.last_t += ctx->editing.durs.data[pkt_idx];
  }
  if ((ret = avcodec_send_frame(ctx->out.enc_ctx, frame)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Encoding error\n");
    return ret;
  }

  while (ret >= 0) {
    ret = avcodec_receive_packet(ctx->out.enc_ctx, enc_pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return 0;

    //  prepare packet for muxing
    enc_pkt->stream_index = str_idx;
    av_packet_rescale_ts(
        enc_pkt, ctx->out.enc_ctx->time_base, ofmt_ctx->streams[str_idx]->time_base);
    ret = av_interleaved_write_frame(ofmt_ctx, enc_pkt);
  }

  return ret;
}

static int flush_encoder(AVPacket* pkt, StreamContext* ctx, size_t str_idx)
{
  if (!(ctx->out.enc_ctx->codec->capabilities & AV_CODEC_CAP_DELAY))
    return 0;

  av_log(NULL, AV_LOG_INFO, "Flushing stream #%zu encoder\n", str_idx);
  return encode_write_frame(NULL, pkt, ctx, str_idx);
}

/*************************************************************************************************/

int main(int argc, char** argv)
{
  AVPacket* packet = NULL;
  int ret;

  if (argc != 4) {
    av_log(
        NULL, AV_LOG_ERROR, "Usage: %s <input file> <speed ranges file> <output file> \n", argv[0]);
    return 1;
  }
  if ((ret = open_input_file(argv[1])) < 0)
    goto end;
  if (fill_edit_specs(argv[2]) < 0) {
    av_log(NULL, AV_LOG_WARNING, "No speed ranges in %s\n", argv[2]);
    goto end;
  }
  if ((ret = open_output_file(argv[3])) < 0)
    goto end;
  if (!(packet = av_packet_alloc()))
    goto end;
  if (!(ret = fill_editing_durs()))
    goto end;

  // seek begining
  for (size_t i = 0; i < ifmt_ctx->nb_streams; ++i)
    if (stream_mapping[i] != STREAM_DISCARDED) {
      av_seek_frame(ifmt_ctx, i,
          ifmt_ctx->streams[i]->start_time != AV_NOPTS_VALUE ? ifmt_ctx->streams[i]->start_time : 0,
          AVSEEK_FLAG_BACKWARD);
      break;
    }

  while ((ret = av_read_frame(ifmt_ctx, packet)) >= 0) {
    int in_s_idx = packet->stream_index;
    int out_s_idx = stream_mapping[in_s_idx];
    if (out_s_idx == STREAM_DISCARDED)
      continue;

    StreamContext* ctx = &stream_ctx[in_s_idx];
    AVRational in_s_timebase = ifmt_ctx->streams[in_s_idx]->time_base;
    AVRational out_s_timebase = ofmt_ctx->streams[out_s_idx]->time_base;

    if (ctx->editing.done)
      continue; // do not break, other streams might not be done

    av_packet_rescale_ts(packet, in_s_timebase, ctx->in.dec_ctx->time_base);
    if ((avcodec_send_packet(ctx->in.dec_ctx, packet)) < 0) {
      av_log(NULL, AV_LOG_ERROR, "Decoding failed\n");
      break;
    }

    while (ret >= 0) {
      ret = avcodec_receive_frame(ctx->in.dec_ctx, ctx->editing.frame);
      if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
        break;
      else if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Decoding failed\n");
        goto end;
      }

      ret = encode_write_frame(ctx->editing.frame, packet, ctx, out_s_idx);
      av_frame_unref(ctx->editing.frame);
      if (ret < 0)
        goto end;
    }
    av_packet_unref(packet);
  }

  /* flush encoders */
  for (size_t i = 0; i < ifmt_ctx->nb_streams; i++)
    if (stream_mapping[i] != STREAM_DISCARDED)
      if ((ret = flush_encoder(packet, &stream_ctx[i], stream_mapping[i])) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Flushing encoder failed\n");
        goto end;
      }

  av_write_trailer(ofmt_ctx);
end:
  av_packet_free(&packet);
  for (size_t i = 0; ifmt_ctx && i < ifmt_ctx->nb_streams; i++) {
    avcodec_free_context(&stream_ctx[i].in.dec_ctx);
    if (ofmt_ctx && i < ofmt_ctx->nb_streams && ofmt_ctx->streams[i] && stream_ctx[i].out.enc_ctx)
      avcodec_free_context(&stream_ctx[i].out.enc_ctx);
    av_frame_free(&stream_ctx[i].editing.frame);
  }

  av_free(edit_ranges.data);
  if (stream_mapping)
    for (size_t i = 0; i < ifmt_ctx->nb_streams; ++i)
      if (stream_mapping[i] != STREAM_DISCARDED) {
        av_free(stream_ctx[stream_mapping[i]].in.p_times.data);
        av_free(stream_ctx[stream_mapping[i]].editing.durs.data);
      }
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
