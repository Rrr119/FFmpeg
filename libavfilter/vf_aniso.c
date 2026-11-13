/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <math.h>

#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "video.h"

typedef struct AnisoContext {
    const AVClass *class;

    int iterations;
    float kappa;
    float lambda;
    int reset;
    int planes;

    int depth;
    int max;
    int nb_planes;
    int planewidth[4];
    int planeheight[4];

    float scale;
    float inv_scale;
    float inv_kappa_sq;

    float *state[4];
    float *next[4];
    int first_frame;
} AnisoContext;

#define OFFSET(x) offsetof(AnisoContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption aniso_options[] = {
    { "iterations", "set number of iterations", OFFSET(iterations), AV_OPT_TYPE_INT,   {.i64=5},     1, 100, FLAGS },
    { "kappa",      "set edge threshold",        OFFSET(kappa),      AV_OPT_TYPE_FLOAT, {.dbl=10.0}, 0.001, 255, FLAGS },
    { "lambda",     "set time step",             OFFSET(lambda),     AV_OPT_TYPE_FLOAT, {.dbl=0.2},  0.001, 0.25, FLAGS },
    { "timestep",   "set time step",             OFFSET(lambda),     AV_OPT_TYPE_FLOAT, {.dbl=0.2},  0.001, 0.25, FLAGS },
    { "reset",      "reset the diffusion state for every frame", OFFSET(reset), AV_OPT_TYPE_BOOL, {.i64=1}, 0, 1, FLAGS },
    { "planes",     "set planes to filter",      OFFSET(planes),     AV_OPT_TYPE_INT,   {.i64=1},     0, 0xF, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(aniso);

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY10, AV_PIX_FMT_GRAY12,
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV444P10,
    AV_PIX_FMT_YUV420P12, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV444P12,
    AV_PIX_FMT_NONE
};

static void free_planes(AnisoContext *s)
{
    for (int p = 0; p < 4; p++) {
        av_freep(&s->state[p]);
        av_freep(&s->next[p]);
    }
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    AnisoContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    free_planes(s);

    s->depth = desc->comp[0].depth;
    s->max = (1 << s->depth) - 1;
    s->scale = 255.0f / s->max;
    s->inv_scale = s->max / 255.0f;
    s->inv_kappa_sq = 1.0f / (s->kappa * s->kappa);

    s->planewidth[0] = s->planewidth[3] = inlink->w;
    s->planewidth[1] = s->planewidth[2] =
        AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planeheight[0] = s->planeheight[3] = inlink->h;
    s->planeheight[1] = s->planeheight[2] =
        AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->nb_planes = av_pix_fmt_count_planes(inlink->format);

    for (int p = 0; p < s->nb_planes; p++) {
        size_t size;

        if (!(s->planes & (1 << p)))
            continue;

        size = (size_t)s->planewidth[p] * s->planeheight[p];
        s->state[p] = av_malloc_array(size, sizeof(*s->state[p]));
        s->next[p]  = av_malloc_array(size, sizeof(*s->next[p]));
        if (!s->state[p] || !s->next[p]) {
            free_planes(s);
            return AVERROR(ENOMEM);
        }
    }

    s->first_frame = 1;
    return 0;
}

static void import_plane8(const AVFrame *in, int plane, int width, int height,
                          float scale, float *dst)
{
    const uint8_t *src = in->data[plane];
    const int linesize = in->linesize[plane];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dst[y * width + x] = src[y * linesize + x] * scale;
    }
}

static void import_plane16(const AVFrame *in, int plane, int width, int height,
                           float scale, float *dst)
{
    const uint16_t *src = (const uint16_t *)in->data[plane];
    const int linesize = in->linesize[plane] / sizeof(*src);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dst[y * width + x] = src[y * linesize + x] * scale;
    }
}

static void export_plane8(uint8_t *dst, int linesize, int width, int height,
                          float inv_scale, const float *src)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dst[y * linesize + x] = av_clip_uint8(lrintf(src[y * width + x] * inv_scale));
    }
}

static void export_plane16(uint16_t *dst, int linesize, int width, int height,
                           float inv_scale, const float *src)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dst[y * linesize + x] = av_clip_uint16(lrintf(src[y * width + x] * inv_scale));
    }
}

static int diffuse_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    AnisoContext *s = ctx->priv;
    const float inv_kappa_sq = s->inv_kappa_sq;
    const float lambda = s->lambda;

    for (int p = 0; p < s->nb_planes; p++) {
        const int width = s->planewidth[p];
        const int height = s->planeheight[p];
        const int slice_start = ff_slice_pos(height, jobnr, nb_jobs);
        const int slice_end = ff_slice_pos(height, jobnr + 1, nb_jobs);
        const float *src = s->state[p];
        float *dst = s->next[p];

        if (!(s->planes & (1 << p)))
            continue;

        for (int y = slice_start; y < slice_end; y++) {
            const int ym = y > 0 ? y - 1 : y;
            const int yp = y + 1 < height ? y + 1 : y;

            for (int x = 0; x < width; x++) {
                const int xm = x > 0 ? x - 1 : x;
                const int xp = x + 1 < width ? x + 1 : x;
                const float value = src[y * width + x];
                const float dxm = src[y * width + xm] - value;
                const float dxp = src[y * width + xp] - value;
                const float dym = src[ym * width + x] - value;
                const float dyp = src[yp * width + x] - value;
                const float gxm = expf(-dxm * dxm * inv_kappa_sq);
                const float gxp = expf(-dxp * dxp * inv_kappa_sq);
                const float gym = expf(-dym * dym * inv_kappa_sq);
                const float gyp = expf(-dyp * dyp * inv_kappa_sq);
                const float filtered = value + lambda * (gxm * dxm + gxp * dxp +
                                                         gym * dym + gyp * dyp);

                dst[y * width + x] = av_clipf(filtered, 0.0f, 255.0f);
            }
        }
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    AnisoContext *s = ctx->priv;
    AVFrame *out;
    const int bytes = (s->depth + 7) / 8;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    for (int p = 0; p < s->nb_planes; p++) {
        if (!(s->planes & (1 << p))) {
            av_image_copy_plane(out->data[p], out->linesize[p],
                                in->data[p], in->linesize[p],
                                s->planewidth[p] * bytes,
                                s->planeheight[p]);
        }
    }

    if (s->reset || s->first_frame) {
        for (int p = 0; p < s->nb_planes; p++) {
            if (!(s->planes & (1 << p)))
                continue;

            if (s->depth == 8)
                import_plane8(in, p, s->planewidth[p], s->planeheight[p],
                              s->scale, s->state[p]);
            else
                import_plane16(in, p, s->planewidth[p], s->planeheight[p],
                               s->scale, s->state[p]);
        }
    }

    for (int i = 0; i < s->iterations; i++) {
        ff_filter_execute(ctx, diffuse_slice, NULL, NULL,
                          ff_filter_get_nb_threads(ctx));

        for (int p = 0; p < s->nb_planes; p++) {
            float *tmp;

            if (!(s->planes & (1 << p)))
                continue;

            tmp = s->state[p];
            s->state[p] = s->next[p];
            s->next[p] = tmp;
        }
    }

    for (int p = 0; p < s->nb_planes; p++) {
        if (!(s->planes & (1 << p)))
            continue;

        if (s->depth == 8)
            export_plane8(out->data[p], out->linesize[p],
                          s->planewidth[p], s->planeheight[p],
                          s->inv_scale, s->state[p]);
        else
            export_plane16((uint16_t *)out->data[p],
                           out->linesize[p] / sizeof(uint16_t),
                           s->planewidth[p], s->planeheight[p],
                           s->inv_scale, s->state[p]);
    }

    s->first_frame = 0;
    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    AnisoContext *s = ctx->priv;

    free_planes(s);
}

static const AVFilterPad aniso_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
};

const FFFilter ff_vf_aniso = {
    .p.name        = "aniso",
    .p.description = NULL_IF_CONFIG_SMALL("Apply anisotropic diffusion denoising."),
    .p.priv_class  = &aniso_class,
    .p.flags       = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC |
                     AVFILTER_FLAG_SLICE_THREADS,
    .priv_size     = sizeof(AnisoContext),
    .uninit        = uninit,
    FILTER_INPUTS(aniso_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
};
