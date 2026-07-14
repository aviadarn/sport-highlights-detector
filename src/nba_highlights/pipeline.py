import logging
import subprocess

from nba_highlights import media, scenes
from nba_highlights.audio import loudness, prosody, keywords
from nba_highlights.config import Settings
from nba_highlights.features import ShotFeatures
from nba_highlights.fuse import percentile_normalize, audio_composite
from nba_highlights.interfaces import ActionRecognizer, Transcriber
from nba_highlights.merge import merge_shots
from nba_highlights.models import (
    AudioSignals, ActionSignals, ShotScore, Report, VideoInfo,
)
from nba_highlights.scoring.weighted import WeightedScorer
from nba_highlights.select import select_highlights

log = logging.getLogger(__name__)


def run_pipeline(
    source: str,
    settings: Settings,
    transcriber: Transcriber,
    action: ActionRecognizer,
    *,
    workdir: str,
    generated_at: str,
    title: str = "",
    downloader=media._default_downloader,
    runner=subprocess.run,
    detector_fn=scenes._default_detector,
    audio_loader=loudness.load_audio,
    scorer=None,
    emit_features_path: str | None = None,
) -> Report:
    settings.check()
    if scorer is None:
        scorer = WeightedScorer(settings.weights)
    video_path, audio_path = media.ingest(source, workdir,
                                          downloader=downloader, runner=runner)
    duration_s = media.probe_duration(video_path, runner=runner)
    shots = scenes.detect_scenes(video_path, duration_s, detector_fn=detector_fn)
    transcript = transcriber.transcribe(audio_path)
    samples, sr = audio_loader(audio_path)
    times, rms = loudness.rms_envelope(samples, sr)

    raw_loud: list[float] = []
    raw_pros: list[float] = []
    raw_kw: list[float] = []
    peak_times: list[float | None] = []
    matched: list[list[str]] = []
    texts: list[str] = []
    for shot in shots:
        try:
            peak, t_peak = loudness.shot_loudness(times, rms, shot.start_s, shot.end_s)
            pros = prosody.shot_prosody(samples, sr, shot.start_s, shot.end_s)
            kw_score, kw_matched = keywords.shot_keywords(
                transcript, shot.start_s, shot.end_s, settings.hype_lexicon)
        except Exception as e:  # never let one shot abort the run
            log.warning("audio scoring failed for scene %s: %s", shot.scene_id, e)
            peak, t_peak, pros, kw_score, kw_matched = 0.0, None, 0.0, 0.0, []
        raw_loud.append(peak)
        raw_pros.append(pros)
        raw_kw.append(kw_score)
        peak_times.append(t_peak)
        matched.append(kw_matched)
        texts.append(transcript.text_in(shot.start_s, shot.end_s))

    loud_n = percentile_normalize(raw_loud)
    pros_n = percentile_normalize(raw_pros)
    kw_n = percentile_normalize(raw_kw)

    feats_list: list[ShotFeatures] = []
    shot_scores: list[ShotScore] = []
    for i, shot in enumerate(shots):
        comp = audio_composite(loud_n[i], pros_n[i], kw_n[i], settings.weights)
        act_score, act_label = 0.0, ""
        if settings.full_action or comp >= settings.audio_gate:
            try:
                res = action.score_clip(video_path, shot.start_s, shot.end_s)
                act_score, act_label = res.score, res.label
            except Exception as e:
                log.warning("action scoring failed for scene %s: %s", shot.scene_id, e)
        feats = ShotFeatures(
            scene_id=shot.scene_id, start_s=shot.start_s, end_s=shot.end_s,
            loudness_n=loud_n[i], prosody_n=pros_n[i], keywords_n=kw_n[i],
            keyword_count=float(len(matched[i])), action_score=act_score,
            shot_duration_s=max(0.0, shot.end_s - shot.start_s),
            rel_position=(shot.start_s / duration_s) if duration_s > 0 else 0.0)
        feats_list.append(feats)
        fused = scorer.score(feats)
        shot_scores.append(ShotScore(
            scene_id=shot.scene_id, start_s=shot.start_s, end_s=shot.end_s,
            audio=AudioSignals(loudness=loud_n[i], prosody=pros_n[i], keywords=kw_n[i]),
            action=ActionSignals(score=act_score, label=act_label),
            audio_composite=comp, fused=fused,
            matched_keywords=matched[i], peak_loudness_s=peak_times[i],
            transcript=texts[i]))

    if emit_features_path:
        with open(emit_features_path, "w") as fh:
            for f in feats_list:
                fh.write(f.model_dump_json() + "\n")

    merged = merge_shots(shot_scores, settings.threshold, settings.max_gap_s)
    highlights = select_highlights(merged, settings.threshold)

    config = {
        "weights": settings.weights.model_dump(),
        "threshold": settings.threshold,
        "audio_gate": settings.audio_gate,
        "max_gap_s": settings.max_gap_s,
        "action_backend": settings.action_backend,
        "asr_backend": settings.asr_backend,
        "asr_model": settings.asr_model,
        "full_action": settings.full_action,
        "fusion": settings.fusion,
    }
    return Report(
        video=VideoInfo(source=source, title=title, duration_s=duration_s),
        config=config, generated_at=generated_at, highlights=highlights)
