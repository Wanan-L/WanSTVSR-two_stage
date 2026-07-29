#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, re, shlex, shutil, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INF = ROOT / 'inference'
RES = ROOT / 'results_test'
WORK = INF / 'goal_experiment'
LOGS = WORK / 'logs'
TEST_SH = INF / 'test_wan_stvsr.sh'
EVAL_SH = INF / 'eval_metrics.sh'
BASE_CSV = INF / 'UDM10_metrics.csv'
OUT_DIR = RES / 'UDM10'
METRICS_JSON = OUT_DIR / 'all_metrics_results.json'
HISTORY = WORK / 'experiment_history.csv'
BATCH = WORK / 'batch_candidates.csv'
STATE = WORK / 'state.json'
STATUS = WORK / 'STATUS.md'
BEST_PARAMS = WORK / 'best_params.json'
BEST_METRICS = WORK / 'best_metrics.csv'
FINAL_SUMMARY = WORK / 'final_summary.csv'
FINAL_REPORT = WORK / 'final_report.md'
LOCK = WORK / 'run.lock'
CSV_ENCODING = 'gb18030'
MAX_EFFECTIVE = 600
EPS = 1e-12

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode('utf-8'))

def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def metric_id(name: str) -> str:
    return {'CLIP-IQA':'clipiqa','CLIP-IQA+':'clipiqa_plus','DOVER (Overall)':'dover_overall','DOVER-A (Aesthetic)':'dover_aesthetic','DOVER-T (Technical)':'dover_technical','E-Warp (x1000)':'ewarp_x1000','MD-VQA':'md_vqa'}.get(name, norm(name))

def canonical(prompt: str, negative: str, cfg: float) -> str:
    value = '\x1f'.join((' '.join(prompt.split()), ' '.join(negative.split()), f'{cfg:.6f}'))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def script_params() -> dict[str, Any]:
    text = TEST_SH.read_text(encoding='utf-8')
    out: dict[str, Any] = {}
    for key in ('prompt', 'negative_prompt', 'cfg_scale'):
        matches = re.findall(rf'(?m)^\s*--{key}\s+(.+?)(?:\s+\\)?\s*$', text)
        if len(matches) != 1:
            raise RuntimeError(f'Expected one --{key}, found {len(matches)}')
        toks = shlex.split(matches[0].strip())
        if len(toks) != 1:
            raise RuntimeError(f'Cannot parse --{key}: {matches[0]!r}')
        out[key] = toks[0]
    out['cfg_scale'] = float(out['cfg_scale'])
    return out

def patch_script(original: bytes, prompt: str, negative: str, cfg: float) -> bytes:
    text = original.decode('utf-8')
    vals = {'prompt': shlex.quote(prompt), 'negative_prompt': shlex.quote(negative), 'cfg_scale': f'{cfg:.6f}'}
    for key, val in vals.items():
        pat = rf'(?m)^(\s*--{key}\s+).*?$'
        def repl(m: re.Match[str]) -> str:
            return m.group(1) + val + (' \\' if m.group(0).rstrip().endswith('\\') else '')
        text, count = re.subn(pat, repl, text, count=1)
        if count != 1: raise RuntimeError(f'Refused to patch --{key}')
    return text.encode('utf-8')

def load_metrics() -> list[dict[str, Any]]:
    with BASE_CSV.open('r', encoding=CSV_ENCODING, newline='') as f:
        rows = list(csv.reader(f))
    methods = rows[0][1:]
    osd = next(i for i, m in enumerate(methods) if 'osdenhancer' in norm(m))
    cur = next(i for i, m in enumerate(methods) if 'wan21' in norm(m) and 'baseline' in norm(m))
    direction_by_name = {
        'PSNR':'higher','SSIM':'higher','LPIPS':'lower','DISTS':'lower','CLIP-IQA':'higher','CLIP-IQA+':'higher','NIQE':'lower','ILNIQE':'lower','LIQE':'higher','MUSIQ':'higher','MANIQA':'higher','BRISQUE':'lower','DOVER (Overall)':'higher','DOVER-A (Aesthetic)':'higher','DOVER-T (Technical)':'higher','FasterVQA':'higher','MD-VQA':'higher','E-Warp (x1000)':'lower'}
    metrics = []
    for row in rows[1:]:
        label = row[0].strip()
        name = label.replace('↑','').replace('↓','').replace('�','').strip()
        metrics.append({'name': name, 'label': label, 'direction': direction_by_name[name], 'osd': float(row[1+osd].strip()), 'csv_current': float(row[1+cur].strip())})
    return metrics

def flatten(v: Any, prefix: str = '') -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(v, dict):
        for k, child in v.items(): out.update(flatten(child, f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(v, (int, float)) and not isinstance(v, bool): out[prefix] = float(v)
    return out

def observed_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    payload = json.loads(METRICS_JSON.read_text(encoding='utf-8'))
    leaves = flatten(payload.get('average', {}))
    explicit = {'CLIP-IQA':'clipiqa','CLIP-IQA+':'clipiqa+','DOVER (Overall)':'dover','DOVER-A (Aesthetic)':'dover_aesthetic','DOVER-T (Technical)':'dover_technical','FasterVQA':'fastvqa','MD-VQA':'mdvqa','E-Warp (x1000)':'ewarp_scaled_1000'}
    by_norm: dict[str, list[str]] = {}
    for k in leaves: by_norm.setdefault(norm(k), []).append(k)
    out = {}
    for m in metrics:
        keys = [explicit[m['name']]] if m['name'] in explicit else by_norm.get(norm(m['name']), [])
        keys = [k for k in keys if k in leaves]
        if len(keys) != 1: raise RuntimeError(f'Cannot map metric {m["name"]}: {keys}')
        out[m['name']] = leaves[keys[0]]
    return out

def history_fields(metrics: list[dict[str, Any]]) -> list[str]:
    fields = ['iteration','timestamp_start','timestamp_end','duration_seconds','parameter_key','prompt','negative_prompt','cfg_scale','hypothesis','parent_iteration','search_phase','inference_exit_code','evaluation_exit_code','run_status','failure_reason','metrics_json_sha256','output_video_path','composite_relative_gain','metrics_beating_osd','metrics_gain_ge_2pct','psnr_delta_db','target_met','is_best']
    for m in metrics:
        sid = metric_id(m['name']); fields += [f'metric_{sid}', f'delta_vs_osdenhancer_{sid}', f'relative_gain_{sid}']
    return fields

def read_history() -> list[dict[str, str]]:
    if not HISTORY.exists(): return []
    with HISTORY.open('r', newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def write_history(metrics: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    fields = history_fields(metrics)
    fd, name = tempfile.mkstemp(prefix=f'.{HISTORY.name}.', dir=HISTORY.parent, text=True)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader()
            for row in rows: w.writerow({k: row.get(k, '') for k in fields})
            f.flush(); os.fsync(f.fileno())
        os.replace(name, HISTORY)
    finally:
        if os.path.exists(name): os.unlink(name)

def append_history(metrics: list[dict[str, Any]], row: dict[str, Any]) -> None:
    rows = read_history(); rows.append(row); write_history(metrics, rows)

SUBJECTIVE_TARGET_METRICS = {'CLIP-IQA', 'CLIP-IQA+', 'ILNIQE', 'LIQE', 'MUSIQ', 'MANIQA', 'BRISQUE'}

def score(metrics: list[dict[str, Any]], obs: dict[str, float]) -> tuple[dict[str, float], float, int, int, float, bool]:
    gains = {}
    for m in metrics:
        base = m['osd']; val = obs[m['name']]
        gains[m['name']] = (val - base) / max(abs(base), EPS) if m['direction'] == 'higher' else (base - val) / max(abs(base), EPS)
    target_gains = [gains[m['name']] for m in metrics if m['name'] in SUBJECTIVE_TARGET_METRICS]
    beating = sum(v > 0 for v in target_gains)
    gain2 = sum(v >= 0.02 for v in target_gains)
    psnr_delta = obs['PSNR'] - next(m['osd'] for m in metrics if m['name'] == 'PSNR')
    avg_gain = sum(target_gains) / len(target_gains)
    # Rank candidates by how many target metrics cross OSD first; average gain alone over-rewards LIQE/BRISQUE while CLIP-IQA/MANIQA remain below target.
    composite = beating + 0.1 * gain2 + avg_gain
    target = len(target_gains) == len(SUBJECTIVE_TARGET_METRICS) and all(v > 0 for v in target_gains)
    return gains, composite, beating, gain2, psnr_delta, target

def _script_gpu(script: Path) -> str:
    text = script.read_text(encoding='utf-8')
    m = re.search(r'CUDA_VISIBLE_DEVICES=([0-9,]+)', text)
    return m.group(1) if m else ''

def _start_eval_memory_guard(log_file: Any) -> subprocess.Popen[bytes] | None:
    gb = float(os.environ.get('GOAL_EVAL_GPU_GB', '20'))
    if gb <= 0:
        return None
    elems = int(gb * 1024 ** 3 / 2)
    code = (
        'import torch,time\n'
        f'x=torch.empty(({elems},), device="cuda", dtype=torch.float16)\n'
        'x.fill_(0)\n'
        'print("eval memory guard allocated", x.numel()*x.element_size(), flush=True)\n'
        'time.sleep(10**9)\n'
    )
    env = os.environ.copy()
    gpu = _script_gpu(EVAL_SH) or _script_gpu(TEST_SH)
    if gpu:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    return subprocess.Popen([sys.executable, '-c', code], cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)

def run_shell(script: Path, log: Path, eval_memory_guard: bool = False) -> tuple[int, float]:
    start = time.monotonic()
    guard = None
    with log.open('wb') as f:
        if eval_memory_guard:
            f.write(b'[runner] starting eval GPU memory guard (~20GB)\n'); f.flush()
            guard = _start_eval_memory_guard(f)
            time.sleep(8)
        try:
            p = subprocess.run(['bash', str(script)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=False)
        finally:
            if guard is not None:
                guard.terminate()
                try:
                    guard.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    guard.kill(); guard.wait()
    return p.returncode, time.monotonic() - start

def save_status(state: dict[str, Any], detail: str) -> None:
    atomic_text(STATUS, '\n'.join(['# UDM10 prompt experiment status','',f'Updated: {now()}',f'State: {state.get("status")}',f'Next iteration: {state.get("next_iteration")}',f'Effective candidates: {state.get("effective_candidates",0)}/{MAX_EFFECTIVE}',f'Current best iteration: {state.get("best_iteration")}',f'Current batch: {state.get("batch_id")}', '', '## Latest event', detail, '']))

def acquire() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as f: f.write(json.dumps({'pid': os.getpid(), 'started': now()}))

def release() -> None:
    if LOCK.exists(): LOCK.unlink()

def load_state() -> dict[str, Any]:
    metrics = load_metrics(); params = script_params(); hist = read_history()
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding='utf-8'))
        if state.get('runner_version') == 2: return state
        state['legacy_state_preserved_at'] = now()
    else: state = {}
    tested = {r.get('parameter_key','') for r in hist if r.get('parameter_key')}
    next_iter = max([int(r['iteration']) for r in hist if r.get('iteration','').isdigit()] + [-1]) + 1
    state.update({'runner_version':2,'status':'ready_for_baseline_cfg5','original_parameters':params,'metrics':metrics,'next_iteration':next_iter,'effective_candidates':sum(1 for r in hist if r.get('run_status') in {'completed','success_candidate'}),'tested_keys':sorted(tested),'best_iteration':None,'best_score':None,'batch_id':0,'final_retests_done':0})
    atomic_json(STATE, state); save_status(state, 'Initialized v2 runner from current test_wan_stvsr.sh parameters.'); return state

def copy_best(iteration: int) -> str:
    root = RES / 'goal_experiment' / 'best'; target = root / f'iteration_{iteration:04d}'; staging = root / f'.iteration_{iteration:04d}.staging'
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in sorted(OUT_DIR.glob('*.mp4')):
        shutil.copy2(p, staging / p.name); copied.append(p.name)
    if not copied: raise RuntimeError('No generated mp4 outputs found')
    if target.exists(): shutil.rmtree(target)
    os.replace(staging, target)
    atomic_json(RES / 'goal_experiment' / 'best_manifest.json', {'iteration':iteration,'path':str(target.relative_to(ROOT)),'videos':copied,'created':now()})
    return str(target.relative_to(ROOT))

def update_best_from_history(state: dict[str, Any]) -> None:
    best = None
    for r in read_history():
        try: val = float(r.get('composite_relative_gain',''))
        except ValueError: continue
        if best is None or val > float(best.get('composite_relative_gain','-inf')): best = r
    if best: state['best_iteration'] = int(best['iteration']); state['best_score'] = float(best['composite_relative_gain'])

def run_one(state: dict[str, Any], cand: dict[str, Any], allow_duplicate: bool = False) -> dict[str, Any]:
    metrics = state['metrics']; iteration = int(state['next_iteration'])
    prompt, neg, cfg = cand['prompt'], cand['negative_prompt'], float(cand['cfg_scale']); key = canonical(prompt, neg, cfg)
    if not allow_duplicate and key in set(state.get('tested_keys', [])): raise RuntimeError('Duplicate candidate')
    original = TEST_SH.read_bytes(); source_hash = sha(TEST_SH); start_iso = now(); start = time.monotonic(); LOGS.mkdir(parents=True, exist_ok=True)
    inf_rc = eval_rc = -1; failure = ''
    try:
        atomic_bytes(TEST_SH, patch_script(original, prompt, neg, cfg)); inf_rc, _ = run_shell(TEST_SH, LOGS / 'latest_inference.log')
    finally: atomic_bytes(TEST_SH, original)
    if sha(TEST_SH) != source_hash: raise RuntimeError('test_wan_stvsr.sh was not restored')
    if inf_rc == 0: eval_rc, _ = run_shell(EVAL_SH, LOGS / 'latest_evaluation.log', eval_memory_guard=True)
    if inf_rc != 0: failure = 'inference_failed'
    elif eval_rc != 0: failure = 'evaluation_failed'
    elif not METRICS_JSON.exists(): failure = 'metrics_json_missing'
    row: dict[str, Any] = {'iteration':iteration,'timestamp_start':start_iso,'timestamp_end':now(),'duration_seconds':round(time.monotonic()-start,3),'parameter_key':key,'prompt':prompt,'negative_prompt':neg,'cfg_scale':f'{cfg:.6f}','hypothesis':cand.get('hypothesis',''),'parent_iteration':cand.get('parent_iteration',''),'search_phase':cand.get('search_phase',''),'inference_exit_code':inf_rc,'evaluation_exit_code':eval_rc,'run_status':'failed' if failure else 'completed','failure_reason':failure,'metrics_json_sha256':sha(METRICS_JSON) if METRICS_JSON.exists() else '','output_video_path':str(OUT_DIR.relative_to(ROOT)),'target_met':'0','is_best':'0'}
    if not failure:
        obs = observed_metrics(metrics); gains, comp, beating, gain2, psnr_delta, target = score(metrics, obs)
        row.update({'composite_relative_gain':comp,'metrics_beating_osd':beating,'metrics_gain_ge_2pct':gain2,'psnr_delta_db':psnr_delta,'target_met':'1' if target else '0','run_status':'success_candidate' if target else 'completed'})
        for m in metrics:
            sid = metric_id(m['name']); val = obs[m['name']]
            row[f'metric_{sid}'] = val; row[f'delta_vs_osdenhancer_{sid}'] = val - m['osd']; row[f'relative_gain_{sid}'] = gains[m['name']]
        if state.get('best_score') is None or comp > float(state['best_score']):
            row['is_best'] = '1'; state['best_score'] = comp; state['best_iteration'] = iteration; best_path = copy_best(iteration)
            atomic_json(BEST_PARAMS, {'iteration':iteration,'prompt':prompt,'negative_prompt':neg,'cfg_scale':cfg,'composite_relative_gain':comp,'metrics_beating_osd':beating,'metrics_gain_ge_2pct':gain2,'psnr_delta_db':psnr_delta,'success_conditions_met':target,'output_video_path':best_path,'timestamp':now()})
            fd, name = tempfile.mkstemp(prefix=f'.{BEST_METRICS.name}.', dir=BEST_METRICS.parent, text=True)
            try:
                with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=history_fields(metrics), extrasaction='ignore'); w.writeheader(); w.writerow({k: row.get(k, '') for k in history_fields(metrics)}); f.flush(); os.fsync(f.fileno())
                os.replace(name, BEST_METRICS)
            finally:
                if os.path.exists(name): os.unlink(name)
    append_history(metrics, row)
    state['next_iteration'] = iteration + 1
    if not allow_duplicate: state.setdefault('tested_keys', []).append(key)
    if not failure: state['effective_candidates'] = int(state.get('effective_candidates', 0)) + 1
    state['status'] = 'target_met_pending_retests' if row['target_met'] == '1' else 'running'
    atomic_json(STATE, state); save_status(state, f'Iteration {iteration}: {row["run_status"]}; best={state.get("best_iteration")}; score={row.get("composite_relative_gain","")}')
    return row

def bootstrap_candidates(state: dict[str, Any]) -> list[dict[str, Any]]:
    p = state['original_parameters']['prompt']; n = state['original_parameters']['negative_prompt']
    p2 = 'Faithful video restoration preserving original structure and identity, natural fine details, stable edges, realistic colors, motion-consistent textures, temporal consistency, and minimal hallucination.'
    return [{'prompt':p,'negative_prompt':n,'cfg_scale':5.0,'search_phase':'baseline','hypothesis':'Required cfg=5.0 original baseline.'},{'prompt':p,'negative_prompt':n,'cfg_scale':2.0,'search_phase':'smoke','hypothesis':'Smoke test: lower CFG with original text.'},{'prompt':p,'negative_prompt':n,'cfg_scale':7.0,'search_phase':'smoke','hypothesis':'Smoke test: higher CFG with original text.'},{'prompt':p2,'negative_prompt':n,'cfg_scale':5.0,'search_phase':'smoke','hypothesis':'Smoke test: English fidelity wording at cfg=5.0.'}]

PROMPTS = ['photorealistic high image quality video restoration, natural sharpness, vivid but realistic colors, balanced contrast, clean pleasing frames, faithful original content, stable motion', 'aesthetic natural video restoration with high visual quality, clean crisp details, realistic color harmony, pleasing contrast, stable frames, no added objects', 'CLIP-IQA oriented faithful video restoration: high aesthetic quality, natural clear image, realistic color, clean detail, stable temporal consistency, preserve input content', 'best quality, high quality, sharp and clear video, vivid natural colors, pleasing contrast, clean details, stable frames, faithful to the input content', 'high aesthetic quality, clear sharp natural video, vivid realistic color, clean texture, pleasing visual quality, stable temporal consistency, no new content', 'A clean, vibrant, aesthetically pleasing faithful restoration of the input video, preserving all original content while improving natural sharpness, pleasing color, clear contrast, stable detail, and overall visual quality.', 'Beautiful natural faithful video restoration with crisp clean appearance, realistic vivid color, balanced contrast, stable temporal detail, and no invented objects or scene changes.', '忠实低失真视频恢复，保持输入视频的原始结构、运动轨迹、边缘和色彩，只恢复可信自然细节，避免重绘、幻觉纹理和时序漂移。 Faithful low-distortion video restoration preserving original structure, motion, edges, and colors with only credible natural details.', '高审美质量、自然清晰、真实色彩的视频恢复，保持输入内容不变，使画面观感干净、舒适、细节可信、边缘稳定。 High aesthetic quality faithful video restoration with natural clarity, realistic color, clean pleasing appearance, credible detail, and stable edges.', '最佳观感的忠实视频增强，保留原始场景和运动，提升清晰度、自然色彩、局部对比度和整体视觉质量，不创造新内容。 Best visual quality faithful video enhancement preserving the original scene and motion while improving clarity, natural color, local contrast, and perceptual appeal.', '高观感且忠实的视频修复，保持原始内容和运动不变，呈现自然清晰、真实色彩、干净边缘、舒适对比度和高审美质量。 High perceptual quality faithful video restoration with natural clarity, realistic color, clean edges, pleasant contrast, and no invented content.', '自然清晰的高质量视频恢复，保留输入视频的场景结构、主体形状和时序运动，提升真实观感、色彩协调和可信细节，不添加新物体。 Natural clear high-quality faithful restoration preserving scene structure, shapes, and temporal motion while improving realistic visual appeal.', '保持原视频内容不变的清晰自然修复，强调几何稳定、时序一致、低感知失真、自然纹理和真实色彩，不添加新物体或新场景细节。 Clear natural faithful restoration with stable geometry, temporal consistency, low perceptual distortion, natural texture, and realistic color.', 'Natural high-quality faithful restoration of the input video, preserving the original scene and motion while improving clean credible detail, stable edges, realistic color, and perceptual quality without invented content.', 'Faithful clean video restoration with natural sharpness, realistic texture, stable temporal alignment, low artifacts, preserved geometry, and no hallucinated objects or scene changes.', 'Low-distortion faithful restoration of the input video, preserving geometry, motion, edges, and textures with minimal redraw, low perceptual distortion, stable temporal alignment, and natural colors.', 'Restore the input video conservatively with low distortion, unchanged structure, stable motion, clean edges, natural color, and only details supported by the source frames.', 'Conservative faithful reconstruction of the input video, preserving original pixels, shapes, motion, colors, and structure with clean natural detail, stable frames, and no invented content.', 'Restore only confirmed details from the input video, keep geometry and motion unchanged, maintain temporal stability, natural colors, low artifacts, and faithful fine structure.', "Faithful restoration of the input video, preserving original structure and identity, natural fine details, stable edges, realistic colors, temporal consistency, motion-consistent textures, minimal hallucination.','High-quality video restoration with faithful scene structure, clean stable edges, realistic colors, natural fine details, consistent textures across frames, and no invented content.','Restore the video faithfully with preserved shapes and identity, natural texture detail, smooth motion consistency, stable edges, realistic color balance, and restrained enhancement.','A clear natural restored video that keeps the original content unchanged, improves credible fine details, maintains temporal consistency, stable edges, realistic colors, and minimal hallucination.','Faithful spatio-temporal super-resolution, preserving original structure, motion-consistent textures, natural details, stable edges, realistic colors, and low artifact visibility."]
NEGATIVES = ['low aesthetic score, low image quality, blurry, dull, flat, noisy, artifacts', 'unaesthetic, low quality, blur, dull colors, bad contrast, artifacts', '', 'low quality', 'blurry, dull, low quality', 'low quality, blurry, dull, washed out, ugly, noisy, artifacts', 'blur, dull, flat contrast, low aesthetic quality, noise, artifacts', 'low quality, blurry, dull, artifacts', 'blur, dull color, low contrast, flicker, ghosting', 'low perceptual quality, dull texture, fake detail, artifacts', 'low perceptual quality, dull texture, temporal flicker, unstable edges, warped geometry, ghosting, ringing, oversharpening, fake texture, color drift, noise amplification', 'low quality, blurry, dull, flat, ugly, noisy, flicker, ghosting, artifacts', 'blur, low contrast, dull color, unnatural texture, flicker, ghosting, ringing, noise', 'low aesthetic quality, dull colors, flat contrast, muddy texture, unnatural color, visual artifacts, temporal flicker, ghosting, warped edges, fake details, oversharpening, ringing, noise amplification', 'perceptual distortion, LPIPS artifacts, DISTS artifacts, temporal misalignment, warped edges, geometry drift, hallucinated texture, fake detail, flicker, ghosting, ringing, oversharpening, color drift', '感知失真, 时序错位, 几何扭曲, 边缘漂移, 幻觉纹理, 虚假细节, 闪烁, 鬼影, 振铃, 过锐化, 颜色漂移, 噪声放大', 'warped geometry, temporal misalignment, perceptual distortion, texture redraw, unstable edges, hallucinated detail, flicker, ghosting, ringing, oversharpening, noise amplification, color drift', 'invented content, hallucinated texture, temporal flicker, motion-inconsistent detail, warped structure, ghosting, ringing, oversharpening, color drift, noise amplification, compression artifacts', "flickering, temporal inconsistency, unstable details, hallucinated textures, oversharpening, ringing artifacts, color shifts, ghosting, motion trails, warped structures, plastic texture, excessive smoothing, noise amplification','blur, flickering, temporal inconsistency, ghosting, warped structures, hallucinated textures, oversharpening, ringing artifacts, compression artifacts, color shifts, plastic texture, noise amplification','unstable edges, temporal jitter, motion trails, ghosting, hallucinated details, fake textures, oversharpening, ringing, color shifts, excessive smoothing, noise amplification, distorted structures','flickering, unstable details, temporal inconsistency, warped structures, ghosting, motion trails, plastic texture, excessive smoothing, oversharpening, ringing artifacts, noise, color shifts','hallucinated textures, invented details, temporal flicker, ghosting, ringing artifacts, over-sharpened edges, color drift, motion-inconsistent texture, plastic surfaces, excessive denoising']"]
def batch_fields() -> list[str]: return ['batch_id','candidate_index','parameter_key','prompt','negative_prompt','cfg_scale','search_phase','hypothesis','parent_iteration','status','iteration']

def write_batch(rows: list[dict[str, Any]]) -> None:
    fd, name = tempfile.mkstemp(prefix=f'.{BATCH.name}.', dir=BATCH.parent, text=True)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=batch_fields()); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
        os.replace(name, BATCH)
    finally:
        if os.path.exists(name): os.unlink(name)

def read_batch() -> list[dict[str, str]]:
    if not BATCH.exists(): return []
    with BATCH.open('r', newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def generate_batch(state: dict[str, Any], size: int = 30) -> list[dict[str, Any]]:
    seen = set(state.get('tested_keys', [])); batch_id = int(state.get('batch_id', 0)) + 1; rows = []
    base_p = state['original_parameters']['prompt']; base_n = state['original_parameters']['negative_prompt']
    def add(p: str, n: str, cfg: float, phase: str, hyp: str, parent: str = '') -> None:
        if len(rows) >= size: return
        key = canonical(p, n, cfg)
        if key in seen or any(r['parameter_key'] == key for r in rows): return
        rows.append({'batch_id':batch_id,'candidate_index':len(rows),'parameter_key':key,'prompt':p,'negative_prompt':n,'cfg_scale':f'{cfg:.6f}','search_phase':phase,'hypothesis':hyp,'parent_iteration':parent,'status':'pending','iteration':''})
    if batch_id == 1:
        coarse_cfgs = [1.0,3.5,5.0,7.5,9.0]
        prompt_cfgs = [3.5,4.5,5.0,5.5,6.5,7.5]
    elif batch_id == 2:
        coarse_cfgs = [0.5,0.75,1.25,1.5,2.0,2.5,3.0]
        prompt_cfgs = [0.5,0.75,1.0,1.25,1.5,2.0,2.5]
    else:
        coarse_cfgs = [2.0,2.5,3.0,3.5,4.0,4.25,4.5,4.75,5.0,5.5,6.0]
        prompt_cfgs = [2.0,2.5,3.0,3.5,4.0,4.25,4.5,4.75,5.0,5.5,6.0]
    for cfg in coarse_cfgs: add(base_p, base_n, cfg, 'coarse_cfg', 'First-batch fixed text CFG sweep.' if batch_id == 1 else 'Low-CFG refinement with original text after low CFG led prior batches.')
    for p in PROMPTS:
        for n in NEGATIVES:
            for cfg in prompt_cfgs: add(p,n,cfg,'prompt_negative_cfg','Low-CFG fidelity prompt with artifact-focused negative prompt.')
    ranked = sorted([r for r in read_history() if r.get('run_status') in {'completed','success_candidate'} and r.get('composite_relative_gain')], key=lambda r: float(r['composite_relative_gain']), reverse=True)
    for parent in ranked[:5]:
        base_cfg = float(parent['cfg_scale'])
        for cfg in sorted({max(0.2, base_cfg - 0.5), max(0.2, base_cfg - 0.25), base_cfg + 0.25, base_cfg + 0.5}): add(parent['prompt'], parent['negative_prompt'], cfg, 'local_cfg', f'Local CFG refinement around iteration {parent["iteration"]}.', parent['iteration'])
    state['batch_id'] = batch_id; atomic_json(STATE, state); write_batch(rows); save_status(state, f'Generated batch {batch_id} with {len(rows)} candidates.'); return rows

def run_bootstrap(state: dict[str, Any]) -> None:
    seen = set(state.get('tested_keys', []))
    for i, cand in enumerate(bootstrap_candidates(state)):
        key = canonical(cand['prompt'], cand['negative_prompt'], float(cand['cfg_scale']))
        if key in seen: continue
        run_one(state, cand, allow_duplicate=(i == 0))
        if state.get('status') == 'target_met_pending_retests': break

def run_batch(state: dict[str, Any]) -> None:
    rows = read_batch()
    if not rows or all(r.get('status') not in {'pending','failed'} and not r.get('status','').startswith('error:') for r in rows): rows = generate_batch(state)
    history = read_history()
    for row in rows:
        if row.get('status') not in {'pending','failed'} and not row.get('status','').startswith('error:'): continue
        cand = {'prompt':row['prompt'],'negative_prompt':row['negative_prompt'],'cfg_scale':float(row['cfg_scale']),'search_phase':row['search_phase'],'hypothesis':row['hypothesis'],'parent_iteration':row.get('parent_iteration','')}
        attempts = sum(1 for r in history if r.get('parameter_key') == row['parameter_key'])
        if attempts >= 3:
            row['status'] = 'failed_max_retries'; write_batch(rows); continue
        while attempts < 3:
            result = run_one(state, cand, allow_duplicate=attempts > 0)
            history.append(result); attempts += 1
            row['status'] = result['run_status']; row['iteration'] = str(result['iteration']); write_batch(rows)
            if result['run_status'] != 'failed': break
            if attempts < 3: time.sleep(10)
        if row.get('status') == 'failed' and attempts >= 3:
            row['status'] = 'failed_max_retries'; write_batch(rows)
        if state.get('status') == 'target_met_pending_retests' or int(state.get('effective_candidates',0)) >= MAX_EFFECTIVE: break

def finalize(state: dict[str, Any]) -> None:
    if not BEST_PARAMS.exists(): return
    best = json.loads(BEST_PARAMS.read_text(encoding='utf-8'))
    if not best.get('success_conditions_met'): return
    done = int(state.get('final_retests_done', 0))
    while done < 2:
        cand = {'prompt':best['prompt'],'negative_prompt':best['negative_prompt'],'cfg_scale':best['cfg_scale'],'search_phase':'final_retest','hypothesis':f'Final verification retest {done+1}/2.','parent_iteration':best['iteration']}
        run_one(state, cand, allow_duplicate=True); done += 1; state['final_retests_done'] = done; atomic_json(STATE, state)
    final_rows = [r for r in read_history() if r.get('search_phase') == 'final_retest' or r.get('iteration') == str(best['iteration'])]
    write_history(state['metrics'], final_rows); shutil.copy2(HISTORY, FINAL_SUMMARY)
    atomic_text(FINAL_REPORT, f"# Final UDM10 Prompt Experiment Report\n\nBest iteration: {best['iteration']}\nCFG scale: {best['cfg_scale']}\nMetrics beating OSDEnhancer: {best['metrics_beating_osd']}/{len(state['metrics'])}\nMetrics with >=2% relative gain: {best['metrics_gain_ge_2pct']}\nPSNR delta vs OSDEnhancer: {best['psnr_delta_db']:.6f} dB\nComposite relative gain: {best['composite_relative_gain']:.6f}\n\nPrompt:\n{best['prompt']}\n\nNegative prompt:\n{best['negative_prompt']}\n")
    state['status'] = 'finalized'; atomic_json(STATE, state); save_status(state, 'Final retests complete.')

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--bootstrap', action='store_true'); ap.add_argument('--generate-batch', action='store_true'); ap.add_argument('--run-batch', action='store_true'); ap.add_argument('--finalize', action='store_true'); args = ap.parse_args()
    if sum([args.bootstrap,args.generate_batch,args.run_batch,args.finalize]) != 1: ap.error('choose exactly one action')
    acquire()
    try:
        state = load_state(); update_best_from_history(state)
        if args.bootstrap: run_bootstrap(state)
        elif args.generate_batch: generate_batch(state)
        elif args.run_batch: run_batch(state)
        elif args.finalize: finalize(state)
        atomic_json(STATE, state)
    finally: release()
    return 0
if __name__ == '__main__': raise SystemExit(main())
