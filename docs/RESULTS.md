# Stage 1 results (base5), appended as runs finish
## 14 Nov 18:12 base s0, 0.66 MB (659,616 numbers), 5.4 h, 60 ep, causal norm 3 s, look-ahead 180 ms (phones_out/base5/base_P655924_s0.json)
- libriA fp32 .4810 int8 .4831 | libriB fp32 .4725 int8 .4740 | testother fp32 .4747 int8 .4763 | him fp32 .6033 int8 .6131; hyp/ref ~.87
- vs B0 (libriB 42.84, testother 43.40 int8): +4.56 / +3.63 worse. B0 saw whole clips (non-streaming); this base streams. So "within ~1 of B0"
  is not a fair known answer for a streaming base; read as the cost of streaming (UNVERIFIED as cause: would verify with a non-causal-norm,
  wide-look-ahead control at the same size).
- libriA still falling at ep 60 (.4834 ep56 -> .4813 ep60): not converged; epoch budget is a candidate constant for the 100 h base.
## 18:23 broken s0 (labels broken control): libriA int8 .9454 | libriB .9376 | testother .9499 | him .9475; hyp/ref ~.08 -> FAILS as it must.
## 18:35 base s1: libriA int8 .4872 | libriB .4834 | testother .4866 | him .6098. Seed gap s0-s1: libriB 0.94, testother 1.03 points (int8).
- A ~1-point seed gap is large against the 0.5 pass rule: the power check after s2-s4 decides seeds per arm.
## 18:36 100 h features done: tc100 28,538 utts (1 skipped), 100.65 h / sp0.9 111.81 h / sp1.1 91.49 h, 251 speakers; ~17.5 GB; disk 74 GB free.
## 18:38 base5_next.py (copy; base5.py untouched while stage 1a seeds 2-4 run): DATA=tc100 memmap loader (Row/load_mm), dev-other reported at the
end only (selection stays libriA). Read check: tc100 28,538 utts 100.65 h, sp0.9 111.81 h, devother 2,863 / 5.13 h (= prep log); labels 0..38;
feature scale like the 5.4 h set (mean -5.82 vs -5.09, std 4.20 vs 3.99, natural-log mel; corpus difference, removed by the causal norm).
Next after stage 1a: smoke DATA=tc100 PARAMS=1700000 2 epochs (GPU MiB at d 388, s/epoch, config in force), then the full 2 MB / 100 h runs.
## 19:05 stage 1a DONE. base 0.66 MB / 5.4 h, int8, seeds 0-4:
| seed | best ep | libriA | libriB | testother | him |
| 0 | avg60+58+52 | .4831 | .4740 | .4763 | .6131 |
| 1 | avg58+50+60 | .4872 | .4834 | .4866 | .6098 |
| 2 | 52 | .4855 | .4871 | .4871 | .6393 |
| 3 | 58 | .4865 | .4816 | .4840 | .6295 |
| 4 | 60 | .4826 | .4731 | .4782 | .6230 |
mean libriB 47.984 / testother 48.244; seed sd 0.608 / 0.493 points; libriB-testother corr 0.97 across seeds. (stage1_power.py)
Power of the pass rule (gain >= 0.5 on both, one-sided 95 % t lower bound > 0, unpaired, n seeds per side; 20k sims from the measured cov):
n3: eff0 .030, eff0.5 .188, eff1.0 .494, eff1.5 .794 | n5: .027 .301 .759 .972 | n8: .018 .406 .920 .999.
-> at this spread a true 1-point gain needs 8 seeds per side for P(PASS) >= 0.8; 5 seeds suffice for 1.5 points. False-pass rate ~2-3 %.
Seed spread at 2 MB / 100 h is to be measured; this table is for the 5.4 h runs (claim B small-data arms).
## 19:50 SMOKE 2 MB / 100 h (DATA=tc100 PARAMS=1700000, 2 epochs, logs_base5/smoke100/): config in force = set (d 388, 1,708,404 numbers,
100.646 h, speeds 1.0/0.9/1.1, 180 ms); GPU 772 MiB (< 1000); epoch 1 1598 s (cold disk reads of the memmaps), epoch 2 682 s; 1,567 batches.
After 2 epochs int8: libriA .4661 libriB .4751 testother .4686 him .6066 devother .4582 (already about the 5.4 h 60-epoch base).
Full run choice (mine, reversible): 20 epochs (~4 h; 2,000 h-passes vs 324 for the 5.4 h recipe), same early stop / averaging; seed 0 first,
its curve decides whether more epochs pay before more seeds.
## 20:0x ESC-50 downloaded (645,695,005 B zip, 2,000 wav, testzip OK) -> data/esc50/ESC-50-master (noise source for arm 2a + comparator)
## 20:2x noise + dials (both 2a and its comparator)
- noise_prep.py: ESC-50 -> cache/noise_esc50_f1234.npz (TRAINING noise, 1,600 clips, 801,600 frames, 50 categories) and noise_esc50_f5.npz
  (400 clips, kept apart for a later noisy TEST set; never in training).
- aug2a.py: vtlp(x, alpha) (piecewise warp, band-power interpolation) + add_noise(x, n, snr) (mel-power SNR) with selftest: identity at 1.0, flat
  spectrum unchanged at 0.9 / 1.1, a tone band moves 30 -> 32 at 1.1 and 30 -> 28 at 0.9, SNR exact at 5 / 20 dB, silent band stays -13.8155.
  Bug caught by the selftest: a second +1e-6 power floor inside the log (raw_logmel already floors at 1e-6) lifted silent bands by +0.69.
## 20:5x base5_next.py AUG=vtlp,noise (off by default; the running s0 job's script is frozen as logs_base5/full100/base5_next_AS_RUN_s0.py).
CPU check on fit[0]: 6 draws, mean |change| 3.906 / 0.73 / 0.321 / 0.152 / 3.238 / 3.013 (big = noise drawn), shapes kept, all finite.
GPU smoke (2 ep, 5.4 h, AUG on) waits for the GPU: the 2 MB run is alone on my 1 GB (776 MiB). full100 s0 at ep 4 (3,086 s, ~12 min/epoch
-> ends ~00:05); libriA .4972 at ep 4 (LR still near peak; the smoke's .4654 at ep 2 had its whole OneCycle annealed).
## 21:0x full100 s0 try 1 CRASHED: CUDA OOM in backward at epoch 5 (1 h 12 min lost; log *.CRASHED_OOM_ep5.log): allowed 593 MiB (ALLOC 620 / 8192 of
7.65 GiB), process 782 MiB. Epoch curve before it: ep2 libriA .5025, ep4 .4972 (secs 1073 / 1709 / 2662 / 3086).
Cause check (base5_memtest.py, the review-item-6 test I had NOT run before): worst one-step peak at d 388 / 100 h: MAXFR 24000 most-clips batch
494 MiB alloc / 800 MiB process; 20000: 443 / 688 (longest-clip batch) ; 16000: 345 / 618. Longest clip 2,726 frames (speed 0.9).
-> 24000 leaves ~100 MiB under the 593 cap for allocator caching between unlike batches; chose MAXFR 20000 (150 MiB margin), cap unchanged.
Resume added (ckpt after every epoch; cfg must match): kill after ep 1 + restart -> resumed_from_epoch 1, ep 4 loss 2.3483 = uninterrupted 2.3483
(ep2 3.1987 vs 3.2013: GPU nondeterminism, ep1 already 4.4201 vs 4.4202 without resume). libriB int8 .6816 vs .6828.
try 2 started 21:12, PID 499784, MAXFR 20000, script frozen as logs_base5/full100/base5_next_AS_RUN_s0_try2.py.
## 01:10 15 Nov full100 s0 try 2 DONE: 2 MB (1,708,404) / 100 h, 20 ep, MAXFR 20000, 13,981 s (3.9 h), GPU <= 766 MiB. int8:
libriA .4330 | libriB .4196 | testother .4312 | him .5770 | devother .4188 (fp32: .4308 / .4205 / .4276 / .5607 / .4186).
Curve libriA (every 2 ep): .5155 .5156 .5025 .4902 .4721 .4649 .4468 .4397 .4322 .4308 -> still falling at ep 20 (end of OneCycle anneal).
vs B0 (0.66 MB, whole-clip BiGRU, 5.4 h): libriB 42.84 -> 41.96, testother 43.40 -> 43.12: the streaming 2 MB / 100 h base now beats B0 slightly.
vs the 0.66 MB / 5.4 h streaming base (5-seed mean): libriB 47.98 -> 41.96 (-6.0), testother 48.24 -> 43.12 (-5.1), him 62.3 -> 57.7.
Cost: ~3.9 GPU-h per 2 MB / 100 h run -> arms are screened at 0.66 MB / 5.4 h first (review decision 9); 100 h only for survivors.
Open: does 40 epochs pay (ep 16 -> 20 gained 0.9)? Bracket later with one 40-epoch run, not before the arms.
## 01:1x-01:3x comparator 'base+std' built (ARM_2a_DESIGN section 3)
- AUG smoke 4 ep (vtlp,noise; 5.4 h): config in force aug [vtlp, noise]; 14 s/epoch (plain 8); ep 4 libriB int8 .7089 (plain 4-ep .6819/.6828).
- spk_stats.py: leave-one-out per-speaker band mean / sd for fit, cal (+ sp0.9 / sp1.1), libriA, libriB, testother, him -> cache/spk_*.npz.
  Speakers: fit 30, cal 10, libriA 4, libriB 6, testother 33 (10 utts each), him 1. NOTE: libriA / libriB judge speaker effects on only 4 / 6
  speakers; testother (33) is the set where a speaker-normalisation gain can show.
- Per-speaker CMVN under the causal norm is NOT a no-op: mean |cnorm(x) - cnorm(cmvn(x))| 0.2456 vs mean |cnorm(x)| 0.6581 (ratio 0.373, 120 testother utts).
- Plain (x - mu) / sd stalled learning (4 ep: hyp/ref .079, libriB .9297): it shrank inputs ~3.5x under the causal norm's +1.0 floor. Fixed by a global
  rescale GSD = 3.549 (median fit speaker sd). After: cmvn-only 4 ep libriB .6821 (plain .6819), him .7574 (plain .7902); std (aug+cmvn) .7189.
- Default path unchanged: plain 4 ep loss 2.3459 / libriB .6819 vs earlier 2.3483 / .6828 (GPU nondeterminism).
- 01:31 STARTED comparator 'std' seeds 0-7, 60 ep, 0.66 MB / 5.4 h (logs_base5/std/, results phones_out/base5_std/), ~20 min each -> ~04:15.
  CALIBRATION of the comparator; vs the plain base (5 seeds) it reads as "what the standard tricks give", exploratory.
  First launch failed (frozen script copy could not import aug2a under python -I); fixed by freezing aug2a.py next to it.
## 01:4x-02:0x arm 2a voice-size factor
- Route 1 CUT (vtln_alpha.py, closed-form F3-region peak of the loud-frame mean spectrum): FAILED its known answer twice. v1 raw argmax sat on the
  1.8 kHz band edge (fit medians F 1842 / M 2266 Hz, P(F > M) .295); v2 detrended over 300-5000 Hz: fit F 2980 / M 3301, P(F > M) .281, cal .292,
  testother .831. Inverted on the 40 training speakers, right on testother: cause unknown (stop rule 8.5: cut after two failures, not repaired a 3rd time).
- Route 2 (vtln_grid.py, classic unsupervised VTLN): the trained 0.66 MB base s0 scores each warp by mean negative posterior entropy; leave-one-out
  per speaker. testother, grid 0.88-1.12: F median alpha 0.88, M 1.04, P(F alpha < M alpha) .79 (right direction: female formants higher -> warp down);
  108 / 330 on the 0.88 edge -> grid widened to 0.80-1.20. All sets running (logs_base5/vtln_grid.log), CPU, ~1 h.
## 02:2x comparator 'std' loop STOPPED after seed 0 (01:54) for his LR request; it will be rerun with time-varying noise ('events') anyway.
## 02:2x aug2a.add_noise_events (his 15 Nov "random background audio ... not stable per clip"): 1-4 events, random onset/length, own SNR ramp
(U(0,20) dB each end), 100 ms fades. Selftest: one constant event reproduces 10 dB. Real fit clip, 3 draws, per-frame SNR p10/50/90:
4.1/15.3/22.8 | 2.3/8.0/28.6 | 26.8/110.6/110.6 (short event, rest clean). base5_next AUG=events.
## 02:3x HIS RUN: peak LR 0.05 (B0 recipe 0.003), 20 passes OneCycle + 5 tail passes at 0.003 -> 0, 0.66 MB / 5.4 h, no early stop
(phones_out/base5_lr05/lr05_P655924_s0.json, logs_base5/lr05/small_s0.log). Loss 3.74, 4.52, then 20.39 at ep 3 (LR reached ~0.05) = divergence;
output collapsed to near-silence (libriA hyp/ref .017, PER .9843 from ep 4 to 25); the tail did not recover it. Best checkpoint = ep 2:
int8 libriA .9125 libriB .9135 testother .9136 him .9279 (plain 0.003 base 60 ep: libriB .474-.487).
## 02:3x vtln_grid.py (grid 0.80-1.20) DONE on all sets (cache/vtln_*.npz). P(F alpha < M alpha), ties half: fit .924 (30 spk), cal .854 (10),
libriA 1.0 (4), libriB 1.0 (6), testother: F/M medians as before (28 / 330 utts on the 0.80 edge = 8.5 %, noted); him alpha 0.96 x2 / 1.00 x2 / 1.04 x6.
-> the voice-size factor separates sexes in the right direction on every set: route 2 is the 2a voice-size estimator.
## 02:46 LR bracket, plain 0.66 MB / 5.4 h, 60 passes OneCycle, seed 0, int8 (logs_base5/lr05/sweep_*.log, phones_out/base5_lr*):
| peak LR | libriA | libriB | testother | him |
| 0.003 (stage 1a s0; 5-seed mean libriB .4798) | .4831 | .4740 | .4763 | .6131 |
| 0.006 | .5230 | .5220 | .5218 | .6361 |
| 0.012 | .6243 | .6359 | .6295 | .8590 |
| 0.024 | .6908 | .7035 | .6973 | .7836 |
| 0.05 + 5 tail (his run, 20 passes) | .9125 | .9135 | .9136 | .9279 (diverged ep 3) |
-> higher than 0.003 is worse at every step (0.006 already +4.8 libriB, outside the 0.61 seed sd). Lower side not yet bracketed: 0.0015, 0.00075 next.
## 03:00 vtln speed versions (vtln_grid_sp.log): sp0.9 (formants scaled down 0.9) -> alphas 1.00-1.20 (warp UP); sp1.1 -> 0.80-0.96 (warp DOWN):
the estimator recovers a known imposed shift both ways. Some sp1.1 alphas sit on the 0.80 edge (fit 188 / 1977).
## 03:25 LR lower bracket: 0.0015 int8 libriA .4802 libriB .4781 testother .4836 him .6426 (60 ep) | 0.00075: .5008 / .5018 / .5070 / .6328
(early stop at 58). -> optimum interior at 0.0015-0.003 (0.0015 within seed sd of the 0.003 5-seed mean .4798 / .4824). Peak LR stays 0.003:
the learning rate is not what holds the base back.
## 15 Nov 04:26 ARM 2a TEST LOCKED (before any 2aN / 2aR run). removal2a.py: control (a) identity PASS bit-exact on 10 sets; floor known answers: a
10x-p5 floor on p10-p50 frames |noisy-clean| .195 -> .022 nat after removal, est floor - log(Pn) .096 (= log 1.1, right); clean loud frames
change .0000. RATE CUT (2 routes failed the imposed-speed known answer: peaks .957 / 1.038, modulation centroid .969 / 1.028, expect .9 / 1.1;
peak rate tracks transcript vowel rate r .800 across 40 speakers). Trainer: FEAT env (diff vs std2 AS_RUN = FEAT lines only). Hashes:
9ce113391debefd891f89535520e311a235561b9505d00262b67d866bb6bd8ec  CRITERIA_2a.md
66db90e45f26449791086b3baf10f2d109e7496d7780112d6da512ef89d409d5  score_2a.py
e0771e6c4bc4885e878f05014364927a363f761adf9de7b4d3e147f73ce30d59  removal2a.py
## 15 Nov 04:45 removal2a 2aN_ + 2aR_ files built (rc 0, 20 sets, lens unchanged, all finite). Mean |change| per value: fit .813, testother 1.347, him .858 (2aN); ~5-7 % of values hit the floor (BETA). Control (b) alphas differ from 2aN on 76-92 % of utterances (mean |diff| .03-.105). cmvn sd median rises (testother 3.655 -> 4.24) because floored frames go deeper; the per-speaker cmvn divides it out.
## 15 Nov 06:15 COMPARATOR std2 DONE (AUG=vtlp,noise,events SPKNORM=cmvn, 0.66 MB / 5.4 h, 60 ep, 8 seeds, int8): libriB 48.61 sd .33 | testother 49.69 sd .29 | libriA 49.12 sd .28 | him 64.92 sd 1.33. WORSE than the plain base (5 seeds: libriB 47.98 sd .61, testother 48.24 sd .49; him ~61): the dials + cmvn cost ~0.6 / ~1.5 points on clean sets at this size. Cause unknown (candidates: noise dials on clean test sets; cmvn; both). The 2a test stays against std2 as locked; the 2a report will ALSO show 2aN vs the plain base (exploratory, not a verdict), because a win over a weakened comparator alone would overstate 2a. Seed sd ~half the plain base's -> more power than the criteria's estimate.
## 15 Nov 06:39 HIS REWARD RATIO (the author: 'ratio of reward to punishment of 5:1'): base5_next.py REWARD=k scales CTC's push-up (-gamma) by k, push-down (p) unchanged; selftest_reward on CPU: k=1 == loss.backward, gamma >= 0 and sums to the loss weight per frame, k=5 ratio 5.00. Runs (EXPLORATION, plain 0.66 MB / 5.4 h, s0, LR 0.003): REWARD=5 then same-script REWARD=1 reference, inserted after 2aN s0 by run_2a_v2.sh (v1 launcher stopped; 2a test order otherwise unchanged).
## 15 Nov 06:41 HIS clarification 'the reward should aim at the 0.05 equivalent not the punishment' -> run rw5his: REWARD=5, peak LR 0.01, 20 passes + 5 tail (0.003 -> 0), no early stop (his 0.05 recipe with punishment at 1/5). Launcher v3 (v2 stopped; rw5 at LR .003 keeps running as an extra point).
## 15 Nov 06:49 REWARD=5 at LR .003 (rw5, s0): TOTAL COLLAPSE: output empty (hyp/ref 0.000, PER 1.0 on every set) from epoch 2; loss rose 5.9 -> 11.8; early stop ep 12. Mechanism (reasoned, matches the prediction made before the result): blank is the right answer on ~75 % of frames, so a uniform 5x push-up mostly pushes blank's output bias up. New option REWARD_BLANK (phones k:1, blank 1:1), checked on CPU (blank column = plain CTC; phones = 5x push-up). HIS channels (CHAN=level: absolute loudness, loudness over a running 3 s floor, 20 ms rise; +3,480 numbers = 663,096; causal, stream check ok) + same-width noise control CHAN=noise3. Queue runner logs_base5/q/runq.sh (PID 1694082): rw1, rw5pb, rw5hispb, ch_level, ch_noise3, then 2aN s1-7, 2aR s0-3.
## 15 Nov 06:51 rw5his (HIS 0.05-reward / 0.01-punishment, 20+5 passes, s0): same TOTAL COLLAPSE (hyp/ref 0.000, PER 1.0 every set; loss 6.6 -> 20.2 by ep 6; all 25 passes ran). Derivation (INFERRED, first-order, output layer only; Adam makes it approximate): grad p - k*gamma is equivalent up to a uniform logit shift to p - [k*gamma + (1-k)/C], i.e. a target of k*gamma + (1-k)/C. k < 1 = label smoothing with eps = 1 - k (1:5 = eps 0.8; standard eps 0.1 = 0.9:1). k > 1 = negative smoothing: the target exceeds 1 on the aligned class -> pushes to hard one-hot on the majority aligned class (blank) -> collapse. Phones-only k (blank 1) = a tilt against blank (fewer deletions). Queued rw02 (1:5) and rw09 (0.9:1 = standard smoothing) after rw5hispb.
## 15 Nov 06:52 HIS 'lets try 1:3, i trust in it' -> rw13 (REWARD=1/3, ~label smoothing eps 0.67) put FIRST in the queue.
## 15 Nov 07:17 REWARD results (s0, int8 PER / hyp_over_ref): rw1 (1:1, same script) libriB .4781/.885 testother .4829/.877 him .5738 |
rw13 (HIS 1:3) 1.0/0.0 every set (collapse) | rw5pb (5:1 phones, blank 1:1) .9830/.017 (collapse) | rw5hispb loss 401 at ep 16, stopped by me (rc 143).
CORRECTION (3.6) to the 06:5x mechanism note: "blank bias rises" cannot explain k = 1/3 collapsing (the label-smoothing algebra predicts the
argmax is kept). Checked (/tmp/rwdiag.py, plain base s0, real padded libriA batch): custom path at k = 1 equals loss.backward on every
parameter (max diff 0.0); gamma >= -3e-7, sums to the loss weight per frame, 0 on padding -> no bug found in the path. New hypothesis: the
uniform part of the modified gradient (net (1-k) per frame) changes no output but is the same every step; Adam makes it full-size steps in the
trunk. Test: REWARD_PROJ=1 removes it (rw13p, rw5pbp queued first). rw02 / rw09 held (logs_base5/q/hold.txt) until this is known.
## 15 Nov 07:27 HIS IDEA 'silence gives 1/2 the reward, punish choosing silence when its another's turn, low values acceptable, high punished': BLANKPEN=lambda (loss lambda*w*p_blank*(1-gamma_blank); gradient = autograd within 2e-9 on CPU) + REWARD_BLANK=0.5 (with REWARD_PROJ). Queued: bp03, bp10 (penalty only, bracket), his_half_bp03 (full idea).
## 15 Nov 07:28 rw13p (1:3 WITH the uniform part removed): STILL collapses (output 0.0 every set; loss flat ~7.0 from ep 2). -> the 'uniform part + Adam' hypothesis is FALSIFIED as the cause for k = 1/3 (removing it did not remove the collapse). New hypothesis (reasoned, untested): CTC's target gamma comes from the model's own p; from scratch p is flat, so gamma is flat; a 2/3-uniform target keeps p flat and the alignment never forms (plain CTC escapes by becoming peaky). Same bootstrap lock for k > 1 (sharpens toward early blank). Test idea: apply the ratio only after the alignment exists (warm start from the 1:1 checkpoint, or ramp k from 1).
## 15 Nov 07:40 HIS: 'we are probably wasting our time with me shouting random numbers feel free to proceed with the real tests and the actual 52 or whatever phenomes' -> reward / blank-penalty runs HELD (logs_base5/q/hold.txt: bp03, bp10, his_half_bp03, rw02, rw09); rw5pbp (running) finishes. Queue: ch_level, ch_noise3 (black-box strength), 2aN s1-7, 2aR s0-3. Next build: arm 2c (his structure, 52 outputs).

## 15 Nov 07:50 ARM 2c built (his "proceed with the real tests and the actual 52"; reward/blank-penalty runs HELD, bp10 already running finishes)
- labels2c.py: 53 tokens (+SIL) by context rules on MFA; HARD CHECK all 2,703 fit+cal utterances map back to base5 labels exactly. Train counts:
  SCHWA 15207, FLAP_T 2488, FLAP_D 1472, T_NOPUFF 1970, DARK_L 3051, R_POSTV 2686, K_FRONT 1256, K_BACK 3580, G_FRONT 802, G_BACK 962, T_NOREL 893,
  TS 481, P_NOPUFF 373, DZ 219, K_NOPUFF 216, GLOTTAL_T 94, SIL 8978.
- struct2c.py selftest ok: widths base 232, tok 230, lite 232, full 178 (trunk shrinks to pay for heads + specialists at 655,924 counted).
- CRITERIA_2c.md sha256 f61cd050a63f7018abb258589f08cf9c104ecef48c83b253428d6b71f897304c ; score_2c.py e90cf9c31af0074c46a72fa306889a7a29733261bf567c153dcddf58c8d70f36
  (code at hash time: struct2c.py 81853519..., labels2c.py 74be182f...; the frozen AS_RUN copy is hashed again at launch).
- Smokes (4 ep) full / lite / tok queued first.
## 15 Nov 07:52 HIS BLANK PENALTY, the two runs that finished (EXPLORATION, 1 seed, plain 0.66 MB / 5.4 h, int8; ref rw1 1:1 libriB .4781 testother .4829):
- bp03 (BLANKPEN 0.3): libriB .4749 (hyp/ref .909) testother .4802 (.903) libriA .4805 him .6328 devother .4754 -> -0.3 points, inside the base seed sd (.61 / .49): no evidence either way.
- bp10 (BLANKPEN 1.0): libriB .5194 (.970) testother .5202 (.957) him .6623 -> the penalty does what it says (output length .88 -> .97) but costs ~4 points.
- His full idea (half reward for silence + penalty) NOT run (held at his word). Reopen condition: a stage where output length is the measured limit.
## 15 Nov 08:4x 2c smokes (4 ep, s0, int8; plain 4-ep ref libriB .6819): lite .6673 | tok .8004 (hyp/ref .32) | full .8221 (hyp/ref .27): more output
classes start slower (4-ep only, not a test). GPU total 674 / 686 / 720 MiB. First try OOMed (fp32 copy of the 4 trunk layers): heads + specialists
now under bf16 autocast. Frozen: logs_2c/run/ (SHA256 file; struct2c_AS_RUN.py 5db389e6...). 33 test runs queued first (08:4x).
## ch_noise3 s0 FAILED the stream check (08:17: fresh randn per call differs when audio is cut -> not causal-deterministic; the check caught it).
Fixed: fixed N(0,1) table, random offset per utterance in training, offset 0 in eval; re-queued after the 2c runs (logs_base5/chan2/; diff vs the
ch_level copy = default-off reward/blank flags + the noise lines).
- ch_level s0 (his 'more channels' for the black box: level, level above floor, 20 ms rise): libriB .4764 testother .4839 him .6623 devother .4760 -> inside the plain-base seed spread; read only next to ch_noise3 (same width, pure noise).

## 15 Nov 16:13 ARM 2c TEST SCORED (CRITERIA_2c.md sha f61cd050... and score_2c.py e90cf9c3... unchanged; 33/33 runs; int8 PER x100, 8 seeds)
| arm | libriB | testother | libriA | him | devother |
| base (232 wide) | 48.26 sd .58 | 48.63 sd .22 | 48.90 | 63.36 | 48.05 |
| tok (53 tokens + SIL) | 48.27 sd .30 | 49.02 sd .22 | 49.07 | 63.03 | 48.15 |
| lite (18 heads, training only) | 48.01 sd .40 | 48.49 sd .33 | 48.71 | 61.48 | 47.76 |
| full (trunk 178 + heads + 54 specialists) | 50.14 sd .31 | 50.58 sd .34 | 50.35 | 64.96 | 49.66 |
Validity: broken s0 100.0 on every set (fails, ok); base here vs stage 1a: libriB 48.26 vs 47.98, testother 48.63 vs 48.24 (within 1.0, ok).
Gains over base (g, lb95, ub95): tok libriB -0.01 (-0.42, 0.39), testother -0.39 (-0.59, -0.20) -> FAIL (a small cost on testother).
lite libriB +0.25 (-0.19, 0.69), testother +0.14 (-0.10, 0.39) -> FAIL (testother ub < 0.5: any gain is smaller than the 0.5 threshold there).
full libriB -1.88 (-2.29, -1.47), testother -1.95 (-2.20, -1.70) -> FAIL + COST (a finding to explain, PLAN row 2c).
By the rule's literal terms (researcher.md 3.7): no arm passes; tok and lite are shown below the 0.5 threshold on testother; full is shown worse by ~1.9.
Cause of the full cost: UNKNOWN. Candidate (untested): the trunk narrowed 232 -> 178 to pay for heads + specialists (246k counted numbers).
Diagnostic queued (EXPLORATION, 3 seeds each, phones_out/struct2c_diag): base at 178 wide (409,974) = does narrowing alone give ~50.1?;
full at 232 wide (959,385 counted) vs base at 959,385 = on an equal trunk, do the specialists beat spending the same numbers on width?
## 15 Nov 16:5x HIS CHANNELS for the black box (EXPLORATION, 1 seed each, int8): ch_level libriB .4764 testother .4839 him .6623 | ch_noise3 (fixed-table
noise, stream check ok) .4855 / .4891 / .6262 | plain base 5 seeds .4798 / .4824. -> level beats same-width noise by 0.9 / 0.5 but is no better
than no extra channels: the 3 level channels carry no measured gain for the black box (1 seed; noise columns cost ~0.6, as realization 84).
## 16:47 diag full_P959385_s0 OOM at epoch 8 (774 MiB total, 593 cap). Rerun fresh with ALLOC_MIB=720 (s0-s2; memory cap only, not recipe);
old ckpt + log kept as logs_2c/diag/*.oom*. base_P959385 and base_P409974 run with the default cap.

## 15 Nov 19:06 2c DIAGNOSTIC (EXPLORATION, int8 PER x100; 3 seeds unless said) - where the full arm's ~1.9 cost comes from
| model | trunk width | counted | libriB | testother | him |
| base | 178 | 409,974 | 49.27 sd .25 | 49.75 sd .17 | 62.29 |
| full (test arm, 8 seeds) | 178 | 653,529 | 50.14 | 50.58 | 64.96 |
| base (test arm, 8 seeds) | 232 | 655,924 | 48.26 | 48.63 | 63.36 |
| full | 232 | 959,385 | 49.21 sd .35 | 49.54 sd .23 | 63.82 |
| base | 284 | 955,132 | 47.68 sd .16 | 48.17 sd .14 | 61.97 |
Reading: (1) narrowing 232 -> 178 alone costs ~1.0 / 1.1 (base 178 vs base 232); (2) ON THE SAME TRUNK the heads + specialists top is WORSE than the
plain linear output at both widths: full-178 vs base-178 +0.87 / +0.83, full-232 vs base-232 +0.95 / +0.91, although it adds ~245k-300k numbers.
(3) tok (same 53 targets, plain linear output) costs 0.0 / 0.4 and lite (heads as training-only teachers) costs nothing: so most of the top's
harm sits in the SPECIALIST layer (54 separate 16-unit GELU bottlenecks between the trunk top and the joint decision, fed by head middles),
not in the tokens or the head questions. Mechanism NOT yet shown (removal/restoration test next): candidates (a) the 16-unit bottleneck per
token limits the readout, (b) reading head middles adds noise / optimisation trouble, (c) bf16 in the top (base output layer is fp32).
## 19:1x diag2 queued (logs_2c/run2 frozen; struct2c.py + SKIP / MIDS switches, defaults = tested arm): full-232 + direct linear readout (SKIP, 972,200) and full-232 without head middles (MIDS=0, 903,321), 3 seeds each. Prediction: if the specialist bottleneck is the cause, SKIP returns to ~base-232 (48.3 / 48.6) or better; if the head middles are the cause, MIDS=0 does.
## 15 Nov 22:3x diag2 RESULT (EXPLORATION, 3 seeds, int8 x100; libriB / testother / libriA / devother / him)
- full-232 + SKIP (direct linear readout next to the specialists, 972,200): 46.54 sd .14 / 47.58 sd .13 / 47.97 / 46.71 / 62.51
- full-232 MIDS=0 (specialists without head middles, 903,321): 48.47 sd .43 / 49.11 sd .28 / - / - / 60.87
- full-232 (959,385): 49.21 / 49.54 / 49.50 / 48.77 / 63.82 | base-284 (955,132): 47.68 / 48.17 / 48.33 / 47.32 / 61.97 | base-232: 48.26 / 48.63 / 48.90 / 48.05 / 63.36
Mechanism (removal / restoration, researcher.md 1.2): the specialists-only readout is what costs: adding the direct path back turns -0.95 into
+1.7 / +1.05 over base-232 and +1.1 / +0.6 over the same-size base-284 (same direction on libriA and devother). Head middles cost ~0.7 / 0.4
without the skip (interaction with SKIP untested). EXPLORATION: chosen on these sets, 3 seeds.
## 22:4x TEST 2c-2 locked: CRITERIA_2c2.md sha 7ec31d66f061cf5cdb5ecf0d092dbac8db0eb80b55e8dbd7c51213ba406493bb, score_2c2.py e1a5ebef615579b4e9b22ce1e7de5020100c2c2ccd0a1d86f332e39f7849b555;
fullskip at 655,924 (trunk 176) vs base, fresh seeds 8-15, frozen logs_2c/run3; queued AFTER the 2a remainder (2aN s5-7, 2aR s0-3).

## 16 Nov 01:35 ARM 2a TEST SCORED (CRITERIA_2a.md 9ce11339... and score_2a.py 66db90e4... = the hashes recorded at lines 106-107; all runs present)
int8 x100, mean sd: std2 (comparator, 8) libriB 48.61 .33 | testother 49.69 .29 | libriA 49.12 | him 64.92
                    2aN  (arm 2a, 8)     libriB 44.71 .40 | testother 47.18 .22 | libriA 46.58 | him 64.67 sd 2.33
                    2aR  (wrong-speaker alpha, 4) libriB 48.91 .13 | testother 50.45 .27 | libriA 47.78 | him 65.41
Gain vs comparator: libriB +3.90 (lb95 3.57, ub95 4.22), testother +2.51 (lb95 2.28, ub95 2.74) -> VERDICT 2a: PASS.
Control (b): 2aR - 2aN = 3.81 (lb90 3.58) > 0.25 -> by the rule: "attributed to the speaker-correct warp".
Literal reading (researcher.md 3.7): the rule's conditions (g >= 0.5 and lb > 0 on both sets) are met against the COMPARATOR std2 (aug + cmvn).
Not part of the rule, reported: std2 itself is worse than the plain base (struct2c base 0-7: 48.26 / 48.63); 2aN vs that plain base is
+3.55 / +1.45. His voice: no gain (64.67 vs 64.92, inside his sd). Stated limits (criteria 5.3): training alphas biased toward 1; the removal uses
the speaker's OTHER utterances (enrolment), which both sides had. First decisive PASS of this line of work (researcher.md 8.1).

## 16 Nov 03:10 diag_skip.py DONE (EXPLORATION, inference only, fp32, 3 seeds; logs_2c/diag_skip_2026-11-16.json)
libriB / testother: full232+SKIP both 46.40 / 47.57 | base284 47.61 / 48.12 | base232 47.98 / 48.72 | full232 48.99 / 49.41.
- skip path alone 101.03 / 99.73 (deletions .28 / .38, insertions 55.73 / 53.8); specialist path alone 65.91 / 66.24 (deletions 44.39 / 44.98,
  insertions .09 / .08): the two paths split the "nothing new" decision between them (one over-emits, one under-emits), so these two numbers say
  nothing about which path knows the phoneme. reviewer's "skip is the backbone, specialists a small correction" is NOT supported.
- blank bias on full232 (picked on libriA: -0.5): 48.82 / 49.36 -> a blank offset is not the cause of full's cost (gain ~0.1).
- per-phoneme gain vs base284 (error per reference phone, points): broad; best UH 5.5, AO 4.3, OW 3.9, G 3.8, EY 3.1, L 3.1; WORSE OY -1.4, Y -1.3,
  AH -1.0, CH -0.8, T -0.7, V -0.7, S -0.5. The most-split phonemes (T: 5 variants + TS; AH: SCHWA) are among the losers -> the gain does not come
  from the variant tokens; they may cost (fits the tok arm: 0.0 / -0.4).
## 16 Nov 03:17 probe_top.py smoke (fullskip232 s0, first 100 libriB utterances, EXPLORATION): phoneme IDENTITY from one path at the full model's emit
frames: both 42.01 | specialists only 48.6 | direct path only 55.82 | head middles replaced by their utterance mean 58.99. -> both paths carry
identity, the specialists more; the evidence handed down from the tree heads is heavily USED by the trained model (usage, not proof it is needed).
Smoke OOM on the structured top in fp32 (593 MiB cap) -> struct / mlp39 / dense53 probes NOT run tonight (his 03:1x "avoid as much gpu/cpu");
launched only: sanity + ablations + fresh-Linear probe on the fullskip232 and base232 trunks, 3 seeds, 6 epochs, + spec1 rank cut (inference).
## 16 Nov 03:17 PREDICTION written before the rest of TEST 2c-2 is read (arithmetic from the 19:06 / 22:3x tables, not from 2c-2 data): fullskip at
655,924 has a 176-wide trunk; base178 = 49.27 / 49.75; gain of full+SKIP over base on an equal trunk at 232 = 1.72 / 1.05 -> expected fullskip176
~47.55 / ~48.70 vs base232 48.26 / 48.63 = about +0.7 libriB, about -0.1 testother -> expected verdict FAIL on testother (rule: >= 0.5 on both).
The test stays locked and paused; this line only records what I expect. If right: at 0.66 MB the top costs more trunk width than it returns.

## 16 Nov 03:32 probe_top.py RESULT (EXPLORATION; frozen trunks, saved checkpoints, fp32; ran 03:17-03:26, ~500 MiB; logs_2c/probe_top_2026-11-16.jsonl)
Sanity (known answer): each trunk + its original top reproduces diag_skip's libriB exactly, 6 of 6 (e.g. base232 s0 47.83 = 47.83).
1. PATH ABLATIONS on full232+SKIP, 3 seeds, libriB / testother (full model 46.40 / 47.57): phoneme identity from the specialists alone 53.52 / 54.47;
   from the direct path alone 58.12 / 58.78; head middles replaced by their utterance mean 60.50 / 60.49. -> both paths carry identity, the
   specialists more; the trained model uses the tree heads' handed-down evidence heavily (usage shown; need not shown).
2. FRESH LINEAR (39 + blank, 6 epochs, 9,320 numbers) on the frozen trunk: trunk trained under full+SKIP 51.81 / 50.77 / 50.66 libriB (mean 51.08),
   52.14 / 51.67 / 51.28 testother (51.70). Black-box base232 trunk: s1 48.37 / 49.41, s2 49.60 / 49.58; s0 did NOT converge (loss stuck ~5,
   PER 94.31: a failed probe, not a result). Paired seeds 1, 2: the structure-trained trunk is WORSE for a plain readout by 2.40 and 1.06 libriB,
   2.26 and 1.70 testother. The probe itself is ~0.8-1.1 short of the as-trained base (47.26, 48.84), so absolute values are loose; the gap is not.
   -> "train with the structure, ship trunk + Linear" is NOT supported: the gain sits in the top (specialists + tree evidence + direct path
   together), and the trunk has adapted to that top. Route C1 of REVIEW_REPORT closed as NOT KEPT, OPEN (reopen: a probe that also reads the blocks).
3. RANK CUT of the GRU -> specialists matrix (864 x 232 = 200k numbers), no retraining, libriB mean of 3: full 46.40 | rank 128 46.47 | rank 64 46.77 |
   rank 32 47.58 | rank 16 49.93 | rank 8 54.08. -> a 64-wide shared layer in front of the specialists (70k numbers instead of 200k) loses ~0.4
   BEFORE any retraining: the top can likely be made ~130k numbers cheaper.

## 17 Nov 02:21 trainset_check.py (EXPLORATION; CPU inference, fp32, greedy, eval mode, no augmentation; logs_base5/trainset_check_2026-11-17.json)
PER x100 (hyp/ref). fit / cal = random 300 / 150 utterances of the 5.4 h pool; tc100 = random 300 utterances of train-clean-100 (seed 0).
| model | fit | cal | tc100 | libriA | libriB | testother |
| base_P655924_s0 (0.66 MB, 5.4 h) | 22.19 (.948) | 24.21 (.947) | 30.38 (.931) unseen speakers | 48.09 (.865) | 47.25 (.876) | 47.47 (.877) |
| base_P655924_s1 | 23.99 (.947) | 25.10 (.946) | 31.39 (.93) unseen | 48.64 (.861) | 48.44 (.883) | 48.65 (.876) |
| base_P1700000_s0 (2 MB, 100 h) | 25.18 (.952) unseen speakers | 26.90 (.95) unseen | 23.98 (.947) its training pool | 43.09 (.881) | 42.07 (.888) | 42.76 (.892) |
Known answer: libriA / libriB reproduce the fp32 numbers above in this file (s0 .4810 / .4725; 100 h .4308 / .4205). Phones per second 9.49-9.96 on every set.
## 17 Nov 02:28 labelcmp_2026-11-17.py (labels + CPU inference; logs_base5/labelcmp_2026-11-17.json)
Test references (our aligner, realcut_*.npz) vs MFA TextGrid phones of the SAME files: disagreement 4.23 % (libriA), 4.93 % (libriB), 4.90 % (testother)
of our phones; MFA / ours length .997 / .998 / .996; top differences AH<->AE, AA->AH, UW->AH / IH, R<->ER, DH->TH (weak forms). MFA symbols outside our 39: SPN x2.
PER vs our refs | vs MFA refs: base_P655924_s0 libriA 48.09 | 46.71, libriB 47.25 | 45.85, testother 47.47 | 45.80; base_P1700000_s0 43.09 | 41.41, 42.07 | 40.48, 42.76 | 40.90.
## 17 Nov 02:3x where the test sets come from (file paths in realcut_*.npz): libriA = 300 utterances, 4 speakers of dev-other (116, 1255, 1585, 1630);
libriB = 300 utterances, 6 speakers of dev-other (4572, 4831, 5543, 5849, 6123, 6267); testother = 330 utterances, 33 speakers of test-other.
On disk: dev-clean 40 spk / 2703, dev-other 33 / 2864, test-clean 40 / 2620 (unused by the base5 family), test-other 33 / 2939, train-clean-100 251 / 28539.
READING: training is clean-pool only, every reported set is the hard pool. Three gaps: fit (22-25 % on own training speakers), unseen clean speakers
(+6-8 at 40 speakers, +1-3 at 251), clean -> hard (+15-17 at both sizes). Label pipelines agree (1.4-1.9 points). Plan: docs/PLAN.md.
Base s0 log: 60 epochs x 84 batches = 5,025 updates, final training loss 1.2071 (with augmentation); 100 h run 37,828 updates, loss 1.0909.

## 17 Nov 12:3x-13:0x PHASE 0 (the agent on reviewer 5.5; docs/PLAN.md section 2; CPU nice 19, 2 threads; EXPLORATION / CALIBRATION)
0a SETS (phase0_sets.py; logs_base5/phase0_sets_2026-11-17.json; seed 20261117): cleanS = 20 train-clean-100 speakers (10 F / 10 M) x 15 = 300 utts,
   1.01 h, 35,655 phones, MFA labels, cache base5_cleanS.npz; its speakers listed in cleanS_EXCLUDE_speakers.txt (leave every future training set);
   cleanS is TRAINING-speaker data for base_P1700000_s0 (100 h). cleanT = test-clean 40 speakers x 8 = 320 utts, 0.757 h, 26,517 phones,
   base5_cleanT.npz, NOT scored until a finalist is frozen. Disjointness assert vs dev-clean / dev-other / test-clean / test-other speaker dirs passed.
0d PADDING (phase0_pad_sched.py; logs_base5/phase0_pad_sched_2026-11-17.json; base232 struct2c s0): clip alone vs + 400 zero (normalised) frames:
   log-prob differences only in the last 8 output frames (max 3.09 libriA / 8.45 cleanS; 0.0000 from frame 9 from the end on); greedy PER on 100
   clips each: libriA 49.01 alone = 49.01 padded; cleanS 32.84 = 32.84. Mismatch confined to the last ~160 ms; no PER effect on these clips.
0i SCHEDULE (same file): OneCycle total_steps 5,150 vs 5,024-5,035 real updates in every 60-pass run (97.6 % of the cycle; lr_end ~5e-6, not ~1e-8);
   100 h run 37,828 / 38,110. Best epoch 52-60 of 60 in almost every run; the 100 h model's best = its last epoch (20 / 20): the end points were still
   improving -> supports plan 1a (longer schedule).
0c SIZE (phase0_size.py + phase0_compact.py, onnx-venv; logs_base5/phase0_size_2026-11-17.json; first seed of each family; files in
   CACHE/phase0_export/). Bytes: T8 = torch dynamic int8 state_dict (today's form; convs + LayerNorm fp32; struct models also store 2 fp32 masks);
   O32 / O8 = ONNX fp32 / onnxruntime quantize_dynamic QInt8 (GRU appears to stay fp32: inferred from the bytes, not checked op by op);
   COMPACT = computed, not a file: every matrix 1 byte per STORED number (mask = 1 only) + 4-byte row scales + 4-byte vectors.
   | model | params | T8 | O32 | O8 | compact | PER libriA100 / cleanS100: fp32 | T8 | O8 |
   | base232 | 659,616 | 1,010,037 | 2,658,229 | 1,683,696 | 689,824 | 49.18/34.04 | 49.35/34.02 | 49.15/34.06 |
   | base284 | 955,132 | 1,380,597 | 3,840,301 | 2,465,811 | 992,048 | 47.54/32.77 | 47.77/32.78 | 47.69/32.78 |
   | full232SKIP | 1,148,552 | 2,487,392 | 4,633,616 | 2,224,043 | 1,014,040 | 48.00/31.65 | 47.94/31.69 | 48.17/31.55 |
   | full066 | 829,881 | 2,088,664 | 3,358,621 | 1,501,818 | 688,018 | 50.93/36.09 | 50.79/36.02 | 50.88/36.03 |
   | base100 | 1,708,404 | 2,284,917 | 6,853,389 | 4,484,243 | 1,758,736 | 43.17/26.06 | 43.20/26.05 | 43.02/25.89 |
   (100 random clips each, seed 1; O32 = fp32 exactly: export known answer.) READING: int8 costs <= 0.2 PER in every form. "Same size" depends on
   the form: T8 full232SKIP = 1.80x base284, O8 0.90x, compact 1.02x; TEST 2c pair compact 688,018 vs 689,824 (equal). Budget S (plan 0c) =
   T8 of base232 = 1,010,037 bytes until a runnable compact form exists (GRU as MatMuls so it is quantised; packed masked matrices).
   RAM: onnxruntime session RSS delta 14-23 MB at load, 18-30 MB after one 10 s clip: runtime + whole-clip activations, NOT model RAM; a streaming
   export is needed for the RAM half of his limit. ru_maxrss column INVALID (child inherits the parent's peak).
0b-2a + 0h (phase0_2a.py; logs_base5/phase0_2a.jsonl + phase0_2a_2026-11-17.json; inputs built with the unchanged vtln_grid / spk_stats /
   removal2a pipeline for cleanS and author2). 2aN (warp + floor) vs std2 (per-speaker CMVN comparator), 8 seeds each, mean PER:
   cleanS 31.73 vs 34.64, gain +2.91 (every seed, 2.57-3.24) | author2 72.75 vs 74.25, +1.50 (every seed, 0.33-2.15) | him (10 clips) 64.55 vs 64.55,
   0.00 (-5.57..+4.26: too small). cleanS warps: F median 0.92, M median 1.04 (sex separation as on the training pool); author2 estimate 1.00 on all 18.
   0h fixed warps on author2 (2aN s0-s7, floor after warp): 0.80 78.31 | 0.88 75.07 | 0.96 72.69 | 1.00 72.77 | 1.04 73.30 | 1.12 75.62 | 1.20 78.24:
   minimum at 0.96-1.00, symmetric rise -> the warp is NOT his bottleneck (diagnostic only, never his setting).
author2 DIAGNOSIS (17 Nov ~14:3x; base232 s0 75.45 = S 50.08 D 20.01 I 5.37, hyp/ref .854; base100 70.53). Per passage: the 11 passages of memo
   15_30_44 have 7.0-8.8 truth phones/s and PER 68-80 (sub + del): real difficulty. 4 of the 7 passages of memo 15_00_12 have 1.4-4.3 truth phones/s
   with large insertions (e.g. 9.8 s, 5 words: PER 243, I 26 on 8 phones; 45.5 s, 35 words: I 62): UNVERIFIED that the language-model transcript misses
   speech there; would verify: local ASR word count per passage vs truth words. Until then report author2 also as "15_30_44 only".
   CORRECTION 17 Nov ~14:4x (same day, check run): the "transcript misses speech" reading is REJECTED. Phonon-2.5-English word counts on the 7
   passages of 15_00_12 match the truth (5/5, 30/28, 35/38, 45/43, 89/89, 56/54, 17/15 truth/ASR). Low phones/s = long pauses (9.8 s with 5 words).
   So the insertions are the model emitting phones in NON-SPEECH stretches (pauses, car noise). Cause unknown; candidate (UNVERIFIED): the causal
   3 s normaliser stretches near-silent noise to speech level. Would verify: insertion rate inside silent frames vs normaliser window / floor,
   and the same passages with the silences cut.
   SILENCE CHECK (author2_silence_check.py, 17 Nov ~14:5x): pauses (>= 300 ms below 10th pct + 7 dB) = 31 % of author2 frames, but only 134 / 3,401
   (base232) and 105 / 3,312 (base100) output phones fall in them; cutting pauses to 100 ms: PER 75.45 -> 74.67 and 70.53 -> 70.83, insertions
   5.37 -> 5.65 and 4.14 -> 4.07. -> normaliser-in-silence candidate REJECTED (removing the pauses does not remove the errors). The extreme
   per-passage PERs came from tiny truth counts (8 phones). author2 error = substitutions 45-50 + deletions ~20 on his speech; cause not separated
   (accent, microphone / car, conversational speech, dictionary-pronunciation truth). This is the gap 2-III targets.
0b RESCORE (phase0_score.py; logs_base5/phase0_score.jsonl, per-utterance counts + deletion positions; log-prob caches CACHE/phase0_lp/ for seed 0
   of each family). Greedy, CPU fp32, mean PER (sd over seeds). cleanS = fresh speakers for every family except base100 (its training pool).
   | family | seeds | cleanS | libriA | libriB | testother | him |
   | base232 (struct2c base) | 8 | 32.71 (.33) | 48.74 (.38) | 48.18 (.54) | 48.58 (.17) | 62.87 (1.83) |
   | base232 stage 1a | 5 | 32.71 (.48) | 48.36 (.28) | 47.91 (.71) | 48.20 (.59) | 62.03 (1.42) |
   | base284 | 3 | 32.05 (.24) | 48.25 (.29) | 47.62 (.11) | 48.12 (.22) | 61.75 (1.89) |
   | full232SKIP | 3 | 30.98 (.09) | 47.88 (.09) | 46.40 (.19) | 47.57 (.13) | 62.40 (2.75) |
   | full232 | 3 | 33.69 (.44) | 49.37 (.26) | 48.99 (.30) | 49.40 (.29) | 63.82 (3.32) |
   | full066 | 8 | 34.92 (.30) | 50.16 (.20) | 49.96 (.29) | 50.44 (.35) | 65.04 (1.46) |
   | lite | 8 | 32.49 (.34) | 48.58 (.18) | 48.00 (.35) | 48.39 (.31) | 61.64 (1.53) |
   | tok | 8 | 32.96 (.38) | 48.97 (.25) | 48.15 (.30) | 48.98 (.25) | 62.91 (2.37) |
   | base100 | 1 | 25.22 (train spk) | 43.09 | 42.07 | 42.76 | 56.07 |
   READING (EXPLORATION): full232SKIP vs base284 (compact bytes 1,014,040 vs 992,048): cleanS +1.07 on FRESH speakers (plan guess ~+1), libriA +0.37,
   libriB +1.22, testother +0.55: the structure+direct-path gain is not specific to the hard pool. Still no control separating phonetics from extra
   paths (screen 2-II). full066 (TEST 2c full) worst on every set, as scored. lite / tok within ~0.3 of base on clean. Known answer: base5 s0 libriA
   48.09 reproduced exactly before the run.
2-III RUN 1 (p2iii_peruser.py base232 = struct2c base s0; logs_base5/p2iii_base232.jsonl; 17 Nov 13:0x-15:1x; SCREEN, EXPLORATION, true transcripts)
   10 dev-other speakers not in libriA / libriB (700 6841 3663 6467 4570 3915 8254 1701 4153 1650), test = last >= 180 s per speaker, 40 Adam steps,
   batch 8; lr per arm chosen on speakers 1686 + 3660 at 60 s: U1 1e-3, U2 3e-4, F 3e-4. Base N pooled 48.84. Pooled gain (speakers better):
   | enrolment | U1 output 40x40 | U2 input-band 80x80 | F fine-tune all |
   | 30 s | -0.41 (3/10) | +1.33 (8/10) | +1.22 (7/10) |
   | 60 s | +0.73 (7/10) | +1.22 (7/10) | +2.51 (8/10) |
   | 300 s | +1.69 (9/10) | +2.13 (10/10) | +5.45 (10/10) |
   CAVEAT: U1 and U2 chose the LOWEST value of their lr grids (not bracketed): run 2 extends both grids downward (U1 from 1e-4, U2 from 3e-5),
   everything else unchanged; read U1 / U2 only after run 2. Plan guess was "<= 1 min: U ahead of F"; F's own guess "5 min 8-15 on him" was for him.
0e BEAM (phase0_beam.py; logs_base5/phase0_beam_2026-11-17.json; selftest: beam = brute force on 200 tiny cases; 100 random utts per set, seed 2;
   no language model). Greedy -> beam32 PER: base232 s0 cleanS 32.62 -> 32.18 (beam64 32.13), libriA 48.55 -> 48.14, libriB 47.12 -> 46.29 (64: 46.27),
   testother 47.59 -> 47.24; full232SKIP s0 cleanS 30.26 -> 29.81, libriA 48.50 -> 48.16, libriB 46.04 -> 45.66, testother 47.43 -> 46.86;
   base100 cleanS 24.30 -> 23.70, libriA 43.40 -> 43.13, libriB 42.08 -> 41.65, testother 42.42 -> 42.03; him (10 clips) noisy (+-0.7).
   Gain 0.27-0.83 (mean ~0.45), always from fewer deletions; beam64 adds <= 0.05; the same size for every model -> no ranking change.
0f DELETION POSITIONS (from phase0_score.jsonl del_pos; position = midpoint of aligned neighbours, a proxy): share of deletions in the first / last
   0.5 s vs approx. frame share there: base232 s0 cleanS 0.056 vs 0.083; base5 s0 cleanS 0.049 vs 0.085; author2 (tok s0, base100) 0.022-0.023 vs 0.03.
   Deletions are NOT concentrated at clip boundaries (guess was the opposite). By phone length: not done (needs reference durations).
2-III RUN 2 (grids extended; same file): U2 optimum now interior (3e-4; tune 1e-4 45.88, 3e-4 45.13, 1e-3 45.70) -> run-1 U2 numbers stand.
   U1 tune PER rises monotonically with lr (1e-4 47.17 ... 3e-2 51.76): bracketed by lr 0 (= N) and 1e-4. U1 at 1e-4: 30 s +0.65 (7/10),
   60 s +0.76 (8/10), 300 s +0.70 (10/10). F (+1.22 / +2.51 / +5.45) beats both from 60 s on; at 30 s U2 +1.33 vs F +1.22 (tie).
   READING (screen, rule 53): this FORM of the output layer is NOT KEPT, mechanism not shown yet; PROXY caveat: these are mostly native speakers,
   the author's idea targets accent (consistent phone swaps). Mechanism check running: p2iii_oracle.py (each form fitted on the test clips = ceiling).
2-III FORM 2, RUN 1 (p2iii_sparse.py, 17 Nov 15:4x; discussion: the agent sparse KC writes + the agent per-unit gain; SCREEN, EXPLORATION).
   Frozen base232 s0; s = kWTA(relu(g @ P)), P random 232 x 2000 (seed 1117, 0 stored bytes); per-user W 2000 x 40 (80,000 numbers) added to logits.
   Known answer W = 0 == base: passed. Unit activity (k = 100) over 200 fit utts, relative to mean: median 0.463, max 11.354, 1 never active.
   Pooled gain vs N (speakers better), same 10 speakers / clips / 40 steps as form 1:
   | enrolment | S5 (5 % active) | S20 | SD (dense, relu) | S5g (S5 + per-unit gain) |
   | 30 s | +0.86 (7/10) | +0.05 (6/10) | -0.19 (5/10) | +0.60 (6/10) |
   | 60 s | +1.36 (10/10) | +1.11 (7/10) | +0.82 (7/10) | +0.97 (9/10) |
   | 300 s | +1.78 (10/10) | +2.74 (10/10) | +2.73 (10/10) | +1.10 (10/10) |
   INVALID AS A COMPARISON: every arm picked lr 3e-4 = the LOWEST grid value (not bracketed) -> run 2 with grid 1e-5 .. 1e-3. Pattern to check in
   run 2 (not a result yet): sparser is better at small enrolment, denser at 5 min, as the fly's specificity argument predicts.
2-III FORM 2, RUN 2 (grid 1e-5 .. 1e-3; chosen S5 1e-4, S20 3e-5, S5g 3e-4 interior; SD 1e-5 = edge, its tune curve rises monotonically -> bracketed
   by lr 0 = N, like U1). Tuning PER spread across the grid only 46.04-46.87 on 2 speakers (weak signal). Pooled gain vs N (speakers better):
   | enrolment | S5 | S20 | SD | S5g | (form 1: U2 | F) |
   | 30 s | +0.51 (7/10) | +0.54 (7/10) | +0.34 (8/10) | +0.60 (6/10) | +1.33 | +1.22 |
   | 60 s | +0.77 (8/10) | +0.81 (8/10) | +0.31 (7/10) | +0.97 (9/10) | +1.22 | +2.51 |
   | 300 s | +0.84 (10/10) | +0.69 (8/10) | +0.43 (8/10) | +1.10 (10/10) | +2.13 | +5.45 |
   READING (screen): sparse > dense and per-unit gain helps, by tenths of a point; none approaches F. Run 1's "sparse wins small, dense wins large"
   came from one shared lr (3e-4) and does not survive per-arm tuning. One lr per arm picked at 60 s cannot serve 30 s and 300 s (run 1 at 3e-4:
   S5 300 s +1.78, SD +2.73 > run 2): the recipe needs lr per enrolment size, or a stopping rule, tuned on more than 2 speakers.
   STATUS: output-side per-user forms (U1, S*) NOT KEPT, OPEN (mechanism not shown; candidate: the speaker's differences sit in the acoustics,
   which only input-side U2 and F reach; oracle check p2iii_oracle.py still running). PROXY: native LibriSpeech speakers, not his accent.
2-III FORM 2 CHECKS (p2iii_sparse_checks.py; logs_base5/p2iii_sparse_checks.json; the agent's checks, 17 Nov 16:1x):
   (1) per-speaker gains: no single speaker carries the sparse gains (S5g 60 s median +0.97, 300 s +0.99, all speakers >= 0.00 at 60 s); F gains
   far more on most speakers but costs 1-2 at 30 / 60 s (-3.06, -1.35, -1.02, -0.53).
   (2) substitutions per 100 ref phones, close (same broad class) | distant: N 15.80 | 15.89; 60 s S5 15.73 | 15.26, S20 15.78 | 15.32, SD 15.82 |
   15.89; 300 s S5 15.60 | 15.34, S20 15.78 | 15.41, SD 15.72 | 15.80. Sparse and denser codes do not differ on close pairs; gains are on distant
   pairs -> the fly's specificity signature is ABSENT: form 2 closed as "mechanism not shown". Cross-speaker check running (p2iii_cross.py).
   the agent (review notes 16:16): sparse forms harm far less at low enrolment (30 s minima: S5g -0.65, U2 -0.42, F -3.06; 60 s: S5g +0.00, F -1.35) ->
   adopted as its OWN criterion for the run on his voice (worst-case change next to the mean, threshold stated before data).
2-III ORACLE CEILING (p2iii_oracle.py; logs_base5/p2iii_oracle_base232.jsonl; each form fitted on the speaker's OWN test clips, 200 steps; an
   unfair upper bound, never a result). 10 speakers, mean PER: N 48.85 | U1 output remap 37.00 (gain 11.86, range 7.69-18.52) | U2 input-band
   29.45 (19.40, 11.37-30.57) | F 1.29 (memorises the clips: no ceiling). READING: a 40x40 phone remap CAN express ~12 points of each
   speaker's error on these clips; fitted fairly from 30 s - 5 min of other clips it gains ~0.7. The limit is generalisation from enrolment
   (what the update learns from few examples), not what the form can express. Mechanism for the fair gap: not shown.

### 17 Nov 16:53 EET  2-III CROSS-SPEAKER CHECK (EXPLORATION; p2iii_cross.py -> logs_base5/p2iii_cross_base232.jsonl; base232, 10 hard speakers in a ring:
enrol on A, score A and the next speaker B; gain = PER(no adaptation) - PER(adapted), positive = better; mean (min) over 10 pairs)
| arm | enrol | own gain | other-speaker gain | own > 0 | other worse |
| U2  |  60 s | 1.22 (-1.21) | -1.47 (-2.34) | 7/10 | 10/10 |
| F   |  60 s | 2.61 (-1.35) | -3.84 (-7.11) | 8/10 | 10/10 |
| S5g |  60 s | 1.02 (0.00)  | -0.56 (-1.39) | 9/10 | 6/10 |
| U2  | 300 s | 2.20 (0.30)  | -1.32 (-2.84) | 10/10 | 10/10 |
| F   | 300 s | 5.54 (1.51)  | -2.91 (-6.66) | 10/10 | 9/10 |
| S5g | 300 s | 1.15 (0.20)  | -0.31 (-1.34) | 10/10 | 7/10 |
READING: every arm's gain is speaker-specific (it helps A and hurts B), so it is not generic adaptation to the hard-pool recordings (the agent's confound
ruled out for these arms). F gains most and harms others most; S5g gains least and harms least. Consequence for use: an adapted layer must be tied to its
enrolled user (per-user file, voice-ID gate), never shared across speakers.

### 17 Nov 17:2x EET  PHASE 1a schedule budget (base6.py, 5.4 h, d 232, 659,616 params, RIGHT 2, causal norm, seed 0, final fp32 greedy; logs_base6/)
| cell | GPU time | fit | cleanS | libriA | libriB | testother | author2 | him |
| e60  | 470 s  | 22.84 | 32.57 | 49.23 | 47.92 | 48.24 | 75.48 | 59.67 |
| e180 | 1390 s | 20.74 | 30.44 | 47.19 | 46.16 | 47.04 | 74.70 | 64.92 |
DECISION (plan rule 1a, selection libriA + cleanS): ADOPT 180 passes (-2.13 cleanS, -2.04 libriA; one seed, seed sd ~0.5-0.6). Per the plan: the old end
point was improvable; survivors get re-screened. Noted: him (10 old clips, 99 words) +5.25 worse while author2 (9.76 min) -0.78 better; small-set noise is the
likely reading (UNVERIFIED: would check with a second seed in 1d). Still curve-limited at 180? last 12 epochs moved libriA 46.76 -> 46.69 (flat).

### 17 Nov 18:4x EET  PHASE 1b look-ahead x input norm (base6.py, 180 passes, 5.4 h, d 232, seed 0, final fp32 greedy; logs_base6/e180*_s0.log)
| cell | look-ahead | GPU s | fit | cleanS | libriA | libriB | testother | author2 | him |
| RIGHT 2, causal (= 1a e180) | 192 ms | 1390 | 20.74 | 30.44 | 47.19 | 46.16 | 47.04 | 74.70 | 64.92 |
| RIGHT 12, causal | 992 ms | 1380 | 20.12 | 29.23 | 46.70 | 44.73 | 45.89 | 73.87 | 63.28 |
| RIGHT 2, fixed | 192 ms | 1025 | 19.81 | 30.87 | 48.52 | 46.18 | 48.19 | 83.38 | 59.02 |
| RIGHT 12, fixed | 992 ms | 1013 | 19.48 | 30.44 | 47.52 | 45.11 | 47.23 | 80.47 | 60.00 |
READING: causal 3 s norm beats fixed training-set norm on hard and on author2 (fixed +8.7 / +6.6 worse on author2: a global norm does not follow a new
recording channel); fixed fits its training speakers better (19.5-19.8) but generalises worse. Look-ahead 992 ms vs 192 ms (causal): better on all 7 sets,
cleanS -1.21, libriA -0.49 (selection sets), testother -1.15, author2 -0.83; smaller than my guess (2-3) and libriA is within ~1 seed sd. RAM: GPU peak 654 MiB
in all cells; the extra look-ahead buffer is 40 frames x 232 x 4 blocks (small). DECISION (pending seed pair): candidate frozen cell = RIGHT 12 causal
(his allowance: 1-2 s delay "if it helps"); seed pair s1 for RIGHT 2 and RIGHT 12 causal now, before freezing (adopt RIGHT 12 if its mean over s0+s1
is better on cleanS + libriA; otherwise RIGHT 2, the shorter delay). Fixed norm: NOT KEPT (OPEN only for a per-recording norm variant, not planned).

### 17 Nov 19:0x EET  PHASE 1c arm B data (base6.py MIX=B, mix_b(); CPU-checked with /tmp/mixtest.py logic; speakers from cache/base5_speakers.json)
B = 2.698 h clean (1,344 utts, 23 of the 40 dev-clean fit+cal speakers, seeded rng 1234) + 2.697 h hard (1,517 utts, the 23 dev-other speakers not in
libriA / libriB) = 5.394 h, equal to arm A; same utterances at speeds 0.9 (5.991 h) and 1.1 (4.903 h); labels 0-38 in both. Asserts: no test speaker,
labels equal across speeds, length ratio within 5 % of the speed. NOTE: the hard 23 include all 10 speakers of the 2-III ring (700, 1650, 1701, 3663,
3915, 4153, 4570, 6467, 6841, 8254): any 2-III on a B-trained model must use other speakers.

### 17 Nov 19:58 EET  PHASE 1b seed pair -> look-ahead DECIDED (s1 of RIGHT 12 resumed from its moved checkpoint at epoch 25; logs on scratch storage)
| cell | seed | fit | cleanS | libriA | libriB | testother | author2 | him |
| RIGHT 2 causal  | s0 | 20.74 | 30.44 | 47.19 | 46.16 | 47.04 | 74.70 | 64.92 |
| RIGHT 2 causal  | s1 | 20.45 | 30.48 | 47.37 | 45.58 | 46.51 | 74.00 | 60.33 |
| RIGHT 12 causal | s0 | 20.12 | 29.23 | 46.70 | 44.73 | 45.89 | 73.87 | 63.28 |
| RIGHT 12 causal | s1 | 19.66 | 29.45 | 46.41 | 44.19 | 45.96 | 74.12 | 61.64 |
Means RIGHT 12 - RIGHT 2: cleanS -1.12, libriA -0.73 (both seeds agree), libriB -1.41, testother -0.85, author2 -0.35, him -0.15.
DECISION (rule written 18:4x): RIGHT 12 (992 ms look-ahead) for every later arm. him differs 4.6 between two seeds of the SAME cell (64.92 / 60.33): the
10-clip 'author' set is too small to read single runs (the 1a 'him got worse' note was noise); author2 is the his-voice number to read.

### 17 Nov 20:35 EET  PHASE 1c training mix (base6 MIX=B, RIGHT 12 causal, 180 passes, seed 0; log logs_base6/e180_mixB_r12_s0.log)
| arm | fit (own training spk) | cleanS | libriA | libriB | testother | author2 | him |
| A clean 5.4 h, s0 | 20.12 | 29.23 | 46.70 | 44.73 | 45.89 | 73.87 | 63.28 |
| A clean 5.4 h, mean s0+s1 | 19.89 | 29.34 | 46.56 | 44.46 | 45.93 | 74.00 | 62.46 |
| B 2.7 h clean + 2.7 h hard, s0 | 28.93 (mixed pool) | 31.52 | 45.85 | 46.16 | 46.19 | 71.99 | 61.31 |
RULE (plan 1c, verbatim): "B better by >= 1 on libriA with <= 0.5 loss on cleanS: second seed pair, then B is the data recipe for every arm. Otherwise
park THIS mixture (not 'harder data does not help')". Met: neither condition (libriA -0.85 vs s0 / -0.71 vs mean, below 1; cleanS +2.18 / +2.18 worse,
above 0.5). -> PARKED: this 50/50 mixture at 5.4 h. Not a finding that hard data does not help: libriB got WORSE (+1.4) while libriA improved, and
B had half the clean speech. OPEN (one seed): author2 better by 1.9-2.0 with B; reopen condition = more hard speech without losing clean speech (add, not
swap: e.g. 5.4 h clean + 2.7 h hard, or train-other-500 later per plan step 2 of section 3), judged on cleanS + libriA + author2.

### 17 Nov 22:1x EET  PHASE 1d seed spread of the FROZEN base6 (RIGHT 12 = 992 ms look-ahead, causal 3 s norm, clean 5.4 h, d 232, 659,616 params)
| schedule | seeds | fit | cleanS | libriA | libriB | testother | author2 | him |
| 180 passes | 3 | 20.19 sd .57 | 29.56 sd .39 | 46.75 sd .36 | 44.76 sd .58 | 46.18 sd .45 | 74.11 sd .24 | 63.06 sd 1.32 |
| 60 passes  | 5 | 21.65 sd .51 | 30.99 sd .36 | 47.57 sd .20 | 46.45 sd .26 | 47.07 sd .17 | 73.08 sd .31 | 63.61 sd 2.68 |
(values per seed in phones_out/base6/e{180,60}_r12_causal_s*.json; sd = sample sd over seeds; 3-seed sd is rough.)
POWER TABLE (two arms, n seeds each, one-sided alpha 0.05, power 0.8: n = 12.37 sd^2 / delta^2, rounded up; sd from the 60-pass row, the screening schedule):
| delta (points) | cleanS (sd .36) | libriA (sd .20) | testother (sd .17) | author2 (sd .31) |
| 0.5 | 7 | 2 | 2 | 5 |
| 1.0 | 2 | 1 (use 2) | 1 (use 2) | 2 |
-> Phase 2 screens at 60 passes: 2 seeds per arm detect a 1-point effect on every set; a 0.5-point claim on cleanS needs 7 seeds per arm.
'author' (10 clips) sd 1.3-2.7: never decide on it alone. NOTE: author2 is BETTER at 60 passes (73.08) than at 180 (74.11): longer clean-only training moves
his voice the wrong way by ~1 point while every LibriSpeech set improves 0.9-1.7 (OPEN; candidate reading: overfitting to the clean pool's channel).

### 17 Nov 22:3x EET  PHASE 1 END: export of the frozen base6 (phase1_export.py, e180_r12_causal_s0; json logs_base6/phase1_export_e180_r12_causal_s0.json)
Same forms and method as phase0_size.py; PER on the same 100 libriA + 100 cleanS clips (subset numbers, not the full-set ones above).
| form | bytes | PER libriA | PER cleanS | RAM load | RAM after a 10 s clip |
| torch fp32 | - | 47.23 | 30.52 | - | - |
| T8 (torch dynamic int8: GRU + Linear int8, convs / LN fp32) | 1,010,565 | 47.46 | 30.29 | - | - |
| O32 (ONNX fp32) | 2,658,229 | 47.23 | 30.52 | 15.5 MB | 18.8 MB |
| O8 (ONNX dynamic int8) | 1,683,696 | 47.31 | 30.38 | 14.2 MB | 18.4 MB |
Known answer holds: O32 = fp32 exactly. int8 costs <= 0.23 points. Smallest runnable form = T8 at 1.01 MB (his limit 2 MB on the higher of disk and RAM:
the RAM figure of the ONNX forms, ~18 MB, is model + onnxruntime working memory in a fresh process, above the 2 MB line; per-process peak RSS 563 MB is
the onnxruntime import, not the model). OPEN: RAM of the T8 form inside the app runtime (sherpa / onnxruntime-mobile), and whether his 2 MB means model
bytes in RAM or total process growth (I read it as the model's own bytes, 14 Nov; to confirm with him once).

### 17 Nov 22:30 EET (date read) 2-I started. TIME CORRECTION: CRITERIA_2I.md says "~22:5x"; the build_2aJ.py log shows `date` = 22:16:20 EET when the
build started, after the criteria were written: the criteria were written ~22:1x EET, before any 2aJ_ / base6+2aN_ data (the order holds; only my
time label was wrong; the hashed file is left unchanged). 2aJ_ build OK (known answer: 2aJ_ <= original everywhere; mean change libriA .436, author2 .996 nat).

### 17 Nov 23:4x EET  2-I SCORED (CRITERIA_2I.md sha 0ad1b593... OK, score_2I.py sha 96db096f... OK; 6 runs at 60 passes, seeds 0+1; enrolled-speaker setting)
| arm | fit | cleanS | libriA | libriB | testother | author2 | him | hard (libriA, testother) |
| A warp + floor (2aN_) | 18.68 | 28.24 | 44.62 | 42.16 | 45.55 | 71.92 | 59.67 | 45.08 |
| B identity warp + floor (2aJ_) | 21.96 | 31.50 | 48.11 | 46.47 | 47.74 | 73.87 | 63.12 | 47.92 |
| C plain base6 (1d runs e60 s0, s1) | 21.22 | 30.66 | 47.47 | 46.27 | 47.03 | 73.22 | 62.30 | 47.25 |
Per seed (A / B / C) libriA 44.67, 44.56 / 48.49, 47.72 / 47.50, 47.44; testother 45.51, 45.58 / 47.74, 47.74 / 47.18, 46.89.
VERDICT by the rules: 1. A - B hard -2.84 -> WARP KEPT. 2. B - C hard +0.67 -> FLOOR adds nothing detectable on base6 (point estimate slightly
harmful, inside the 1.0 band). 3. his voice (reported only): author2 A - B -1.95, A - C -1.30, both seeds of A at 71.9 -> recorded OPEN (a gain on
author2 of >= 1 point, first time any arm moved author2 the right way by that much); him (10 clips) A - C -2.6 (noisy set).
Reading (CANDIDATE): the whole gain is the per-speaker warp; it also lowers the fit on the TRAINING speakers (21.2 -> 18.7), i.e. it removes
speaker variation the model otherwise spends capacity on. The test does not separate "warp alone" from "warp + floor" (no warp-only arm):
an exploration arm 2aW_ (warp, no floor) answers it. Limit: factors come from the speaker's OTHER utterances incl. later ones (enrolled
setting); the deployable version is warp from PAST audio only (plan row 2-I, next step).
Run note: b6_2aJ_s1 stalled ~33 min at start (D state, HDD sda saturated; scratch storage and the project disk are one disk), then ran normally; result valid.

### 18 Nov 00:1x EET  2-IV step 1 DONE (p2iv_lm.py; logs_base6/p2iv_lm_2026-11-17.json; EXPLORATION-grade screen, rule from the plan row)
Phone n-gram (interpolated Kneser-Ney, discount .75) from the 2,703 training transcripts (193,644 phones) of the 5.4 h set; CTC prefix beam W 16 with
shallow fusion (lam * LM + beta * length); n, lam, beta chosen on 150 libriB utts only. Known answers: KN sums to 1 (n 2-4); lam = beta = 0 == old beam.
LM bytes (2 per stored n-gram, lower bound): n2 2,540; n3 32,180; n4 149,332.
| model (180 passes) | chosen | set: greedy / beam / beam+LM |
| e180 s0 | n4 lam .7 beta 2.0 (GRID EDGE) | libriA 46.72/46.30/43.86; testother 45.89/45.57/43.44; cleanS 29.23/28.89/26.10; author2 73.82/73.82/73.49; him 63.28/63.61/67.21 |
| e180 s1 | n4 lam .4 beta 1.0 (interior) | libriA 46.42/46.11/43.46; testother 45.96/45.43/42.69; cleanS 29.46/29.00/25.46; author2 74.10/74.15/72.09; him 61.64/62.30/58.36 |
LM gain over the same beam on hard: s0 2.29, s1 2.69 -> rule (< 0.5 parks) NOT met: the n-gram is KEPT as the text-knowledge baseline his sequence
part must beat (same text, same bytes). Gain is ~3x the beam's own gain (0.3-0.5). his voice: author2 -0.3 (s0, heavier weight) / -2.1 (s1); the heavy
s0 setting hurts the 10 old clips (+3.6) -> candidate: lighter LM weight transfers better to him (OPEN). s0 at the grid edge: extended grid next.

### 18 Nov 00:1x EET  2aW_ WARP ONLY (EXPLORATION after 2-I; base6, 60 passes, seeds 0+1, enrolled setting)
| arm | fit | cleanS | libriA | libriB | testother | author2 | him | hard |
| W warp only (2aW_) | 17.88 | 27.57 | 44.11 | 41.51 | 45.09 | 71.19 | 57.70 | 44.60 |
| A warp + floor (2aN_) | 18.68 | 28.24 | 44.62 | 42.16 | 45.55 | 71.92 | 59.67 | 45.08 |
| C plain | 21.22 | 30.66 | 47.47 | 46.27 | 47.03 | 73.22 | 62.30 | 47.25 |
W is 0.5-0.7 better than A on every set: the floor is not needed (consistent with B - C +0.67). Front end for later arms: WARP ONLY.
W - C: hard -2.65, cleanS -3.09, author2 -2.03, fit -3.34. Next: the deployable warp (alpha from the speaker's PAST utterances only), first as
inference-only on the W models (train with leave-one-out alphas, test with past-only alphas), then retrain if the mismatch costs.

### 18 Nov 00:3x EET  DEPLOYABLE WARP, inference only (p2i_past.py; logs_base6/p2i_past_2026-11-18.json; EXPLORATION)
W models (b6_2aW s0, s1; trained with leave-one-out alphas), tested with alpha from: loo (= 2aW_ files; known answer reproduces the json PER
within 0.01 on 11 of 12 cells, author2 s0 0.03: CPU vs GPU fp32), PAST utterances only (stored order = time; first utterance alpha 1), or alpha 1.
Mean of 2 seeds: | set | loo | past | alpha 1 | plain base6 C |
| libriA | 44.11 | 45.06 | 48.08 | 47.47 |  | testother | 45.09 | 45.97 | 50.11 | 47.03 |
| cleanS | 27.57 | 28.21 | 31.23 | 30.66 |  | libriB | 41.49 | 42.36 | 52.24 | 46.27 |
| author2 | 71.20 | 71.21 | 71.20 | 73.22 |  | him (10 clips) | 57.70 | 65.08 | 59.02 | 62.30 |
-> past-only costs ~0.9 on hard, 0.6 on cleanS; still beats the plain model by 1.7 (hard) / 2.5 (cleanS). By position within the speaker: the first
utterance (nothing heard yet) is the worst; from the 8th on past ~= loo.
FINDING (his voice): the estimator gives HIS voice alpha 1.0 (loo == alpha 1 exactly on author2), so the author2 gain W vs C (-2.0) is NOT from warping
his voice: it comes from training on speaker-normalised data (the model sees fewer speaker differences in training). On the 10 old clips the
past-only alphas move off 1 early and hurt (+7.4; noisy set, 10 clips). OPEN: train-side-only warp (no test warp) as its own arm.
Cost note: the warp estimator scores 11 warps with a second model (stage-1a base, ~1 MB): not counted yet; vtln_self.py tests base6 as its own scorer.

### 18 Nov 00:5x EET  CACHE CLEAN-UP (the author: "yes you can delete old copies")
Deleted the full feature copies base5_2aI_ / 2aJ_ / 2aR_ / 2aW_* (44 files, 4.1 GB; rebuildable: build_2aJ.py <variant> / removal2a.py). Kept their
small alpha / stats files (rm2a_*, spk_*, vtln_*). base5_2aN_* (1.1 GB) is deleted after p2iv run 2 (it reads them). From now on warps are applied at
LOAD time (base5_next.py WARP=W|S; known answer passed: WARP=W == the 2aW_ copies bit for bit on libriA, author2, cal_sp1.1, checked before deletion;
warp_check() needs the copies and cannot be rerun now). build_2aJ.py refuses new copies.

### 18 Nov 01:1x EET  SELF-SCORED WARP (vtln_self.py + WARP=S load-time warp; b6_2aS s0, s1, 60 passes; EXPLORATION)
vtln_self known answer: old scorer through the new path max |diff| 3.1e-4 vs vtln_libriA.npz. Self alphas vs old-scorer alphas: corr .85-.99 on
fit / cal / libriA / libriB / cleanS, .75 testother; author2 1.0 (same); him (10 clips) ALL 0.80 = grid edge (old scorer differed) -> suspect.
| arm | fit | cleanS | libriA | libriB | testother | author2 | him | hard |
| S self-scored warp | 17.70 | 27.83 | 44.49 | 41.72 | 45.56 | 71.16 | 70.50 | 45.02 |
| W old-scorer warp | 17.88 | 27.57 | 44.11 | 41.51 | 45.09 | 71.19 | 57.70 | 44.60 |
| C plain | 21.22 | 30.66 | 47.47 | 46.27 | 47.03 | 73.22 | 62.30 | 47.25 |
-> base6 can score its own warps: 0.4 worse than the second model on hard (2 seeds; inside the ~1-point screen resolution), still -2.2 vs plain;
no second model needed (estimator cost = 11 forward passes per enrolment utterance, 0 extra bytes). his voice author2 same gain (alpha 1 both).
The 10 old clips: the self scorer puts all at the 0.80 edge and him gets +8.2 worse vs plain: the estimator needs a guard (grid edge / bracket
widening, or shrink toward 1 with few utterances) before it is used on his phone. OPEN.
18 Nov 01:4x: also deleted the closed experiment caches (the author's OK, his follow-up "did you not fix the drive issue"): cue*, fmt_*, g4_*, gate_*, n4_build,
phase0_lp, phones7x/8x_post, px115_*, seg13, segmel_* (10 GB; no script changed in the last 3 days reads them). KEPT: ph_* (labels2c.py, needed by 2-II),
realcut_* (2-III), bb142_* + buckeye/ (bb142.py label module), the originals, devother, base5_tc100 (100 h). base5_2aN_* goes when p2iv run 2 has cached its log-probs.

### 18 Nov 02:48 EET  2-IV run 2 (EXPLORATION; p2iv_lm.py, grid n 4,5 x lam .4-1.4 x beta 1-4, tuned on 150 libriB utts; logs_base6/p2iv_lm_2026-11-18_{plain,2aN}.json)
| model | best (n, lam, beta) | hard beam -> beam+LM (libriA / testother) | LM gain hard | cleanS beam -> LM | author2 beam -> LM | him beam -> LM |
| e60 plain s0 | 5, .7, 2.0 | 47.14 / 46.79 -> 44.59 / 43.62 | 2.86 | 30.33 -> 27.06 | 73.22 -> 72.97 | 63.93 -> 65.57 |
| e60 plain s1 | 5, .4, 1.0 | 46.94 / 46.36 -> 44.38 / 43.21 | 2.85 | 30.04 -> 26.41 | 73.39 -> 72.14 | 60.00 -> 61.31 |
| 2aN s0 (warp+floor) | 5, .7, 2.0 | 44.32 / 45.15 -> 41.01 / 41.83 | 3.31 | 27.89 -> 24.16 | 71.69 -> 70.78 | 61.31 -> 59.67 |
| 2aN s1 | 5, .7, 2.0 | 44.11 / 45.13 -> 40.94 / 41.65 | 3.33 | 27.74 -> 24.01 | 71.94 -> 72.57 | 57.38 -> 59.02 |
READING: (1) the LM gain STACKS on the speaker warp and is larger there (3.3 vs 2.9): warp + LM hard ~41.4 vs plain beam ~46.8 (-5.4). (2) n = 5 is the
top of the grid on all four (edge): n 6 untested. LM bytes (2 per stored n-gram, lower bound): n4 149,332; n5 372,508 -> n5 + a 1.0 MB model is
~1.4 MB, inside 2 MB; n6 must be sized before it is tried. (3) his voice: small and mixed: author2 PER change with LM -0.25 / -1.25 / -0.91 / +0.63 (3 of 4
better), him +1.64 / +1.31 / -1.64 / +1.64 (3 of 4 worse) -> far below the 2.9-3.3 on LibriSpeech: OPEN.
NEXT (CPU, after p2iii_replay; load 24 on 12 cores now): n 6 + bytes; lighter lam on author2 reported only (his set is never used for choosing).

### 18 Nov 03:3x EET  2-V TEST scored (CRITERIA_2V.md sha 92abe5c9 unchanged; score_2V.py; 20 runs b6v_{C,W}_d{0-4}_s{0,1}, all rc 0, no rerun, pad check ok in all)
1 h of whole speakers (8 per draw), plain C vs speaker warp W (alphas from the plain model trained on that hour only), gain = C - W (positive = warp better):
| set | C mean | gain per draw d0..d4 | mean | sd over draws | draws > 0 |
| hard (libriA+testother)/2 | 57.38 | 0.99 / 3.18 / 2.29 / 3.42 / 2.16 | +2.41 | 0.96 | 5/5 |
| cleanS | 44.56 | 1.50 / 3.48 / 2.84 / 4.34 / 3.00 | +3.03 | 1.03 | 5/5 |
| author2 | 78.63 | -0.49 / 0.84 / 0.20 / 1.29 / -0.84 | +0.20 | 0.89 | 3/5 |
| him (10 clips) | 76.10 | -5.41 / 2.13 / -3.44 / -1.31 / 1.31 | -1.34 | 3.17 | 2/5 |
VERDICT by the rule (mean g_d >= 1.0 over 5 draws): PASS = "the warp helps at 1 h". Rule applied literally (researcher 3.7): the title asks whether it
helps MORE at small data; the rule decides only "helps at 1 h". The 5.4 h reference gap is 2.23 (2aS vs C, 60 passes); 2.41 at 1 h is within one
draw-sd of it: NOT shown to help more. His voice: no gain at 1 h (author2 +0.20, 3/5; him -1.34). Next by the plan row: 0.5 h and 2 h; then 2-V again
with the 2-II survivor.

### 18 Nov 04:0x EET  INTERIM (no verdicts; incomplete)
2-II seed 0 only (CRITERIA_2II sha 23e435fd; verdict needs seed 1, running): PER libriA / cleanS / libriB / testother / author2 / him
  S 46.70 / 29.56 / 44.89 / 46.19 / 72.74 / 60.66;  G 47.25 / 29.94 / 45.71 / 46.62 / 74.60 / 68.52;  P 47.27 / 30.75 / 46.01 / 46.81 / 74.50 / 65.25
p2iii_replay (EXPLORATION), 6 of 10 pairs (COVERAGE INCOMPLETE). Known answer: F rerun = p2iii_cross F exactly on all 4 compared cells.
  mean gain own / other (positive = better), 60 s: F +2.71 / -4.35; FR +2.80 / -1.07; FA0.01 +2.91 / -3.76; FA0.1 +2.88 / -2.69; FA1 +2.15 / -0.94; FRA0.1 +2.51 / -1.01
  300 s: F +5.85 / -2.80; FR +4.93 / -0.80; FA0.01 +5.85 / -2.71; FA0.1 +5.39 / -1.86; FA1 +3.11 / -0.69; FRA0.1 +4.09 / -0.79

### 18 Nov 04:26 EET  2-II SCREEN scored (CRITERIA_2II sha 23e435fd unchanged; score_2II.py -> logs_base6/score_2II.json; 6 runs, all rc 0, valid)
mean of seeds 0,1, PER: libriA / cleanS / libriB / testother / author2 / him / fit (own training speakers)
| S his readout (1,376,393 B) | 46.72 / 29.89 / 45.35 / 46.49 / 72.73 / 59.35 / 18.30 |
| G generic readout (1,376,909 B) | 47.24 / 30.14 / 45.77 / 47.04 / 73.91 / 64.10 / 19.21 |
| P plain widened (1,380,389 B) | 47.20 / 30.67 / 46.24 / 46.91 / 74.05 / 64.43 / 20.83 |
gain vs P: S +0.48 / +0.78 / +0.90 / +0.43 / +1.32 / +5.09; G -0.04 / +0.53 / +0.47 / -0.13 / +0.14 / +0.34.
VERDICT AS LOCKED (rule 1: >= 0.5 on BOTH libriA and cleanS): S misses on libriA by 0.02 (0.48) -> "PARK readout line (neither S nor G beats P)".
Rule 53: NOT KEPT by the screen, OPEN (cause unknown; not refuted). Rule applied literally (researcher 3.7): the screen did not show S beats P.
HIS PARITY RULE (CRITERIA_2II_AMENDMENT_2026-11-18.md, written after seed 0 was seen): S is not worse than G by 0.5 on either decision set (S is
better than G on every set: libriA 0.52, cleanS 0.25, author2 1.18, him 4.75) -> under his reading S is KEPT FOR STACKING. A parity or win CLAIM still
needs a locked test on fresh seeds. Caveat: him (10 clips) is noisy (G seeds 68.52 vs 59.67). S is the only arm better than P on all 7 sets.
FOLLOW-UP (EXPLORATION, my choice): arm R (shuffled head-to-specialist wiring, fan-in kept, same bytes) seeds 0,1 started 04:27 (lock taken 04:27:03).

### 18 Nov 04:4x EET  2-III FINE-TUNING WITH REPLAY / ANCHOR (EXPLORATION; p2iii_replay.py -> logs_base5/p2iii_replay_base232.jsonl; COVERAGE 20/20 rows,
all 6 arms each; known answer: F = p2iii_cross F exactly). Ring of 10 hard speakers, base232, 40 Adam steps, lr 3e-4. gain = PER(N) - PER(arm), positive = better.
| arm | 60 s own (min, >0) | 60 s other (min, worse) | 300 s own (min, >0) | 300 s other (min, worse) | weight drift 300 s |
| F fine-tune | +2.61 (-1.35, 8/10) | -3.84 (-7.11, 10/10) | +5.54 (+1.51, 10/10) | -2.91 (-6.66, 9/10) | 2.84 |
| FR + replay 8 clean training utts / step | +2.73 (-0.14, 9/10) | -0.77 (-2.30, 7/10) | +4.83 (+1.26, 10/10) | -0.87 (-3.81, 7/10) | 2.51 |
| FA0.01 anchor | +2.72 (-0.80, 8/10) | -3.40 (-6.27, 10/10) | +5.61 (+1.85, 10/10) | -2.80 (-6.23, 9/10) | 2.53 |
| FA0.1 | +2.79 (-0.75, 9/10) | -2.35 (-5.02, 10/10) | +4.99 (+1.22, 10/10) | -2.15 (-5.42, 10/10) | 1.44 |
| FA1 | +2.19 (-0.05, 9/10) | -0.92 (-2.93, 7/10) | +2.93 (+0.54, 10/10) | -0.83 (-2.57, 7/10) | 0.43 |
| FRA0.1 replay + anchor | +2.46 (-0.05, 9/10) | -0.85 (-2.09, 8/10) | +4.06 (+0.78, 10/10) | -0.87 (-2.90, 8/10) | 1.45 |
READING: replay is the best trade: it keeps the own gain (60 s equal, 300 s 87 %) and cuts the harm to other speakers ~4x (-3.84 -> -0.77; -2.91
-> -0.87), and at 60 s no enrolled speaker is hurt by more than 0.14. A pull-to-base anchor only trades own gain for less harm (FA1 keeps 53 % of
the gain for the same harm cut); adding it to replay buys nothing. Mechanism (INFERRED, not tested): replay keeps the clean-speech solution in the
loss, so the update moves only along directions that serve the user without breaking the rest; drift alone is not the cause (FR drifts 2.51 and harms
little, FA0.1 drifts 1.44 and harms more). For use: per-user fine-tuning WITH replay (a fixed pool of ~400 training clips on the phone ~ size TBD),
still gated by voice ID. Next: the same on HIS voice (needs his 5-minute enrolment reading).

### 18 Nov 04:45 EET  2-II arm R, shuffled wiring (EXPLORATION; base6s_R.py, sigma 10,5,17,7,1,9,14,0,15,11,13,3,8,6,16,12,2,4; same bytes as S; seeds 0,1)
PER libriA / cleanS / libriB / testother / author2 / him / fit:  R s0 46.53 / 29.43 / 45.07 / 45.50 / 73.64 / 62.62 / 17.87;  s1 46.00 / 29.33 / 44.46 / 46.18 / 73.90 / 55.74 / 17.04
means R 46.27 / 29.38 / 44.77 / 45.84 / 73.77 / 59.18 / 17.46  vs S 46.72 / 29.89 / 45.35 / 46.49 / 72.73 / 59.35 / 18.30
READING: on LibriSpeech the SHUFFLED wiring is as good or better (R - S: -0.46 / -0.51 / -0.58 / -0.65): the LibriSpeech gain of S over plain and
generic is the shape (heads trained on their questions, specialists with their own paths, the extra readout), NOT which phonetic question feeds which
specialist. On his voice (author2) the TRUE wiring is better by 1.04 (S seeds 72.74 / 72.72, R 73.64 / 73.90: consistent across seeds; author2 is not a
decision set and 2 seeds only -> a HINT, OPEN). him (10 clips) no difference. Consequence: no phonetic attribution on LibriSpeech; the his-voice hint
is the thing to test on fresh seeds (and on his enrolment data when it exists).

### 18 Nov 04:57 EET  2-IV run 3: n 6 (EXPLORATION; p2iv_lm.py GN 5,6 x lam .4/.7/1.0 x beta 1/2/3, e60 plain s0 -> logs_base6/p2iv_lm_2026-11-18_n6_plain.json)
Selection PER (150 libriB utts): best n5 42.60 (lam .7, beta 2.0) vs best n6 42.88 -> n 6 does not help; n 5 is now INTERIOR (edge closed).
LM bytes (lower bound): n5 372,508; n6 669,612. Keep n5. The 2aN half crashed at start (FileNotFoundError base5_2aN_fit.npz: the variant feature
copies were deleted at 02:09 with his OK; the LM text path still loads FEAT features) -> not rerun: the plain model already answers the edge question.

### 18 Nov 05:15 EET  2-V arm R at 1 h (shuffled wiring, EXPLORATION; run_2V_R.sh, 10/10 rows: draws 0-4 x seeds 0,1, 60 passes; phones_out/base6/b6vs_R_d*_s*.json)
Mean PER (sd over 10): fit 15.20 (1.05) | cleanS 44.13 (.38) | libriA 57.11 (.56) | libriB 57.99 (.78) | testother 57.76 (.48) | author2 79.18 (.73) | him 76.36 (2.98).
fit 15 vs cleanS 44: at 1 h the 1.38 MB model memorises its training clips. Compare with S / G / P at 1 h when run_2V_S.sh ends (same draws, seeds).

### 18 Nov 05:3x EET  2-II head-meaning knock-out (IDEAS item 16; EXPLORATION, CPU; p2ii_headmean.py -> logs_base6/headmean/{S,R}_s{0,1}_{cleanS,libriA}.json)
Method: replace one head's 12-unit middle by its mean (60 utts), decode the whole set, align, count substitutions crossing each split (split 0 =
deletions + insertions), per 1000 ref phones. Known answer: nothing ablated = run json PER (diff <= 0.02 on all 8). Mean of seeds 0, 1:
S (his wiring), change in the head's OWN split confusions when it is knocked out (cleanS | libriA; % of that split's baseline crossings):
  consonant heads carry their split: 9 nasal|approx +25 % | +4 %; 11 stops|CH JH +34 | +3; 12 P B|T D K G +16 | +16; 13 T D|K G +58 | +25;
  14 HH|fric +10 | +13; 15 sibilant|non-sibilant +89 | +183; 16 S Z|SH ZH +42 | +36; 17 F V|TH DH +54 | +14. Mean change at OTHER splits ~0.1-0.6/1000.
  upper-tree and vowel heads carry little of theirs: 1-8 between -22 % and +5.5 %; silence head 0 +3.9 / +3.3 per 1000 (del+ins).
  PER cost of any single knock-out <= 1.57.
R (shuffled wiring): own-split change ~0 for every head (-17 % .. +3 %) except 13 T D|K G (+30 | +23); silence head +0.3 / +1.2 per 1000.
Reading (exploration, small counts: deep splits have 1-4 crossings per 1000): in his wiring the consonant heads are USED for their own distinction
(named, separable parts: one measured benefit for the parity rule, review notes rule 73); vowel / upper-tree distinctions go through the direct path.
With the wiring shuffled the heads are not used for any split. Not shown: that this helps accuracy (R ~ S on LibriSpeech, 2-II R block).

### 18 Nov 06:29 EET  2-V-S TEST scored (CRITERIA_2V_S.md sha d9636c69, score_2V_S.py sha ca7a358a: both unchanged; 30/30 runs rc 0; logs_base6/score_2V_S.out)
Locked verdict: FAIL (S vs P at 1 h, hard = (libriA + testother)/2: mean gain -0.568, sd .356, 0/5 draws positive; needed >= +1.0).
His parity rule (stated before data: S not worse than G by 0.5 on hard AND cleanS): FALSE (S behind G by 0.379 hard, 0.611 cleanS).
Mean PER over 10 runs (5 draws x 2 seeds, 1 h each, 60 passes):
  arm | fit | cleanS | libriA | libriB | testother | author2 | him
  S   | 15.73 | 44.24 | 57.12 | 58.05 | 57.77 | 79.53 | 77.67
  G   | 12.06 | 43.63 | 56.58 | 57.68 | 57.55 | 78.80 | 76.13
  P   | 18.75 | 43.62 | 56.48 | 57.61 | 57.27 | 79.12 | 78.53
  R   | 15.20 | 44.13 | 57.11 | 57.99 | 57.76 | 79.18 | 76.36   (R from run_2V_R.sh, same draws/seeds; EXPLORATION)
Reading: at 1 h his readout gives no gain and costs ~0.4-0.6 against both the generic readout and the plain model; the shuffled wiring R equals S.
At 5.4 h S was best of S/G/P (2-II). So the structure is not a small-data aid; its 5.4 h advantage (shape, per R) does not appear at 1 h, where every
arm memorises (fit 12-19 vs cleanS ~44). him (10 clips) S ahead of P by 0.85 but behind G by 1.54 (sd 1.2; 10 clips: noise level).

### 18 Nov 07:09 EET  NOISE SCREEN (NOISE_AUG_DESIGN_2026-11-18.md, EXPLORATION; run_noise.sh 4/4 rc 0, smoke showed AUG in force; p2noise_eval.py -> logs_base6/p2noise.jsonl)
base6 60 passes, 5.4 h clean, seeds 0,1 each (plain = Phase 1d e60_r12_causal_s0/s1; plain 5-seed sd: cleanS .36, libriA .20, author2 .31).
New noisy test sets from cleanS + ESC-50 fold 5 (never in training): nz5 steady 5 dB, nzE time-varying events 0-20 dB.
  arm   | fit | cleanS | libriA | libriB | testother | author2 | him | cleanS_nz5 | cleanS_nzE
  plain | 21.22 | 30.66 | 47.47 | 46.27 | 47.03 | 73.22 | 62.30 | 54.20 | 50.75
  N     | 24.48 | 31.23 | 47.14 | 45.84 | 46.72 | 71.97 | 65.09 | 48.20 | 45.60   (steady noise, half the utts, 5-20 dB)
  E     | 25.98 | 32.68 | 47.70 | 46.68 | 47.91 | 70.52 | 66.23 | 48.69 | 46.06   (events)
Expectation written before data: clean cost 0..0.6, noisy gain >= 2. Seen: noisy gain 5-6 points (both arms); clean cost N +0.57, E +2.02;
hard pool N -0.3..-0.4 (slightly better), E +0.2..+0.9. HIS VOICE: author2 (9.76 min, 3,984 phones) N -1.25, E -2.70 (4x / 9x the plain seed sd) =
first change in Phase 2 that moves author2 clearly; him (old 10 clips, 99 words) N +2.8, E +3.9 (worse; small set, but same direction both seeds).
Reading: noise in training is the first lever that reaches his recent memos; the old 10 clips disagree (cause unknown: candidates = those clips are
quiet / differently recorded). N is the safe form (no clean / hard cost beyond 0.6); E helps author2 most at a 2-point clean cost. OPEN: N or E + warp.

### 18 Nov 07:49 EET  NOISE + WARP (EXPLORATION; run_noiseW.sh 4/4 rc 0, smoke showed WARP W + AUG in force; logs_base6/p2noise.jsonl, 12 rows)
Same table as 07:09, + warp (W = b6_2aW s0/s1, the 2-I warp-only arm; WN / WE = warp + steady noise / events). Noisy sets use the CLEAN copy's
warp alphas (upper bound for warp under noise; estimating alpha on noisy audio is untested).
  arm   | fit | cleanS | libriA | libriB | testother | author2 | him | cleanS_nz5 | cleanS_nzE
  plain | 21.22 | 30.66 | 47.47 | 46.27 | 47.03 | 73.22 | 62.30 | 54.20 | 50.75
  W     | 17.88 | 27.57 | 44.11 | 41.51 | 45.09 | 71.19 | 57.70 | 51.34 | 47.63
  WN    | 21.18 | 27.96 | 43.75 | 41.52 | 44.66 | 68.66 | 61.31 | 44.88 | 42.22   (author2 seeds 69.03, 68.30)
  WE    | 22.46 | 28.91 | 43.81 | 41.79 | 45.45 | 68.90 | 62.13 | 45.16 | 42.20
WN vs plain: cleanS -2.70, libriA -3.72, testother -2.37, author2 -4.56, noisy -9.3 / -8.5, him -0.99. WN vs W: cleanS +0.39, hard -0.4, author2 -2.53.
Warp and noise ADD on author2 (W -2.03, N -1.25, WN -4.56). WN = best candidate so far for the phone base; WE no better than WN.
him (old 10 clips) again worse with noise than without (W 57.70 -> WN 61.31): OPEN, cause unknown.
Next (started 07:5x): WN at 180 passes (the frozen base6 recipe length) vs e180_r12_causal s0-s2.

### 18 Nov 08:32 EET  WN at 180 passes (EXPLORATION; run_WN180.sh 2/2 rc 0; plain e180_r12_causal s0-s2 scored on the noisy sets by the same script)
  arm         | fit | cleanS | libriA | libriB | testother | author2 | him | cleanS_nz5 | cleanS_nzE
  plain 180 (3)| 20.19 | 29.56 | 46.75 | 44.76 | 46.18 | 74.11 | 63.06 | 53.08 | 49.61
  WN 180 (2)  | 19.88 | 26.70 | 42.59 | 40.38 | 43.78 | 68.23 | 60.49 | 43.58 | 40.80   (author2 seeds 68.52, 67.95)
WN180 vs plain180: cleanS -2.86, libriA -4.16, libriB -4.38, testother -2.40, author2 -5.88, him -2.57, noisy -9.5 / -8.8: better on EVERY set,
by 8-20x the plain seed sd. Same model bytes as base6 (warp alpha + noise are not stored; the warp needs a per-speaker alpha at use time).
Caveats: alphas come from vtln_*.npz (estimated per speaker from clean audio); noisy sets use the clean copy's alpha; on the phone alpha must be
estimated from his own (noisy) audio: self-scored warp (vtln_self.py) cost ~0.4 on clean (01:4x block), untested under noise.
CANDIDATE new base for the phone ("base7" = base6 + load-time warp + steady ESC noise on half the utterances, 180 passes).

### 18 Nov 08:41 EET  Warp alphas estimated ON the noisy audio (p2noise_alpha.py; known answer: on clean cleanS the path gives vtln_cleanS.npz alphas
exactly, 1.0 agreement). Under noise the estimate moves: agreement with the clean alpha 0.323 (nz5) / 0.417 (nzE), mean |diff| 0.040 / 0.034.
Scored with these alphas (WARP=P WARPPRE=N_; logs_base6/p2noise_Nalpha_score.out; rows in p2noise.jsonl carry warp/warppre):
  WN 180 s0/s1: nz5 44.84 / 44.70 (mean 44.77; clean-alpha 43.58: cost +1.19), nzE 41.86 / 41.65 (41.76; clean-alpha 40.80: +0.96)
  W 60 s0/s1:   nz5 52.27 / 52.48, nzE 48.31 / 48.64
vs plain 180 (no warp, unaffected): nz5 53.08, nzE 49.61 -> WN180 still -8.3 / -7.9 with phone-realistic alphas. Gain survives.

### 18 Nov 08:53 EET  WN 180 seed 2 done -> 3 seeds (mean / sd): fit 20.06/.29 | cleanS 26.84/.21 | libriA 42.82/.35 | libriB 40.40/.32 | testother 43.89/.17 |
author2 68.61/.57 (68.52, 67.95, 69.35) | him 61.09/1.27 | nz5 43.69/.17 | nzE 40.95/.21. vs plain 180 (3 seeds): every set better, as at 2 seeds.
Runner fault (no data effect): the s2 runner's EXIT trap released the lock a second time, after run_2II_WN.sh had taken it (both owners start
"the agent-"); lock missing 08:53:42-08:54:03, re-taken by hand for 2II-WN. Fixed in every non-running runner: rel() matches its own job name and clears it.

### 18 Nov 09:43 EET  2-II on the WN recipe (NOISE_AUG_DESIGN follow-up, reading written before data; EXPLORATION; run_2II_WN.sh 6/6 rc 0, smoke ok)
S / G / P at equal exported bytes (~1.38 MB), WARP=W AUG=noise, 60 passes, seeds 0,1; mean PER (hard = (libriA + testother)/2):
  arm      | fit | cleanS | libriA | libriB | testother | author2 | him | hard
  S (his)  | 18.76 | 26.71 | 42.91 | 40.25 | 44.09 | 69.29 | 59.84 | 43.50 (43.08, 43.92)
  G        | 19.37 | 27.09 | 43.09 | 40.51 | 44.15 | 70.23 | 62.13 | 43.62
  P        | 20.51 | 27.63 | 43.55 | 41.08 | 44.93 | 69.66 | 61.15 | 44.24
  (plain-recipe 2-II for reference: S 29.89 / hard 46.60 / author2 72.73; G 30.14 / 47.14 / 73.91; P 30.67 / 47.06 / 74.05)
Reading as written: PARITY HOLDS (S ahead of G on hard by 0.12 and on cleanS by 0.38); "S leads" NOT met (ahead of P by 0.74, of G only 0.12).
S is best of the three on all 7 sets, as on the plain recipe. The recipe gain (~3-3.5 points) is the same for all three arms: warp + noise and his
readout add. S at 60 passes ~ plain WN at 180 passes (cleanS 26.71 vs 26.84, libriA 42.91 vs 42.82; author2 69.29 vs 68.61).
Next: S / G / P on the WN recipe at 180 passes (the final-length comparison), 6 runs.


## 18 Nov 12:16 2-II on the WN recipe at 180 passes (EXPLORATION, seeds 0,1; logs_base6/run_2II_WN180.sh; phones_out/base6/b6iiWN180_*)
PER mean of 2 seeds (equal exported bytes: S 1,376,393 / G 1,376,909 / P 1,380,389):
| arm | fit | cleanS | libriA | libriB | testother | author2 | him |
| S his readout | 16.95 | 25.65 | 41.95 | 39.17 | 43.12 | 68.50 | 58.36 |
| G generic     | 18.83 | 26.53 | 42.64 | 40.59 | 44.18 | 68.67 | 63.28 |
| P plain wide  | 20.21 | 26.79 | 42.81 | 40.27 | 44.00 | 69.14 | 61.80 |
S-G: cleanS -0.88, libriA -0.69, libriB -1.42, testother -1.06, author2 -0.17, him -4.92; per seed hard (mean libriA/B/testother) -1.45 / -0.66.
S-P: cleanS -1.14, libriA -0.86, libriB -1.10, testother -0.88, author2 -0.64, him -3.44.
S best on all 7 sets, every seed on every LibriSpeech set vs G. At 60 passes S led G by only 0.12 (hard): the lead grows with training (60 -> 180).
P (wider plain) = base WN180 (26.84 / 42.82 / 68.61, 3 seeds): extra width buys nothing; S lower fit too (not less overfitting).
NOT YET: shuffled wiring R at 180 on WN (at 60 passes plain recipe R = S on LibriSpeech). Next: locked TEST on fresh seeds 2-4 (CRITERIA_2II180.md).

## 18 Nov 16:03 TEST CRITERIA_2II180 SCORED (sha e557b5cc / scorer 9ec59f27, both re-checked OK; 9/9 runs rc 0; seeds 2,3,4 fresh; logs_base6/score_2II180.json)
Means of 3 seeds, 180 passes, WN recipe, equal bytes:
| arm | fit | cleanS | libriA | libriB | testother | author2 | him |
| S his readout      | 16.29 | 25.39 | 41.87 | 38.78 | 42.67 | 68.46 | 59.67 |
| G generic          | 18.84 | 27.01 | 43.02 | 40.46 | 44.09 | 69.02 | 60.76 |
| R shuffled wiring  | 15.83 | 25.03 | 41.81 | 38.68 | 42.43 | 68.16 | 60.88 |
Q1 S LEADS G: D(G-S) hard = 1.414 (per seed 2.03 / 0.64 / 1.58) >= 0.5 -> PASS (rule met literally; all three conditions of Q1 are the one threshold).
Q2 HIS PARITY: D(S-G) hard -1.41, cleanS -1.62, both <= +0.5 -> HOLDS.
Q3 WIRING: D(R-S) hard = -0.133 (per seed 0.00 / -0.25 / -0.15), |D| < 0.5 -> SHAPE: the gain is the structure, not the phonetic head-to-specialist assignment.
  author2 R-S per seed +0.35 / -1.48 / +0.25 (mean -0.30): the 60-pass author2 hint for the true wiring (+1.04) did NOT replicate.
What R keeps: heads still trained on their own phonetic questions (aux), same fan-in, same sharing, direct path; only which heads each specialist reads is permuted.
So credited: tree heads with own targets + small specialists + direct path at equal bytes. Not credited: the specific phonetic wiring.
Next (EXPLORATION): S with AUXW=0 (heads get no phonetic targets) seeds 2,3 -> does the head supervision carry the gain, or the shape alone?

## 18 Nov 16:59 S with AUXW=0 (no phonetic targets on the tree heads), WN 180 passes, seeds 2,3 (EXPLORATION; logs_base6/run_S0_aux0.sh)
Means over seeds 2,3 (same seeds of the 2II180 TEST for the others):
| arm | fit | cleanS | libriA | libriB | testother | author2 | him | hard |
| S0 no head targets | 15.97 | 25.17 | 41.72 | 39.13 | 42.84 | 68.40 | 59.67 | 41.23 |
| S  (aux 0.3)       | 16.33 | 25.44 | 42.06 | 38.80 | 42.80 | 68.53 | 58.85 | 41.22 |
| R  shuffled        | 15.82 | 24.98 | 42.02 | 38.76 | 42.52 | 67.97 | 58.86 | 41.10 |
| G  generic         | 18.86 | 27.02 | 43.09 | 40.42 | 44.15 | 69.38 | 60.66 | 42.55 |
S0-S hard per seed -0.23 / +0.25: the head teaching loss carries none of the gain. With R = S (TEST Q3) the whole S-over-G gain at 180 passes
is the readout's SHAPE (many small tree heads + small specialists + direct path), not phonetic supervision or phonetic wiring.
Caveat: the head-meaning knock-out (05:3x) showed S's consonant heads DO carry their own split when trained with aux; meaning is available at
no cost, but it is not what makes S better. Open: which shape element carries it (sparsity / many small parallel paths / tree depth).

### 18 Nov 17:34 - 2-III ON HIS REAL VOICE (him3 recording, EXPLORATION; p2iii_him3.py -> logs_base6/p2iii_him3.jsonl, 15/15 rows)
Base: WN180 plain base6 seeds 0-2 (N known answer: seed 0 author2 68.52 = saved). Adapt pool him3a (first 45 min of the 15 Nov recording, 18.61 min speech,
Phonon pseudo-labels); him3t = held-out last ~11 min (3.09 min, pseudo-labels: approximate). author2 / him = other days, real refs. Mean of 3 seeds (d vs N, paired):
| arm | him3t | author2 | him (old 10) | cleanS | libriA |
| N none | 58.97 | 68.57 | 61.09 | 26.83 | 42.83 |
| F 300 s | 55.31 (-3.66) | 66.65 (-1.92) | 58.69 (-2.41) | 29.18 (+2.35) | 43.73 (+0.91) |
| FR 300 s | 56.13 (-2.84) | 66.68 (-1.89) | 59.12 (-1.97) | 28.06 (+1.22) | 43.19 (+0.37) |
| F all | 51.53 (-7.44) | 64.93 (-3.64) | 53.88 (-7.21) | 30.59 (+3.75) | 44.52 (+1.69) |
| FR all | 52.34 (-6.63) | 64.98 (-3.59) | 54.64 (-6.45) | 28.30 (+1.47) | 43.25 (+0.42) |
Reading: adaptation on one quiet-room session transfers to other days/rooms/car (author2 -3.6, old clips -6.5..-7.2 with all 18.6 min); more minutes help;
replay (FR) keeps ~all the gain and cuts other-speaker harm ~2.5x (cleanS +1.47 vs +3.75). Best for use: FR all, voice-ID gated. Next: same on arm S.

### 18 Nov 18:41 - 2-III ON HIS VOICE for arms S and G (EXPLORATION; p2iii_him3_S.py -> logs_base6/p2iii_him3_S.jsonl, 18/18 rows)
Checkpoints b6iiWN180_{S,G}_s{2,3,4} (CRITERIA_2II180 runs); FR (fine-tune + replay) only; adaptation CTC in a 41-class space (39 + TS + DZ, see header);
N known answers within 1 edit of the saved GPU scores (tolerances 2 edits; S s2 author2 1 edit, G s3 him 1 edit). Fix during the run: copy.deepcopy drops
S's gradient-mask hooks -> masks re-imposed after every step (assert at end). Mean of 3 seeds (d vs own N):
| arm enr | him3t | author2 | him (old 10) | cleanS | libriA |
| S none | 57.86 | 68.45 | 59.67 | 25.38 | 41.87 |
| S FR 300 | 53.50 (-4.37) | 65.69 (-2.76) | 54.43 (-5.24) | 27.07 (+1.69) | 42.80 (+0.93) |
| S FR all | 51.08 (-6.78) | 63.82 (-4.63) | 50.93 (-8.74) | 27.58 (+2.20) | 42.78 (+0.90) |
| G none | 59.02 | 69.03 | 60.66 | 27.01 | 43.01 |
| G FR 300 | 54.23 (-4.78) | 66.43 (-2.61) | 55.08 (-5.57) | 29.19 (+2.19) | 43.69 (+0.69) |
| G FR all | 50.31 (-8.70) | 64.83 (-4.21) | 53.77 (-6.89) | 29.39 (+2.39) | 44.21 (+1.20) |
S-G difference in author2 GAIN, per seed: 300 s +0.09 / -0.11 / -0.43; all -0.63 / -0.38 / -0.25 (S gains more on all 3 seeds with all 18.6 min).
Reading: his structure adapts at least as well as the generic readout (parity shown on gain; slightly more on author2 / him with all data) and ends best:
S FR all author2 63.82 vs G 64.83 vs plain FR all 64.98 (17:34 block). Harm with replay is larger for S/G (+2.2-2.4 cleanS) than for plain (+1.47): OPEN.

### 18 Nov 19:1x - G's shared bank: packing + gradient shares (EXPLORATION; p2ii_packing.py, seeds 2,3 (seed 4 cut by a 110 s timeout); prediction written first)
Per hidden unit of G (H 296), participation ratio over the 55 outputs: median 12.8 / 13.5 (q10 6-7, q90 20); units with PR < 3: 1 / 2 of 296.
-> PACKED: a typical shared unit serves ~13 outputs (S's private units serve 1 by construction). Prediction part 1 CONFIRMED.
Gradient reaching the bank on 100 cleanS utts at the trained point (true labels, 41-class CTC): blank 5.5-5.7 %, SIL < 0.1 %, vowel tokens 42.4 %,
consonant tokens 52.0 %; top IH 7.3-7.7 %, SCHWA 6.1-6.4 %, blank 5.5 %. -> "blank + frequent vowels dominate" WRONG (48 % < half; blank small).
Caveat: shares measured at the end of training on held-out audio, not along training. ALSO (18 Nov 18:5x, seed 2 only): A3 (G + private units) 42.14 vs
G 43.46 vs S 41.43; A1 (no head path) 41.78; A2 (dense fan-in) 41.31.

## 18 Nov 20:0x S<->G blank-odds / symbol swap (p2ii_swap.py, EXPLORATION, CPU, locked 2II180 models s2-s4; reading written in the docstring before running)
Hybrids (a_X, q_Y): a = blank-vs-nonblank log-odds, q = softmax over nonblank; greedy decode as base6s.evaluate. Hard = mean libriA/libriB/testother.
| seed | SS (=S) | GG (=G) | SG (S odds, G symbols) | GS (G odds, S symbols) |
| 2 | 41.43 | 43.45 | 47.77 | 46.63 |
| 3 | 41.02 | 41.64 | 45.37 | 44.65 |
| 4 | 40.88 | 42.46 | 46.18 | 45.19 |
(seed 4 hard from its three set lines; cleanS s2/s3/s4: S 25.40/25.45/25.30, G 28.04/26.00/26.98, SG 33.62/31.04/31.99, GS 31.72/29.60/29.91.)
Known answer: SS / GG reproduced every saved PER within 0.05 except seed 4 author2 G (68.37 vs 68.32 = 2 of 3,984 phones, the known CPU-vs-GPU float
difference seen before); the run stopped there, so seed 4 author2 hybrids are missing. The hard reading does not use author2.
READING (pre-written rule): both hybrids are 2.7-6.3 points WORSE (hard) than both models on every set and seed -> "co-adapted timing/calibration": the two halves
only work together; the swap does not localise S's gain to emission odds or to symbol identity. Weak hint only: GS better than SG everywhere (by
0.72-1.14 on hard; corrected 20:5x, was '0.7-1.9'), i.e. S's symbol choice survives G's timing slightly better than the reverse. Not a test.

## 18 Nov 20:4x Shape ablations A1-A3, seeds 2+3 complete (SHAPE_ABLATIONS_2026-11-18.md; EXPLORATION; reviewer audit PENDING)
hard (libriA/libriB/testother mean) | cleanS | author2:
| seed | S | G | A1 (no head path) | A2 (every head) | A3 (G + private units) |
| 2 | 41.43 / 25.41 / 68.12 | 43.46 / 28.05 / 69.70 | 41.78 / 26.07 / 69.33 | 41.31 / 25.22 / 68.50 | 42.14 / 26.55 / 69.10 |
| 3 | 41.01 / 25.46 / 68.95 | 41.65 / 26.00 / 69.05 | 41.06 / 25.49 / 68.30 | 41.34 / 25.46 / 68.10 | 41.68 / 26.13 / 68.27 |
| mean hard | 41.22 | 42.56 | 41.42 | 41.33 | 41.91 |
Reading: A1 and A2 ~ S on both seeds (head path worth ~0.2 hard on average; fan-in pattern does not matter). A3 recovers 1.32 of G's 2.03 gap on seed 2
but ~0 of 0.64 on seed 3 (41.68 vs 41.65): "private units in G recover most of the gap" was a seed-2 reading and does NOT hold on seed 3; mean
recovery 0.65 of 1.34. The top-only runs (TOPONLY_2026-11-18.md, queued) are the cleaner test of the same question.

18 Nov 20:5x REVIEW AUDIT of the swap (review_audit/swap_audit_web.md): narrower claim = raw-frame halves of independently trained CTC models do not
recombine; it does NOT establish co-adaptation nor localise S's gain (independent models need not spike on the same frames; q sharpness also moves
timing). Controls missing: the same swap between two seeds of S; a frozen-trunk fixed-alignment factorised a/q head test.

18 Nov 21:1x SWAP SEED CONTROL (p2ii_swap_seeds.py, logs_base6/p2ii_swap_seeds.{json,out}; EXPLORATION, CPU, rule written before the run).
Hard mean PER, hybrids (a from seed X, q from seed Y) of the SAME architecture; parents first:
  S2xS3: parents 41.43 / 41.02, hybrids 45.98 / 46.91 (+4.76 / +5.69 vs parent mean)
  S3xS4: parents 41.02 / 40.88, hybrids 44.52 / 45.56 (+3.57 / +4.61)
  G2xG3: parents 43.45 / 41.64, hybrids 50.65 / 50.78 (+8.10 / +8.23)
  G3xG4: parents 41.64 / 42.46, hybrids 46.24 / 46.75 (+4.19 / +4.70)
  (S<->G of the same seed, from p2ii_swap: +3.32 .. +5.33 vs parent mean.)
Verdict by the pre-written rule: same-architecture hybrids are also >= 2 worse -> the S<->G swap says NOTHING about S vs G (independent CTC
alignments do not recombine), as reviewer's audit predicted. Weak hint only (2 pairs each): S seeds recombine somewhat better than G seeds (mean +4.66
vs +6.31). Next test of the a/q question = frozen trunk, fixed alignment, factorised heads (reviewer audit item 4). Audit: batched with the next result.

18 Nov 22:4x REVIEW Q2 FOLLOW-UPS 1-2 (rules review notes, written before the runs; EXPLORATION, WN recipe, 180 passes, seeds 2,3).
HARD (3-set mean) / cleanS / author2:  S s2 41.43 / 25.41 / 68.12, s3 41.01 / 25.46 / 68.95
  P2 (grouping broken, every token on another token's path): s2 41.38 / 25.53 / 69.23, s3 41.01 / 24.86 / 69.08 -> P2 - S hard -0.05 / 0.00, mean -0.03:
     within 0.5 = NO SIGN that the phonetic grouping carries the gain (author2 +1.1 / +0.1 worse: hint only).
  SI (surviving specialist weights at true fan-in): s2 41.54 / 26.00 / 69.03, s3 41.54 / 25.59 / 69.00 -> SI - S hard +0.11 / +0.53, mean +0.32 WORSE:
     the start-up handicap did NOT cost accuracy. Curve libriA at passes 12/30/60/120/180: S2 54.61/47.95/46.62/43.79/41.96, SI2 53.56/47.29/46.76/44.22/41.94;
     S3 53.76/47.32/46.70/43.68/42.73, SI3 53.52/48.15/47.70/43.44/42.27 -> SI ~0.6 faster at pass 12 only; the late lead is not a start-up artefact.
  Follow-up 3 (1 h, 324 passes, S/G/P x draws 0,1) running, ~01:00.

18 Nov 23:4x REVIEW AUDIT of P2 / SI / seed swap (answer review notes/review notes arrived 23:10, no notification).
Verdicts, checked by me where marked: (1) P2 = strong negative for the NECESSITY of the original token-to-path assignment; NOT proof heads are unused (the
top-layer path can bypass the reassigned heads) -> cheapest check = post-training head-path ablation on S and P2 (running: p2ii_specm_ablate.py).
P2 also changes each token's path LENGTH (checked: paths permuted between tokens), so per-output fan-in is not held fixed. (2) SI fan-in maths correct;
SI REDRAWS weights AND bias (checked base6s_SI.py:19), so it is not a pure rescale; cleaner = deterministic rescale of S's own init. Weakens only the
start-scale hypothesis, not training dynamics in general. (3) Seed swap: invalidates the S<->G swap as a mechanism test; does not prove alignment is the
sole cause. (4) 324-pass test: OneCycle total_steps stretches the LR schedule, so it tests update budget + schedule, not step count alone (by construction).
S vs P also differs in trunk width and vocabulary (whole-system comparison). (5) Bytes are an estimated compact size (dense int8 state_dict minus masked
weights and float32 mask buffers), not a measured export: already known (working notes), still to fix before any phone claim. Best next = queued top-only
triplet, key contrast shared-1238 vs private-1238.

18 Nov 23:4x REVIEW Q2 FOLLOW-UP 3: 1 h data, 324 passes (= the 5.4 h update count), plain 2-V-S recipe, seed 0, draws 0/1 (HINT: 2 draws x 1 seed).
HARD / cleanS / author2, 60 passes -> 324 passes:
  S d0 57.21/44.10/80.10 -> 56.47/42.84/77.33   S d1 57.55/44.50/78.97 -> 57.31/43.72/77.23
  G d0 56.63/43.32/79.07 -> 56.60/42.90/78.74   G d1 56.96/43.47/78.11 -> 57.31/43.49/79.32
  P d0 56.46/43.62/78.99 -> 55.48/41.58/77.81   P d1 57.52/44.49/79.27 -> 56.37/42.07/77.54
Pre-written reading (S - P on HARD): 60 passes +0.75/+0.03 (mean +0.39 on these 2 draws), 324 passes +0.99/+0.94 (mean +0.97) -> S stays >= 0.5 behind P:
  the 1 h loss is NOT update count (P gains more from the extra updates: -0.98/-1.15 vs S -0.74/-0.24). S vs G: +0.58/+0.59 at 60 -> -0.13/0.00 at 324
  (S catches G with more updates, G gains ~0). author2: S best at 324 (77.33/77.23 vs P 77.81/77.54, G 78.74/79.32). Caveats (reviewer audit): OneCycle schedule
  stretched with the run (update budget + schedule, not step count alone); S vs P also differs in trunk width and vocabulary (whole system).

18 Nov 23:4x HEAD-PATH ABLATION on trained checkpoints (p2ii_specm_ablate.py, logs_base6/p2ii_specm_ablate.{json,out}; EXPLORATION, CPU, rule before run).
HARD / cleanS: normal -> specm(heads) contribution ZERO -> replaced by its cleanS MEAN:
  S s2 41.43/25.40 -> 54.06/42.11 -> 67.68/59.64     S s3 41.02/25.45 -> 55.85/43.74 -> 64.99/55.23
  P2 s2 41.38/25.53 -> 49.83/36.54 -> 57.15/45.47    P2 s3 41.02/24.86 -> 50.41/36.83 -> 55.51/43.30
Reading (pre-written): S loses >= 1 -> the trained S USES its head path (+13.7 zero / +25.2 mean, mean of seeds). P2 loses 8.9 / 15.0 = ~60 % of S's loss
  (> half) -> P2 did NOT bypass its reassigned heads: the heads are used as generic features whatever token they feed. With A1 (retrained WITHOUT the
  head path ~ S): heads are used when present but substitutable when absent. Odd: mean-ablation hurts more than zero (the cleanS mean is not a neutral
  value for these units; not investigated). Knock-outs measure use, not necessity.

18 Nov 23:5x REVIEW AUDIT of head ablation + 1 h/324 (review notes). Readings hold, tightened:
(1) 324: rules out a simple too-few-updates explanation of S < P at 1 h (2 draws, stretched OneCycle); not data vs reuse vs schedule. S vs P is a whole-system
comparison (trunk width, 39+blank vs 54+blank tokens, aux targets). (2) Ablation: S and P2 both USE their head-path signal; bypass rejected; but "generic
features" is NOT shown (only functional use despite reassignment). Mean worse than zero is expected: a constant mean is not neutral before GELU (removes the
input-varying part, keeps an offset the downstream weights were not trained around). Mean calculation correct. Checked by me 23:5x: P2 checkpoints carry
their own mask (differs from S in 57,600 of 186,624 entries, same 56,064 active), zeros hold in both -> loading P2 via make("S") is valid.
(3) Refined hypothesis: CTC gives every block gradient on every frame (p - gamma), so what private blocks lack at 1 h is POSITIVE evidence and context
diversity, not updates. Cleanest test = shared-1238 vs private-1238 top-only at 1 h AND 5.4 h with matched updates/schedule (2x2); then hierarchical TYING
(W_k = W_parent + Delta_k, tests persistent sharing) rather than hierarchical init; fewer private units = a different question.

### 19 Nov 00:2x - open notes saved from dropped turns (18 Nov evening; not measured)
1. Size on the phone: S's closed (masked) connections are stored as nothing. If a per-user adaptation may open them all, the exported file grows by
   about 150 KB unless the opened set is capped (estimate, not exported). Offered him an extra arm (fine-tuning may open any connection vs closed, same
   seeds, ~30 CPU-min); no answer recorded.
2. Decoder: two consecutive variant tokens of the same phoneme can be counted twice after map-back. Effect NOT measured; the generic readout uses the
   same decoder, so S vs G comparisons share it.
3. Idea under discussion: the design's benefit for him may be a small adapter on one NAMED weak spot (e.g. one confusion pair on his voice), tested
   against a generic adapter of equal size on a new day of his recording (review notes the agent-to-all-discuss-per-user-adaptation.txt).

### 19 Nov 03:4x - 2-III-T running (CRITERIA_2III_T.md + AMENDMENT A1); note written BEFORE any TGT row exists
1. Power check (seed 2, 3 RND draws): sd_draw 0.178, se_boot 0.256, se_D 0.159, P(PASS | D = 0.6) = 0.964 -> 3 draws kept (rule).
2. Learning-rate grid {3e-4, 1e-3, 3e-3}: every S RND draw and R s2/s3 chose 3e-3, the TOP edge (R s4 1e-3). The hashed grid stays; the edge is a stated
   LIMITATION of this test: with ~49k trainable entries (~4 % of S) the best lr may be above 3e-3, for TGT and RND alike. A follow-up exploration with
   {3e-3, 1e-2, 3e-2} on the same arms is the fix, labelled exploration (realization: an edge-picked grid is not a comparison of optima).
3. Seen so far (author2 PER; N0 = unadapted): S s2 N0 68.15, RND 67.95 / 67.72 / 67.60 (gains 0.20 / 0.43 / 0.55; cleanS harm +0.7..+0.85).
   R (secondary, shuffled wiring) N0 -> RT: s2 68.42 -> 67.82, s3 67.49 -> 66.87, s4 68.65 -> 67.07. Gains ~0.4-1.6 vs FR-all full fine-tune ~4.6.

### 19 Nov 05:1x - TOP-ONLY readouts at equal bytes (TOPONLY_2026-11-18.md, EXPLORATION, seeds 2/3, WN recipe 180 passes; run_TOP.sh ended 05:08)
PER (hard = mean libriA/libriB/testother) seed2 | seed3:
  S  hard 41.43 | 41.01  cleanS 25.41 | 25.46  author2 68.12 | 68.95
  G  hard 43.46 | 41.65  cleanS 28.05 | 26.00  author2 69.70 | 69.05
  A1 hard 41.78 | 41.06  cleanS 26.07 | 25.49  author2 69.33 | 68.30
  TS hard 41.26 | 41.42  cleanS 25.37 | 25.39  author2 69.98 | 68.93   (shared top-only, U 1238, 1,376,513 B)
  TP hard 42.84 | 41.95  cleanS 27.21 | 26.18  author2 68.27 | 69.08   (private top-only, U 1513, 1,376,425 B)
  TN hard 41.90 | 40.46  cleanS 26.16 | 24.99  author2 69.45 | 68.07   (private top-only, U 1238, 1,311,264 B)
  G0 hard 43.39 | 42.19  cleanS 28.11 | 26.44  author2 69.83 | 69.43   (G without head targets)
Rule readings (literal, "beats" = mean >= 0.5 AND both seeds):
1. TP - TS hard +1.06 (+1.58, +0.53): TS BEATS TP -> at equal bytes on top-only inputs SHARING wins (cleanS too: +1.31). author2 -0.78 (-1.71, +0.15) no rule.
2. TN - TS hard -0.16 (+0.64, -0.96): tie -> at equal width owning outputs neither helps nor hurts (author2 -0.70 both seeds: TN better there, set not in rule).
3. TS - S hard +0.12 (-0.17, +0.41): within 0.5 -> lower-layer taps through S's projections do NOT matter at this budget; TP trails S by 1.18.
4. G0 - G hard +0.24: tie -> head targets are not a confound of S-G.
Reading (CANDIDATE, reviewer audit pending): a plain shared top-only readout matches S; S's 1.41 lead over G is G's cost, most likely its 5-layer fan-in,
not private capacity. The 18 Nov NN framing "private per-output capacity" (review notes rule 74 note) is NOT supported by this exploration.
Seed disagreement is large (TN: +0.64 vs -0.96) -> 2 seeds rank arms only coarsely.

### 19 Nov 06:3x - reviewer audit of the TOP-only exploration (review notes 2nd attempt, 06:25)
1. reviewer agrees: retire "S wins by private per-output capacity"; working explanation = "G's parameter allocation and/or optimisation is worse than a
   wide top-only bank". NOT shown: S = TS (no equivalence analysis), TN = TS, aux has zero effect, sharing universally better, fan-in as THE cause.
   Its arithmetic checked by me (TP-TN hard +1.22 = +0.94/+1.49; TS-G -1.22 = -2.20/-0.23, i.e. mostly seed 2; TN 65,249 B below TS): correct.
2. Its correction, accepted: TP vs TN does NOT trade away dense cross-terms (both block-diagonal); TP is simply the same private design 22 % wider,
   and it is worse on hard and cleanS on both seeds.
3. Its first diagnostic (train vs eval) done by me from the logs, last-epoch training CTC loss (augmented batches, one number each):
   TS 1.0831 / 1.0867, TN 1.2025 / 1.1082, TP 1.2291 / 1.1708 (s2 / s3). TP fits its training data WORSE than TN and TS -> optimisation /
   init / implementation, not overfitting (reviewer's branch 2). Private banks fit worse than the shared bank at every width (CANDIDATE).
4. author2 TN < TS on both seeds (-0.53/-0.86) = ~28 phone edits over 18 passages of one speaker with auto labels: not evidence of accent robustness.
   reviewer's cheap check (per-passage paired bootstrap, leave-one-passage-out) is open; low priority (author2 does not decide).
5. Next taken: G-top (GT, rules GTOP_2026-11-19.md sha d5ecdf1b, runner logs_base6/run_GT.sh after TOP1h). My disagreement, written in the rules:
   GT is nearly TS by construction (same U 1238); it differs from TS only by a nonlinear blank, so GT-G tests the 5-layer fan-in at equal bytes and
   GT-TS tests the blank choice.

### 19 Nov 07:4x - TEST 2-III-T SCORED (CRITERIA_2III_T.md + amendment A1; all 5 hashes OK; score_2III_T.py -> logs_base6/score_2III_T.out)
Rule 1 quoted: "PASS (S is a better adaptation target): D >= 0.3 AND CI lower > 0 AND cleanS harm of TGT (PER_TGT - PER_N0, mean over seeds) <= +1.5."
Met: D = 0.558 (>= 0.3); 95 % CI [0.187, 1.028] (lower > 0); cleanS harm TGT +1.17 (<= 1.5). -> VERDICT PASS (locked; 3 seeds x 3 RND draws).
Not met / limits: the lr grid edge (most adapters chose the top lr 3e-3; noted before TGT data); A1 amendment touches only the R known-answer
tolerance (R is secondary); author2 labels are a language model -> CMUdict, no human-checked slice (OPEN item 5); one speaker.
Per seed (author2 PER; N0 -> TGT | mean RND | FT): s2 68.15 -> 67.37 | 67.76 | 63.45; s3 68.93 -> 68.12 | 68.65 | 63.81; s4 68.27 -> 66.99 | 67.75 | 64.28.
Mean gains: TGT 0.96, RND 0.40, FT (all weights, reference) 4.60. cleanS harm: TGT +1.17, FT ~+2.1.
Secondary 1 (meaning vs structure): TGT(S) - RT(R) = +0.017, CI [-0.546, 0.588]: targeting the same phonemes' specialists in the SHUFFLED-wiring
model gains as much (RT gains 0.60 / 0.62 / 1.58). So the PASS is about per-output blocks chosen by the speaker's errors, NOT about the phonetic tree.
Secondary 4: in FT the target blocks rank 30.7 / 23.2 / 26.3 of 53 by weight change: full fine-tuning does not concentrate on them.
Plain reading: picking the parts that own his worst phonemes is a better small adapter than random parts of the same size (2.4x the gain), but it
gets about a fifth of full fine-tuning's gain, and the wiring's phonetic meaning adds nothing measurable. reviewer audit queued (the agent-audit-2iii-t).

### 19 Nov 07:5x - reviewer audit of 2-III-T (review notes) + FRQ/RATE control started
1. reviewer: PASS real, mechanism not identified. Agreed corrections: (a) CI lower 0.187 shows D > 0, not D >= 0.3 at 95 % (the rule needed only the
   point estimate); (b) TGT-RT CI [-0.55, 0.59] = "no difference detected", NOT "meaning adds nothing" (a +0.5 tree advantage is still compatible);
   (c) "output ownership" is plausible, not shown (R has no RND control of its own); (d) lr grid edge could flip all three readings for tuned adapters.
2. Rank-1 confound: error COUNT selects frequent phonemes. Control started 07:56 (EXPLORATION, FRQ_2III_2026-11-19.md, p2iii_frq.py; sha file
   logs_base6/FRQ_2III.sha256, rules 4c00aa93 after a time-label fix before data): FRQ (frequency-matched non-target phonemes) and RATE (highest error
   rate, n >= 50, non-target), each at TGT's N; ~2 h CPU. Output logs_base6/p2iii_frq.jsonl.

### 19 Nov 08:5x - TOP1h scored (EXPLORATION, TOPONLY_1H_2026-11-18.md; run_TOP1h.sh ended 08:46:40, all rc 0; GPU shared with W477 06:24-08:46)
PER (hard = mean libriA/libriB/testother | cleanS | author2):
  5.4 h: TS s2 41.26 | 25.37 | 69.98   TS s3 41.42 | 25.39 | 68.93   TN s2 41.90 | 26.16 | 69.45   TN s3 40.46 | 24.99 | 68.07
  1 h (seed 2): TS d0 49.71 | 36.58 | 73.42   TS d1 49.92 | 36.22 | 72.19   TN d0 48.47 | 34.67 | 74.35   TN d1 48.95 | 35.04 | 71.39
D = TN - TS: hard D_5.4h -0.16, D_1h -1.11 (d0 -1.25, d1 -0.97) -> D_1h - D_5.4h = -0.95 (<= -0.5).
Rule quoted: "<= -0.5 -> private relatively BETTER at small data". Met on hard. cleanS same direction (D_1h -1.55, both draws; diff -1.74);
author2 mixed (d0 +0.93, d1 -0.80). Caveat (in the rules): one seed, two draws at 1 h.
Plain reading: the narrowed private bank beats the shared bank at 1 h on both draws (hard and cleanS). This is the OPPOSITE of my 23:xx hypothesis
"private blocks lack positive evidence" and of reviewer's sample-efficiency explanation for TP (sharing should win at small data). CANDIDATE.
reviewer audit queued (the agent-audit-top1h-private-small-data.txt).

### 19 Nov 09:0x - reviewer audit of TOP1h (review notes) + training losses + TSN queued
1. reviewer: rule met; a real counterexample to "private banks need more data"; NOT shown: block structure as the cause (TN is 65,249 B = 4.7 % smaller),
   regularisation as the mechanism, seed robustness. Its point accepted: the 5.4 h reference is itself unstable (+0.64 / -0.96), so against each 5.4 h
   seed the 1 h difference is -1.75 / -0.15. It withdraws its earlier "sample efficiency" explanation for TP. Arithmetic: d0 -1.24 from rounded values.
2. Final 1 h training loss (epoch 971): TS 0.6483 / 0.5399, TN 0.7112 / 0.6346 (d0 / d1): the shared bank fits training data BETTER and generalises
   worse = overfitting signature at 1 h (CANDIDATE). (5.4 h: TS 1.08 vs TN 1.20 / 1.11 with tied eval.)
3. Control queued: TSN = dense TS narrowed to TN's bytes (U 1013, 1,311,297 B), draw 1 primary (pre-committed by reviewer), draw 0 secondary;
   rules TSN_1H_2026-11-19.md (sha 988212a0), runner logs_base6/run_TSN.sh after run_GT.sh.

### 19 Nov 09:5x - G-top (GT) scored (EXPLORATION, GTOP_2026-11-19.md sha d5ecdf1b; run_GT.sh 08:59-09:46, all rc 0, GPU not shared)
PER hard | cleanS | author2 (s2 ; s3):  G 43.46 28.05 69.70 ; 41.65 26.00 69.05   GT 42.22 26.29 70.48 ; 41.72 25.94 68.75
  TS 41.26 25.37 69.98 ; 41.42 25.39 68.93   S 41.43 25.41 68.12 ; 41.01 25.46 68.95
GT - G: hard -0.59 (-1.24, +0.07), cleanS -0.91 (-1.76, -0.06), author2 +0.24.   GT - TS: hard +0.63 (+0.96, +0.30), cleanS +0.73 (+0.92, +0.55), author2 +0.16.
Rule 1 quoted: "GT beats G -> G's deficit is its 5-layer input ... Tie -> fan-in not dominant." "Beats" needs mean >= 0.5 AND same sign: mean -0.59
met, signs DISAGREE (seed 3 +0.07) -> NOT "beats" -> by the rule: fan-in not shown dominant. Rule 2: |GT - TS| < 0.5? No: +0.63, TS better on both
seeds (hard and cleanS) -> the blank choice matters: a separate linear blank (TS) beats routing the blank through the shared nonlinear bank (GT, G).
Final train loss GT 1.1444 / 1.1276 vs TS 1.0831 / 1.0867 (GT fits worse too).
Plain reading: TS - G = -1.22 hard splits into ~-0.59 from dropping the lower-layer inputs (seed 2 only) and ~-0.63 from the blank parameterisation
(both seeds). CANDIDATE: G's deficit is partly how it produces the blank logit, not only its fan-in. Note G's whole S-vs-G gap is mostly seed 2.
reviewer audit queued (the agent-audit-gtop-blank.txt).

### 19 Nov 10:0x - FRQ / RATE control scored (EXPLORATION, FRQ_2III_2026-11-19.md sha 4c00aa93; p2iii_frq.py rc 0; score_frq.py -> logs_base6/score_frq.out)
Targets (all seeds AH IH N T; n in ADAPT 631/447/392/536, error rate 0.41-0.60). FRQ phones: AY D DH K L S Z (n ratio to targets 0.591).
RATE phones (n >= 50, highest error rate, non-target): AA AE AO D HH P SH UW V Z / AA AE AO EH HH P R SH V / AA AE AO D HH P R SH Z.
author2 gains vs unadapted (s2/s3/s4): TGT 0.78 0.80 1.28; FRQ 0.08 -0.05 1.08; RATE 1.15 1.10 1.15; mean RND 0.39 0.28 0.52. All chose lr 3e-3 (grid edge).
Dfreq = TGT - FRQ = +0.586, CI [0.077, 1.118] (0.70, 0.85, 0.20). Drate = RATE - RND = +0.742, CI [0.274, 1.291] (0.76, 0.83, 0.64).
FRQ - RND = -0.028, CI [-0.351, 0.341]. cleanS: FRQ 26.22/26.35/26.24, RATE 26.57/26.45/26.18 (N0 25.40/25.45/25.30: harm ~+0.9 to +1.2, as TGT).
Rules quoted: 1 "Dfreq >= 0.3 AND CI lower > 0 -> error-count targeting beats frequency-matched targeting" -> MET. 3 "If FRQ's n_p are on average
< 0.7x the targets' n_p, the match is poor: reading 2 cannot be claimed" -> ratio 0.591: match POOR (it limits reading 2 only; reading 1 is still
partly open to frequency because FRQ phonemes are 41 % rarer). 4 "Drate >= 0.3 AND CI lower > 0 -> error-RATE targeting helps even among
non-target (less frequent) phonemes" -> MET.
Plain reading: frequency alone does not carry the gain (frequency-near phonemes = random, -0.03); choosing by the speaker's ERRORS does: the
highest-error-RATE phonemes (rarer than the targets) gave the largest gain of all small adapters (1.13 mean, steady over seeds). CANDIDATE; one
speaker, auto labels, lr grid edge. reviewer audit to queue after the G-top one.
(Correction added 19 Nov 09:5x, from `date` = 09:51:47 EET: the two blocks headed "19 Nov 09:5x - GT scored" and "19 Nov 10:0x - FRQ / RATE control
scored" were written at about 09:4x-09:5x; the "10:0x" label is wrong. Numbers unchanged.)

### 19 Nov 09:5x - reviewer audit of G-top recorded (review notes/review notes checked against my numbers)
Checked at source (my GT/TS/S/G per-seed table; base6sy.py ArmGT/ArmT): TS-S hard 0.12 (s2 TS better 0.17, s3 S better 0.41) correct; GT width
1238 vs G H 296 (>4x) correct; GT's bank writes 55 outputs (blank incl.), TS's 54 correct.
Agreed points: 1 Rule 1 not met, and a failed 2-seed rule is NOT equivalence (fan-in is not shown unimportant). 2 Rule 2 met: the TS package beats the
GT package; which difference causes it is NOT isolated (TS changes blank routing, blank parameterisation AND which outputs train the bank).
3 My "0.6 fan-in + 0.6 blank" split is arithmetic between whole configurations, not two causal shares -> WITHDRAWN as a decomposition.
4 G vs GT also changes hidden width >4x, so it is never a pure fan-in test. 5 G seed 2 carries most of G's deficit (G spread 1.81 vs TS 0.16):
possible optimisation sensitivity of G, not established. 6 TS ~ S on hard/clean argues against "54 specialists are needed for S-like hard/clean
PER"; S's author2 lead over TS is seed-2 only (1.86 vs ~0), 18 passages: not shown.
Its next step, taken: blank-bias calibration on the saved checkpoints (beta chosen on libriA only, scored on libriB+testother) + sub/del/ins counts
-> p2ii_blankbias.py (EXPLORATION, CPU, reading written in its header before running), running 09:5x.

### 19 Nov 10:0x - reviewer audit of FRQ/RATE recorded (review notes/review notes) + direct RATE - TGT
Checked at source (logs_base6/p2iii_frq.jsonl, p2iii_tgt.jsonl): mean gains TGT 0.953, FRQ 0.370, RATE 1.133, RND 0.397 correct; RATE-TGT per seed
+0.38/+0.30/-0.13 correct. NEW, computed now (score_frq.py bootstrap, same 2000 passage resamples): RATE - TGT = +0.184, CI [-0.415, 0.808]
-> RATE is NOT shown better than TGT; both are viable error-based policies.
CORRECTION (dated 19 Nov 10:0x): every arm of the run trained N = 48,778 entries (jsonl field N, all seeds/arms equal), not 53,291 (that was the
smoke's target set AH IH R T); my reviewer prompts and the 07:4x/09:5x text said 53,291. Equal budgets across arms hold.
Agreed points: 1 Rule 1 met literally, but "error-count beats frequency-matched" is too strong: FRQ is only the nearest available, 0.59x match,
so frequency is not ruled out; claim = "TGT beats the frequency-nearest control". 2 Neither CI shows the true advantage > 0.3 (the rule needed
point >= 0.3 + CI > 0, a weaker condition). 3 RATE adapts 9-10 specialists vs TGT's 4 at equal entries: coverage is a second explanation for RATE.
4 Regression to the mean does not by itself explain a held-out author2 gain (selection on ADAPT, scoring on author2) - but selection noise limits
stability. Its next steps: a lr grid extension 1e-2 for all four arms (18 CPU runs, select on dev, author2 excluded); b product route: frozen base +
per-user residual delta scaled by alpha (0..1), gated by speaker confidence, alpha chosen on dev for gain vs clean harm (needs saved deltas;
current runs did not save them).

### 19 Nov 10:1x - Blank-bias calibration + error types on G / GT / TS / S (p2ii_blankbias.py, EXPLORATION, CPU, rc 0; logs_base6/p2ii_blankbias.json)
Known answer: beta 0 = saved PER within 0.06 on all 4 sets, all 8 checkpoints (asserted). beta chosen on libriA only; hard2 = mean(libriB, testother).
hard2 beta0 -> at libriA-chosen beta (beta): GT s2 41.73 -> 41.695 (-0.5), s3 41.395 -> 41.30 (-0.25); TS s2 41.00 -> 41.00 (0), s3 41.015 -> 40.825
(-0.25); G s2 43.315 -> 43.11 (-0.75), s3 41.24 -> 41.04 (-0.25); S s2 41.14 -> 41.03 (-0.25), s3 40.455 -> 40.19 (-0.5).
GT - TS gap on hard2: beta0 0.73 / 0.38 (mean 0.555) -> calibrated 0.695 / 0.475 (mean 0.585): does NOT shrink -> reading C (written before running):
a global blank offset is not GT's deficit. G - TS (s2): 2.315 -> 2.11 (shrinks 0.2 of 2.3, < half): blank offset is a small part of G's seed-2 deficit.
Error types, hard2 at beta 0, sub / del / ins (% of ref phones): GT 28.19/11.91/1.64, 27.78/11.92/1.70; TS 27.70/11.54/1.76, 27.56/11.77/1.69;
G 28.54/13.08/1.71, 27.54/11.98/1.72; S 27.27/12.13/1.73, 26.81/12.14/1.51. Non-blank argmax frames 0.200-0.204 in every model.
GT - TS: subs +0.49 / +0.22 > dels +0.37 / +0.15 -> substitution-led (phone discrimination), not deletion-led. G s2 - TS s2: dels +1.54 > subs +0.84 ->
G's bad seed is deletion-heavy, yet a blank offset recovers only 0.2: its missing phones are not a global blank-threshold problem.
S vs TS: S has FEWER subs (-0.43 / -0.75) and MORE dels (+0.59 / +0.37) -> S discriminates phones better and drops more; calibration helps S a bit
more (S s3 -0.27). Every model gains 0.1-0.3 from beta -0.25..-0.5 (a small common blank over-preference; cheap decode-time fix, ~free).
Plain reading: GT's loss to TS is in which phone is chosen, not in whether one is emitted. CANDIDATE (2 seeds); audit batched with the next result.

### 19 Nov 11:5x - LR extension 1e-2 scored (p2iii_lrx.py rc 0, 18 runs; score_lrx.py -> logs_base6/score_lrx_1e-2.out; scorer known answer: with no new
runs it reproduces score_frq exactly: 0.586 / 0.742 / -0.028 / 0.184). EXPLORATION; INTERIM (rule 1 triggered: 1e-2 was dev-best for 12 of 18 arms
-> 3e-2 now runs for ALL arms; final reading after it).
With each arm at its dev-best lr over {3e-4,1e-3,3e-3,1e-2}: gains on author2 (s2/s3/s4): TGT 0.70 0.80 0.08; FRQ -0.58 -0.05 0.58; RATE 1.31 1.13 1.18;
RND 0.54 0.28 0.39. TGT-RND +0.126 CI [-0.366, 0.658] (was +0.558 [0.187, 1.028]); TGT-FRQ +0.544 [0.118, 1.002]; RATE-RND +0.803 [0.288, 1.320];
RATE-TGT +0.678 [-0.030, 1.363]; FRQ-RND -0.418 [-0.647, -0.185].
cleanS (unadapted 25.3-25.45): TGT 28.12 / 26.27 / 27.46 (harm up to +2.7), FRQ 26.68/26.35/26.78, RATE 26.94/27.08/26.71.
Reading so far: the dev split chose 1e-2 where it HURT author2 and clean (TGT s4: dev-best 1e-2, author2 gain 1.28 -> 0.08; s2 clean harm +2.7) -> the dev
split (held-out part of his 57-min recording) does not transfer to author2 at high lr. Under dev-best lr, TGT no longer beats random (the locked
2-III-T PASS at the 3e-3 grid stands as recorded, but it is NOT robust to the lr grid); RATE beats random at both grids (0.74 -> 0.80, CI > 0), and
is the only error-based policy that survives. CANDIDATE until 3e-2 is in.

### 19 Nov 13:1x - TSN (dense top at TN's bytes, 1 h) scored by TSN_1H_2026-11-19.md (sha 988212a0 unchanged; run_TSN.sh rc 0, lock released 13:11)
Inputs check: TS d1 49.92 / 36.22 and TN d1 48.95 / 35.04 = the rows written in the rules before data. hard = mean(libriA, libriB, testother).
TSN d1 (PRIMARY): hard 49.90, cleanS 36.22, author2 71.79; d0: 49.50 / 36.05 / 73.14.
f = (TS - TSN) / (TS - TN): d1 hard 0.02, cleanS 0.00; d0 hard 0.17, cleanS 0.28 (author2: d1 0.5, d0 -0.3; TN is worse than TS on author2 at d0).
Rule (quoted): "f <= 0.3 -> block structure, not size" -> MET on the primary (and on every hard/clean reading). Removing TN's 65 KB from the dense top
reproduces almost none of TN's 1 h advantage: at 1 h of data the private-block STRUCTURE (each output its own units) helps, not fewer parameters.
Final train loss (epoch 971) TSN 0.6711 / 0.6129 (d0 / d1), between TS (0.6483 / 0.5399) and TN (0.7112 / 0.6346): fewer dense numbers fit a bit
less but do not generalise better. CANDIDATE (1 model seed, 2 data draws); reviewer audit queued with the blank-bias result.

### 19 Nov 13:2x - reviewer audit of TSN + blank-bias recorded (review notes) + oracle-beta envelope
Accepted narrowings: 1 "structure, not size" -> "TN's advantage is not reproduced by parameter reduction alone"; WHICH structural property (private
units / less gradient sharing / implicit regularisation) is not identified. 2 "not a blank threshold" holds only for ONE CONSTANT blank offset.
3 "data-dependent" (1 h vs 5.4 h) also changes exposure / optimisation; not isolated. TN's author2 sign flips between draws (+0.80 / -0.93).
Its rank-1 step done now from the saved grid (p2ii_blankbias.json, beta -2..+1, step 0.25 near 0): ORACLE beta chosen on hard2 itself (diagnostic upper
bound, not a benchmark): GT s2 41.675 (beta -0.25), s3 41.30 (-0.25) vs TS at its libriA-chosen beta 41.00 / 40.825 -> gap 0.675 / 0.475 remains.
Optima interior (no grid edge). Rule (reviewer's, stated before computing): if GT's test-set oracle beta cannot close the gap to TS's validated point, no
single constant blank offset explains it -> MET: excluded for these checkpoints. (TS oracle 40.81 / 40.825; G s2 oracle 43.035.)
Next (its rank 2): identical fresh linear CTC probes on frozen GT and TS trunks (feature vs readout); rank 3 = the seed-3 replication already queued.

### 19 Nov 13:5x - author2 WORD check (his ask 13:3x; author2_wordcheck.py rc 0; logs_base6/author2_wordcheck.{json,txt})
Two local ASRs on the same cut author2 audio vs the language-model truth words: parakeet-tdt-0.6b-v2 WER 13.14 %, Phonon-2.5 11.55 % (1,195 truth words).
SUSPECT (both ASRs agree with each other, not with the language model) = 51 words, DISPUTED 24. By hand-reading the 51: ~22 are fillers UH/HUH and ~14 are
repeated / false-start words (UH AFTER UH AFTER, I I, ON THE ON ON) that both ASRs omit by design (they are trained to drop disfluencies), plus
'THIS MANY REQUESTS' at a cut (the residual known from author2 set rule 2d) -> the language model is most likely RIGHT on these (verbatim transcript).
~15 are word swaps: SHELL->CELL x2, (product name)->CLOUD/CODE x2, TOOL->TWO, PAUSE->POSE, END->TEND, DRIVING->DRAGGING, LIKELY->LIKE, ON->AND, IT->THIS,
DOES->THAT, OF->ACTUALLY, A->THE x2, FINISHED->FINISH, THAT'S->THAT, REQUEST->REQUESTS, NEE->NEED. Names and his SH (SHELL heard as CELL by both:
an accent realisation, not a transcript error) explain several; true language-model errors are at most ~15 of 1,195 words (~1.3 %, upper bound,
UNVERIFIED until his ear check). Reading: the WORD transcript is not the problem; the open question is the phoneme realisation (phone_check.py).

### 19 Nov 14:06 - LR extension 3e-2 scored (p2iii_lrx.py run 3e-2 rc 0, 36 rows; score_lrx.py -> logs_base6/score_lrx_3e-2.out). EXPLORATION; INTERIM
again: rule 1 triggered (3e-2 dev-best for 3 of 18 arms, all seed 4: TGT0, RATE0, RND2) -> 1e-1 now runs for ALL arms; final reading after it.
With each arm at its dev-best lr over {3e-4 .. 3e-2} (author2 gain s2/s3/s4): TGT 0.70 0.80 -0.10; FRQ -0.58 -0.05 0.58; RATE 1.31 1.13 0.88;
RND mean 0.54 0.28 0.58. Contrasts (bootstrap as score_frq): TGT-RND 0.006 [-0.443,0.485]; TGT-FRQ 0.485 [0.103,0.905]; RATE-RND 0.641
[0.168,1.142]; RATE-TGT 0.636 [0.052,1.281]; FRQ-RND -0.480 [-0.729,-0.229]. cleanS (harm side) TGT 28.12 26.27 28.72, FRQ 26.68 26.35 26.78,
RATE 26.94 27.08 27.27. Reading so far (not final): TGT's 2-III-T advantage over random has gone (0.558 at the 3e-3 grid -> 0.006); RATE keeps a
positive interval over RND and now over TGT; dev-picked high lr again lowers the author2 gain on seed 4 (TGT -0.10).

### 19 Nov 15:14 - BIG PHONE RECOGNISERS vs OUR LABELS (phone_check.py rc 0; logs_base6/phone_check.json). DESCRIPTIVE (his ask: is his speech mislabelled?)
Models: PhoneticXeus (PX, IPA, realised sounds) and wav2vec2-lv-60-espeak-cv-ft (W2), IPA -> 39 map. PER vs OUR labels:
PX author2 27.59 / libriB 19.13 / testother 18.94; W2 author2 53.11 / libriB 11.32 / testother 10.88 (150 utts each; W2 unmapped 'ᵻ' 31/86/64).
Known answer: Libri PER plausible for both (load OK). DEFECT found in my map: PX emits diphthongs as two '/' tokens (a/ɪ), so ins>AA, AY>IH,
ins>UH, AW>UH top the PX list on EVERY set. Rescore of the stored author2 hyps: unmerged reproduces 27.59; merged (AA IH->AY, AA UH->AW, EY IH->EY,
OW UH->OW, AO IH->OY) author2 = 19.20 (hyp/ref 1.049 -> 0.987). Libri hyps were stored for 20 utts only -> phone_check2.py reruns PX on all
sets (merged + unmerged, raw tokens) and W2 on author2 in 20 s windows (W2 deleted ~18 % of phones on author2 only, hyp/ref .821; length suspect).
PX shows B>P, D>T, G>K on Libri too (PX convention, not his). Reading waits for phone_check2.

### 19 Nov 15:32 - FROZEN-TRUNK PROBES GT vs TS, EP 30 (p2ii_probe_gt_ts.py EP=30 rc 0; logs_base6/p2ii_probe_gt_ts_ep30.json). EXPLORATION
Known answers hold (original tops libriA GT 43.19/42.38, TS 41.79/42.24). hard2 per probe seed: GT_s2 49.79 49.70; GT_s3 50.205 48.715;
TS_s2 48.18 48.34; TS_s3 48.375 48.325. gap_probe (better probe seed by libriA, mean of 2 model seeds) = 0.93 (EP 6: 1.43). gap_orig 0.555.
Locked reading: 0.93 >= 0.39 -> "GT's trunk is worse (features)". BUT NOT CONVERGED again: best epoch is 27-29 of 30 for all 8 probes and the
loss still falls (~0.004/epoch); the gap SHRANK from 1.43 to 0.93 between 6 and 30 epochs; probe seeds differ by 1.49 on GT_s3 (> 0.3: noise);
linear probes sit ~6-8 points above the original tops (48-50 vs 41-43), so they read far less than the tops do. Verdict: CANDIDATE only
("GT's trunk is less LINEARLY readable"), not the features/readout split. Stop rule (researcher 8.5): two unconverged runs -> not repaired a
third time; a decision needing it would use a converged probe (lr bracket 3e-3 / 1e-2, 100 epochs, ~5 h CPU). OPEN.

### 19 Nov 16:02 - phone_check2 (rc 0; logs_base6/phone_check2.json). DESCRIPTIVE. Known answer: unmerged PX reproduces run 1 exactly (27.59/19.13/18.94).
PX MERGED (diphthong pieces joined): author2 19.20 | libriB 10.65 | testother 10.73 -> his labels vs realised sounds 8.5 points worse than on
LibriSpeech read speech (accent + spontaneous/driving speech + label errors together; this check cannot split them: no native spontaneous set).
W2 in 20 s windows on author2: 42.19 (whole passages 53.11): input length explains ~11 points of W2's failure, not all; W2 not used further.
Largest author2-vs-Libri excess (PX merged; % of that ref phone; author2 count / ref n): AH deleted 13.1 vs 0.7 (50/381); IY->IH 7.4 vs 1.8 (10/136);
CH->S 22.2 vs 0.8 (4/18), CH->T 11.1 vs 2.7 (2/18), CH deleted 2/18; JH->Z 2/22; SH->S 6.5 vs 1.6 (2/31); vowel moves AA->AO 5/54, AO->AH 5/64,
AO->OW 4/64, ER->R 5/62; final-ish deletions T 21/353, S 13/235, D 12/171, DH 6/111; D->T 24.6 vs 19.1 (PX convention on both). Insertions on author2:
T 23, D 21, AH 14, DH 10. Counts for the affricates and SH are 2-4 events: CANDIDATES only (his 5 Nov words: "in my own speach its in part the ts
sound" fit CH->S/T). AH deletion and IY->IH have enough events to be real. Next: the same on him3 (57 min) on passages where Phonon and parakeet agree.
reviewer audio ask (review notes) NOT answered: AI Studio account out of quota (503 / hang, proxy log 1651288 >= 800000); proxy worker
died after a 420 s non-stream cut, restarted 15:35 (review note).

## 19 Nov 16:31 EET: SEED-3 replication of TS / TN / TSN at 1 h (EXPLORATION; rules SEED3_1H_2026-11-19.md, written before data)
Runner logs_base6/run_SEED3.sh (14:21-16:30, rc 0 for all 3, config-in-force greps passed); results phones_out/base6/b6t1h_{TS,TN,TSN}_d1_s3.json.
DRAW 1, WARP W, AUG noise, 972 passes. hard = mean(libriA, libriB, testother) PER. Bytes TS 1,376,513 / TN 1,311,264 / TSN 1,311,297.
| arm | seed 2 hard | seed 3 hard | seed 3 cleanS | seed 3 author2 |
| TS  | 49.917 | 50.327 | 36.54 | 72.87 |
| TN  | 48.950 | 48.430 | 34.57 | 70.76 |
| TSN | 49.900 | 49.233 | 35.54 | 71.71 |
Reading 1 (TS - TN hard >= 0.5 at seed 3): 1.897 -> TN's 1 h advantage over TS REPLICATES (2-seed mean 1.432; cleanS 1.97, author2 2.11 at seed 3).
Reading 2 (f = (TS-TSN)/(TS-TN) <= 0.3 at seed 3): f = 0.577 -> structure-not-size does NOT replicate; at seed 3 the dense model cut to TN's
bytes recovers ~58 % of TN's advantage (TS-TSN 1.094). 2-seed mean f 0.388 (seed 2 f 0.02). Reading: part of TN's 1 h advantage is the smaller
size, part is the private structure; the split is seed-dependent with 2 seeds. Seed-2 numbers not re-read (taken from the seed-2 files as written).
SEED3 reviewer audit (answers/the agent-audit-seed3-ts-tn-tsn.answer.md, 16:3x): rule 1 PASS, rule 2 FAIL; f too unstable to be the statistic (report
paired differences). Its matched-size contrast TSN - TN = 0.950 (s2) / 0.803 (s3), checked by me from the table: TN beats the byte-matched dense
top at both seeds -> size alone does not explain TN's lead (EXPLORATION, 2 seeds). Its "enlarged TN" control = existing arm TP (U 1513,
1,376,425 B); TP at 1 h never run -> TP1H_2026-11-19.md (sha logs_base6/TP1H.sha256), run_TP1H.sh seeds 3, 2 queued 16:3x behind the agent-regate.

## 19 Nov 16:41 EET: him3 phone check (his 57-min recording; him3_phone_check.py A/B/C; DESCRIPTIVE, rule in its docstring, fixed before stage B output)
Words: parakeet-0.6b vs Phonon on the 294 him3 segments: EXACT agree 159, ignoring fillers 170. PhoneticXeus merged vs CMUdict labels (PER):
Libri hard 10.65 / 10.73, author2 19.20, him3 all 19.99, him3 EXACT 16.24 (4,458 phones), NOFILL 16.43. -> on segments whose words two recognisers
agree on, the gap to native read speech is ~5.5 points (accent + spontaneous speech together); the other ~3.7 of the all-segment gap goes with
segments whose words the recognisers dispute (Phonon word errors in the pseudo-labels, or harder audio).
Listed patterns (rate >= 2x Libri, >= 5 events, ref >= 30; full list logs_base6/him3_phone_check.txt), him3 all / Libri / author2:
substitutions: CH>S 17.5 % (7/40) / 0.9 / 22.2; CH>T 15.0 % / 2.6 / 11.1; AO>OW 8.3 / 0.0 / 6.2; EY>IY 7.8 / 0.0 / 0.0; IY>IH 5.9 / 1.7 / 7.4;
Z>S 6.3 / 1.5 / 3.3; S>Z 4.2 / 1.5 / 3.4; AY>AA 4.5 / 0.5 / 2.1; AE>AA 4.0 / 0.0 / 0.9; AE>EH 4.5 / 1.0 / 0.9; IH>IY 3.6 / 0.4 / 1.4; OW>AA 5.0 / 0.5;
AA>OW 6.1 / 0.5; AH>AA 3.1 / 0.3 / 0.3. deletions: AH 8.7 / 0.7 / 13.1; AY 6.8 / 0.0; IH 5.6 / 0.5; DH 5.3 / 0.5; T 4.2 / 1.1; S 4.2 / 0.3; R 7.8 / 1.3;
HH 11.2 / 1.5; Y 13.1 / 0.0. CH>S/T now 13 events on him3 + author2 agrees: upgraded from 2-4-event candidate to a seen pattern (still one recogniser).
CONFOUND: Libri is native READ speech; deletions and some vowel reductions are normal in native SPONTANEOUS speech. Accent vs spontaneous is not
separated until the same check runs on native conversational speech (Buckeye) -> next.

### 19 Nov 16:47 - LR extension FINAL (p2iii_lrx.py run 1e-1 rc 0; score_lrx.py -> logs_base6/score_lrx_1e-1.out). EXPLORATION.
Rule 1 closed: 1e-1 was dev-best for NO arm (new_top_chosen false) -> grid {3e-4 .. 1e-1} brackets every arm; chosen lrs 3e-4 .. 3e-2.
Final (author2 gain differences, 2000 passage resamples, per seed 2/3/4): TGT-RND +0.006 [-0.443, 0.485] (0.17, 0.53, -0.68);
RATE-RND +0.641 [0.168, 1.142] (0.77, 0.85, 0.30); RATE-TGT +0.636 [0.052, 1.281]; TGT-FRQ +0.485 [0.103, 0.905]; FRQ-RND -0.48 [-0.729, -0.229].
Gains on author2: RATE 1.31 / 1.13 / 0.88; TGT 0.70 / 0.80 / -0.10; RND 0.54 / 0.28 / 0.58; FRQ -0.58 / -0.05 / 0.58. cleanS (harm side) RATE 26.94-27.27.
Reading: the locked 2-III-T PASS (TGT-RND 0.558 at the old grid) does NOT survive the bracketed lr grid (tie); error-RATE-chosen blocks beat
random blocks and beat his-phoneme-named blocks at every seed. Remaining limit: lr is chosen on a dev split of him3 (pseudo-labels).
LR-final reviewer audit (answers/the agent-audit-lr-extension-final.answer.md, 16:5x): RATE best "within this experiment" (CIs resample passages, not
seeds); old PASS valid for the edge-limited grid only; next control = RATE blocks chosen from ANOTHER speaker's errors (generic hard phonemes vs
his own). Clean harm unassessed in my ask (base clean PER not given).
### 19 Nov 16:52 - GEN control started (p2iii_gen.py, rules in its docstring, sha logs_base6/p2iii_gen_rules.sha256; EXPLORATION, CPU ~2 h)
GEN = RATE ordered by the same base model's error rate on NATIVE libriA instead of his him3. Dry run (known answer: RATE blocks == saved): GEN shares
4 / 5 / 4 of 11 blocks with RATE (seeds 2/3/4). Order of hardest phonemes: native AO ER AA EH UW L R D; his SH AE AO AA HH P (UW Z / V EH).
-> SH, AE, HH, P are hard for HIM but not for native speakers (descriptive). Output logs_base6/p2iii_gen.jsonl -> score_gen.py.

### 19 Nov 17:04 - Buckeye native-conversation phone check (buckeye_phone_check.py rc 0; logs_base6/buckeye_phone_check.{txt,json}). EXPLORATION.
PX merged PER vs dictionary labels: Libri hard 10.65 / 10.73; Buckeye (native US conversation, 400 segs, 16 spk, 11,185 phones) 20.95;
him3 agreed-word segs 16.24, him3 all 19.99; author2 19.2. -> His overall gap over read LibriSpeech is the SPONTANEOUS-speech gap: native
conversational speech is as far from dictionary labels as his (his agreed segments are closer). Deletions (AH, D, T, DH, L, IH, N, final
consonants) are SPONTANEOUS-LIKE (Buckeye >= 2x Libri), not accent.
ACCENT-LIKE (him3 >= 2x Buckeye, rule fixed before data): vowel tense/lax both ways IY>IH 5.9 % (buckeye 0.5), IH>IY 3.6 % (0.7), EY>IY 7.8 % (1.1),
EY>EH 3.6 % (0.0), EH>IH 3.9 % (1.0); back/low vowels AO>OW 8.3 % (0.5), AA>OW 6.1 % (2.1), AE>AA 4.0 % (1.5), AH>AA 3.1 % (1.1);
voicing S>Z 4.2 % (1.9), Z>S 6.3 % (1.4); CH>S 17.5 % (7/40; buckeye 0.0, libri 0.9) - his own "ts" observation; P>del 6.9 % (1.5).
CH>T 15.0 % vs buckeye 5.8 % (2.6x but under min count in exact). Counts are small (5-27 events per row): CANDIDATES, not measured rates.
Implication: labels of his voice are not "wrong" beyond spontaneous speech in general; a dictionary-label test of ANY conversational speech
carries ~10 points of label/pronunciation mismatch (PX on Buckeye). Our author2 PER (68-74) is not mainly label error.

### 19 Nov 17:40 - reviewer accent ask (review notes) + raw-token check. UNVERIFIED (LLM) / EXPLORATION.
Proxy limit raised to 1200 s at 17:32 (review note; skill web search rule 11). reviewer 17:33: 503 again after 113 s (Google-side, 0 chars).
reviewer 17:34-17:39: "Model finished thinking but generated no output"; its THINKING (reviewer notes, 31 kB) lists, with
guessed token counts (LLM numbers, not measured): IH->[i] (~160), AE->[a] (100-130), AA->[ɔ] (80-110), unaspirated p/t/k (80-120), OW->[o] (80-100),
light L (60-80), UH->[u] (40-50), SH->[s] (30-40), h->[ç]/[x] (25-35), non-flapped intervocalic t (15-25), unreduced unstressed vowels (50-60), R tap/drop.
Agrees with the MEASURED accent-like rows on: IH/IY tense-lax, AE->AA, back vowels, SH/CH->S, AH->full vowels (AH>AA/AE/OW).
Raw PX tokens him3 (294 segs) vs Buckeye (400): aspirated p/t/k 257/1582 = 0.162 vs 369/1854 = 0.199; schwa-type share of vowels 0.164 vs 0.179;
ɹ 26.9 vs 37.1 per 1k tokens. Direction fits "unaspirated stops / unreduced vowels / R dropped" but small and not normalised per word (content
differs). PX never emits ɾ, ɫ, ç, x on either set -> flap, dark L, ç/x NOT testable with PX.

### 19 Nov 18:09 - TP at 1 h scored (rules TP1H_2026-11-19.md, sha OK; run_TP1H.sh rc 0 both; phones_out/base6/b6t1h_TP_d1_s{3,2}.json). EXPLORATION.
hard = mean libriA/libriB/testother. s3 | s2:  TS 50.33 | 49.92   TN 48.43 | 48.95   TSN 49.23 | 49.90   TP 48.99 | 48.95.
fit (training PER): TS 0.32 | 0.17, TN 0.45 | 0.79, TSN 0.20 | 0.42, TP 0.65 | 0.66 (TP fits worst, generalises like TN).
Rule 1: TS - TP hard +1.34 (s3), +0.96 (s2): >= 0.5 at BOTH -> private beats shared at EQUAL (larger) bytes too, at 1 h (2 seeds).
  cleanS +1.96 / +0.85, author2 +1.06 / +1.23 (same direction everywhere).
Rule 2: TP - TN hard +0.56 / +0.00 (mean +0.28): growing the private design to TS bytes does not help; size is not what TN's lead comes from.
Rule 3 (paired, hard): TS-TN +1.90 / +0.97; TS-TSN +1.09 / +0.02; TSN-TN +0.80 / +0.95.
Reading: at 1 h the shared-top model loses to the private design at both TN's size and TS's size; the shared dense top cut to TN size is between.
2 seeds, 1 draw, 1 h only: CANDIDATE until reviewer audit.

### 19 Nov 18:17 - reviewer TP audit read + checkpoint check (reviewer answer review notes).
reviewer: rule 1 met; "private beats size-matched shared at both budgets, both seeds" = well-supported DESCRIPTIVE result under this protocol, not a
general law; mechanism (per-output capacity vs gradient/trunk effects) not isolated; "shared overfits" NOT shown by the fit column; ranks: checkpoint
audit, then independent data draw, then frozen-trunk readouts. Its two rounding notes (0.96 vs 0.97, 1.09 vs 1.10) checked: unrounded values, fine.
Checkpoint check (training-time libriA eval, logged every 6 passes, in each json 'log'): no arm peaks early; all still improving late (best libriA at
passes 804-924 of 972). Best-checkpoint libriA TS-TP: s3 50.29-49.59 = +0.70, s2 50.09-49.15 = +0.94; TSN-TN: s3 49.83-49.22 = +0.61, s2 49.72-49.86
= -0.14. -> TS's deficit is not overtraining past a peak; at the smaller budget seed 2 the gap is within checkpoint noise (best-on-libriA is
selection on the reported set, same for every arm).
Draw-0 replication: TS / TN / TSN at draw 0 seed 2 already exist (hard 49.71 / 48.47 / 49.50); TP d0 s2 running (rules TPD0_2026-11-19.md, sha logs_base6/TPD0.sha256).

### 19 Nov 18:57 - TP on draw 0 scored (rules TPD0_2026-11-19.md sha OK; run_TPD0.sh rc 0; b6t1h_TP_d0_s2.json). EXPLORATION.
Draw 0 seed 2 hard: TS 49.71, TN 48.47, TSN 49.50, TP 48.67. Rule: TS - TP = +1.04 >= 0.5 -> REPLICATED on a second data draw.
TP - TN +0.21. TSN - TN +1.03. cleanS: TS-TP +1.39, TSN-TN +1.38. author2 (no verdict): TS-TP +1.13, but TN worst on author2 here (TS-TN -0.93, TSN-TN -1.21).
Over 3 (draw, seed) pairs the private design beats the size-matched shared top on hard at both sizes in 6 of 6 contrasts (+0.80 .. +1.34);
on his voice the small private top (TN) loses on draw 0. fit: TN 1.22 / TP 1.10 vs TS 0.47 (same pattern as draw 1).

### 19 Nov 18:59 - GEN control scored (rules p2iii_gen.py header, sha OK; p2iii_gen.jsonl 18 rows rc 0; score_gen.py -> logs_base6/score_gen.out). EXPLORATION.
GEN = RATE with block choice ordered by the unadapted model's error rate on NATIVE speakers (libriA) instead of his him3. Overlap with RATE 4-5 of 11
blocks per seed; no seed identical. GEN lr dev-best 1e-2 / 3e-3 / 1e-2 (interior). author2 gain over unadapted, seeds 2/3/4: RATE 1.31 / 1.13 / 0.88,
GEN 1.28 / 0.70 / 1.13, RND 0.54 / 0.28 / 0.58, TGT 0.70 / 0.80 / -0.10.
RATE-GEN +0.067 [-0.483, 0.588] (per seed +0.03 / +0.43 / -0.25); GEN-RND +0.575 [0.224, 0.971] (+0.74 / +0.43 / +0.55).
-> RULE 3: the RATE gain is explained by generically hard phonemes; choosing blocks by HIS errors is not shown to add anything (3 seeds, 1 speaker,
CIs resample passages, not seeds). cleanS (harm) similar: RATE 26.94 / 27.08 / 27.27, GEN 27.20 / 26.67 / 26.77.
Practical: the per-user layer can pick its blocks from native error rates (no labels of his needed for the CHOICE; adaptation still uses his audio).

### 19 Nov 23:5x - CORRECTION (dated addition) to the 18:57 block
The line "6 of 6 contrasts (+0.80 .. +1.34)" does not match its own values: TS-TP +1.04/+0.97/+1.34 and TS-TN +1.24/+0.97/+1.90 span +0.97..+1.90.
The 6-of-6 count stands. Source: values in the 08:5x, 18:09, 18:57 blocks. Copied to MASTER_PLAN section 7.4-7.5.
