# MASTER PLAN: small speech model and what makes a small network good (owner the agent)

Version 2, 19 Nov 2026 23:36 (version 1: 19 Nov 23:3x). Rewritten in place; review posts only point here.

Rule (the author 19 Nov; plans/README.md): this file holds the whole plan and what may be broken now. A MAJOR change (goal, method, budget,
anything another agent depends on) needs a second verifier and the agent before it runs (the agent checks
that files and reports agree; not a requirement). Until then a correction below is
PROPOSED and the author's v5.4 (Part B, unchanged) stands as written. Claims not checked at source say so.

## 1. Goals in the author's words
1. 12 Nov: "the goal is just trying to find the best methods. Not to follow anything spesific. ... We are just trying to find improvments over
   industry standard and check which ones of the industry standard still hold up while learning lessons on how to make efficient ML on small
   parameters making the best smalkl model possible"
2. 18 Nov: "Why do you keep making it about acoustics the real question is about NNs not acoustics" -> speech is where we measure; the
   question is about small networks in general (v5.4 adds image tests for that reason).
3. 14 Nov size: "lets stick to the higher number as the limit" -> 2 MB, the higher of disk and memory while running, on the phone.
4. 19 Nov: "optimizing for my voice is an asbolute afterthought its almost entirely misses the point of what is imporatnt."
5. 19 Nov: v5.4 "What makes a small network good?" (Part B) is the research plan: questions Q1-Q7, paid for tranche by tranche.
6. Phone assistant (separate line, paused): a releasable Android assistant that controls the phone by voice or text; everything on the phone
   except the language-model API.

## 2. Current state (measured numbers only; error = phone error rate in %, lower is better)
Test sets: "hard" = LibriSpeech's harder recordings (other pool), "clean" = unseen speakers from the clean pool. Training so far is clean only.
1. Best small model so far: 1.4 MB exported (8-bit weights), trained on 5.4 h with speaker warp + steady background noise, 180 passes,
   3 seeds: hard 42.82, clean 26.84 (plain recipe, same model: 46.75 / 29.56). The older 2.3 MB model on 100 h, plain recipe: hard 42-43,
   clean 25-27. So the two tricks on 5.4 h match 18x more data, on these sets.
2. Not yet measured: the best recipe on 100 h; any like-for-like outside (industry) model at the same size on the same sets; real phone
   speed and memory of the 8-bit file.
3. Broken or open: see section 6 ("What may be broken"); nothing is running now.
4. Phone app: installed and working features (feedback, zoom, shell with confirmation, takeover, noisy wake 18 of 39 accepted, idle CPU
   1.72 %); two owner-only steps untested (accessibility switch + finger test; wake phrase while driving).

## 3. Every method used so far and what it gave (details and seeds in the project results log)
1. Fair black-box base (streaming, causal 3 s normalisation, ~1 s look-ahead): the reference all ideas are compared with.
2. Speaker warp (stretch each voice to a common length scale): test PASSED twice (5.4 h: -2.84 hard; 1 h: -2.41, 5 of 5 draws).
   Uses the true speaker identity; the version that uses only the speaker's earlier speech costs ~0.9 of the gain.
3. Noise-floor removal: added nothing (+0.67, worse). Speaking-rate removal: two estimators failed, cut.
4. Steady background noise in training: helps every set, more with the warp (row 2.1).
5. Phoneme-tree readout (his structure) vs a generic readout of equal size: at 5.4 h a tie or slightly better (+0.48, missed its 0.5
   bar; best on all 7 sets with the noise recipe); the same tree with shuffled wiring was as good on LibriSpeech, so the gain there is the
   shape, not the phonetic assignment. At 1 h it lost to the private-top arm.
6. Private per-output capacity vs a shared layer at equal size: at 1 h private better in 6 of 6 contrasts (+0.97 to +1.90 hard; see 7.5); at 5.4 h
   a tie. Not tested at 100 h. This is v5.4's Q1.
7. Small per-user adapter on named parts: passed at one learning-rate grid, not robust to a wider grid; choosing the parts by error rate
   beats random parts by +0.64 to +0.80; the parts that help are generically hard phonemes, not speaker-specific ones.
8. Phone language model + beam search on the outputs: -2.3 to -2.7 on hard (kept as the text baseline).
9. Did not help: extra output tokens for context variants (FAIL), uneven reward ratios (collapse to silence), blank penalty (~0),
   wider specialists-only readout (cost ~1.9 until a direct path was added).
10. Checks: our labels agree with forced alignment within 4-5 %; 8-bit export costs at most 0.23; his recordings are not badly labelled,
   his extra error is the spontaneous-speech gap (outside recogniser vs dictionary: read speech 10.7, native conversation 21.0, his 19-20).

## 4. The plan, with decision rule and cost per step
Base text: v5.4 (Part B). Its decision rules (section 2), stop rules (section 10) and pivots (section 12) apply as written, with the
proposed corrections of section 4b once verified.
### 4a. Step 0 (midnight fallback; v5.4 target 4-5 GPU-hours; no test-set scoring; v5.4 order "Execute strictly in order", kept)
Verifier (review, 19 Nov 23:35) on the v1 split: my v1 order ran CPU audits first and timed 100 h; v5.4 says strictly in order and lists no
100 h timing run. Adopted (it tightens): v5.4's order, no 100 h training of any length under the fallback.
1. Step 0.1 data, splits, freeze prep: copy the historical 1 h comparator errors from the results log; check the provenance of every input;
   install the image library + download CIFAR-100 (~170 MB); fingerprint the CIFAR splits (45k/5k stratified, nested 10 % / 30 %); write the
   provisional size accounting and the private / shared / Q3-shared geometry; DRAFT (not adopt, not score) the new speech manifests of
   4b.4 / 4b.5 after an audit of which speakers were already used. Cost: CPU, ~1-2 h. Rule: any input whose provenance cannot be shown is
   listed in section 6 and not used.
2. Step 0.2 baselines: three-seed full-CIFAR shared baseline + compact ResNet seed 1 (time measured on a short run first). The three-seed
   5.4 h speech shared baseline runs PROVISIONALLY (the author 19 Nov review notes: plans approved, "do not be over cautious"): current base
   size and the old 5.4 h set, labelled provisional, rerun on the new nested sets once they are frozen (v5.4 itself: "if the 20 Nov
   freeze alters geometry, rerun affected baselines"). Rule: seed spread recorded; ResNet on the same split is the reference for 69 %.
3. Step 0.3 CPU audits: v5.4 section 2 estimator + unit tests on synthetic unequal clusters; coverage simulations at 0 % / 4 %; mixed-generic
   orthogonality audit; Q6 topology draft; COST (estimate from logs, not run) the six 15 % 100 h tuning proxies. Rule: estimator passes its
   tests and coverage is inside v5.4's band, else reported broken and Q1 / Q3 do not start.
   DONE 20 Nov 00:2x (CALIBRATION; code/small_net/sec2_est.py, sec2_test.py -> results/sec2_sims.json): v5.4 gives no numeric band, so the rule
   was set before the sims: 95 % t interval, df = k-1; PASS if coverage >= 0.90 in every plausible setting, < 0.85 = severe. Grid C 20/40,
   k 2/3/5, seed sd 0-2 points, interaction 0-1 point, weight skew 0.3/1.0, Delta 0 / 4 %, 1,500 worlds each: min coverage 0.9553, PASS
   (k = 2 over-covers: wide intervals). d_w equal-weight identity exact; Vc / Monte-Carlo variance 1.058 at C 40. Broken control (cluster-only
   variance, checkpoint df, seeds varying) covers 0.644: fails as required. Simulated seed sd (1-2 points) is larger than measured on base6
   (0.17-0.36 points at 5.4 h). Limit: binomial cells, normal seed effects; heavy-tailed seed effects not simulated.
   CIFAR timing 19 Nov 23:59 (1-epoch smoke): ~3.6 s / epoch for S and ResNet-32 (8.2 s first epoch incl. setup) -> 200 epochs chosen
   (~12-13 min / run, under the 20-min trigger). Baselines S seeds 0-2 + ResNet-32 seed 1 at 200 epochs running from 00:0x (run_base.sh).
   RESULT 20 Nov 00:50 (CALIBRATION, validation split 5k, test never loaded; code/small_net/results/*_ep200.json): S (499,689 params)
   val acc 0.7014 / 0.6970 / 0.7052 (seeds 0-2), 532-582 GPU-s each. ResNet-32 (470,004) seed 1: 0.7022, 1,261.7 GPU-s = 21.0 min.
   Literal check of the trigger ("If full-data CIFAR exceeds 20 minutes/run"): S runs meet it (9-10 min); the ResNet comparator run does
   NOT (21.0 min, uniform ~6.3 s / epoch: the 1-epoch smoke hid it under setup time). Q1 arms (S/P/G) have S's trunk, so ~10 min expected.
   GPU used so far in Step 0: ~51 min (smoke 1 min + batch 50 min). Lesson: time smokes over >= 3 epochs per arm.
4. Step 0.4-0.5 only if time remains (v5.4 items 4-5); drop order as v5.4.
### 4b. Corrections (MAJOR). Second verifier (review, 19 Nov 23:35): all five ADOPT WITH CHANGE; the agent checked its claims at source
(warp code, v5.4 Step 0 text, section 2 wording: confirmed). ADOPTED 19 Nov 23:5x (second verifier + the agent; the agent's integrity check PASS).
1. SIZE DEFINITION. v5.4: "2 MB = 2,000,000 bytes of fp32 weights (~500,000 parameters)". the author 14 Nov: 2 MB on the higher of disk and
   memory while running, on the phone, where the speech model runs in 8-bit. Our speech models: 0.66-1.7 million parameters (1.4 MB exported
   8-bit). Under v5.4 they are all over budget. Proposal: speech arms = the larger of the deployment file and peak on-phone memory <=
   2,000,000 bytes (verifier: freeze device, runtime, batch-one workload and what memory counts); matched parameter budgets within each
   comparison; parameter count, fp32 bytes and compute reported separately; images keep v5.4's fp32 rule. Whether any current model meets
   the phone cap is UNKNOWN (phone memory never measured). '~4x' is weight-only arithmetic, not a measured cost.
2. SPEAKER IDENTITY IN THE WARP. v5.4 Q3: "Warping uses zero oracle speaker information." Our warp chooses a setting per UTTERANCE from the
   summed scores of the same speaker's other utterances (true speaker labels; leave-one-out, so it can differ by utterance). The label-free version (only earlier speech) cost ~0.9 (18 Nov). Proposal: Q3's
   warp arm = label-free version; the label version reported as a second row named oracle-assisted. Verifier change: before calling
   it deployable, define how earlier speech is tied to the speaker without true labels, stream order, resets, cold-start fallback.
3. 100 h COST. v5.4 assumes 2.0-2.5 GPU-h per full 100 h run, stops at > 2.5. Measured 15 Nov: 1.7 M parameters, 20 passes over 100 h =
   3.88 GPU-h (13,981 s). On the measured figure the trigger fires at once. Verifier: at 3.88 h, two seeds each of private, augmented shared and raw
   shared = 23.28 h; six 15 % proxies ~3.49 h; third seeds +11.64 h: ~38.4 h for 100 h work alone. Cut scope or re-approve; no 100 h
   timing run under the fallback.
4. TOO FEW SPEAKERS, NOT SEALED. Selection set 4 speakers, main test 6: v5.4's speaker-clustered uncertainty gets 3-5 degrees of freedom;
   both sets scored many times since 14 Nov (count to be audited). Proposal: new speaker-disjoint sets built once before any scoring: validation = the ~23
   unused speakers of the hard development pool; sealed test = the whole hard test pool (33 speakers) + clean test. Old sets = history. Verifier change: audit exposure and
   speaker ids first (the ~23 unused and 33 untouched are not yet shown); reserve Q2 adaptation and non-target pools and settle Q5 reuse
   before locking the test; where no untouched holdout can be shown, confirmation is reported unavailable.
   AUDIT 20 Nov 00:0x (step0/exposure_audit.py -> exposure_audit.json; CPU, nothing scored): the 23 unused hard-development speakers EXIST
   (never in an eval set; they were TRAINING data in the parked 1c arm B runs only) -> validation candidate stands. The hard test pool is NOT
   untouched: old 'testother' (330 utterances) already contains ALL 33 test-other speakers and was scored many times. Clean test (40 speakers,
   cleanT) was built 17 Nov and never scored. No untouched hard speakers are on disk. Options for a sealed hard test: (i) download LibriSpeech
   train-other-500 (~30 GB; scratch storage has 1.6 TB free) and draw held-out speakers from it (same difficulty-based selection as dev/test-other);
   (ii) use test-other utterances never scored (same speakers: speaker-exposed, weaker); (iii) report hard confirmation unavailable. Not decided.
   CORRECTION 20 Nov 01:27, kept next to the old claim: the 23 dev-other speakers are NOT unexposed. v1 of the audit read
   base5_devother_meta only as the parked 1c arm B pool, but base5_next.py:294-295 and struct2c.py:197 also SCORE the whole dev-other set
   (33 speakers, 2,863 utts, 5.13 h) as the eval column 'devother', reported in STAGE1_RESULTS.md (e.g. lines 30, 53, 136, 148) and in
   >= 105 result/log mentions (step0/exposure_audit_v2.{py,json}: dev_other_unused_by_eval = [], test_other_unused_by_eval = []).
   It was report-only (never used for selection, base5_next.py:290), but scored and seen many times = exposed. So NO hard speakers on disk
   are unexposed; cleanT is still never scored (phase0_sets.py:5, only built). Consequence for Part C C2.0 d: its predeclared fallback
   applies unless train-other-500 (download running) passes md5 + label checks and supplies a sealed hard panel (v6.3 change).
   DRAFT manifests (step0/draft_manifests.json, seed 20261120, not adopted): training pool = train-clean-100 minus the 20 cleanS speakers =
   231 speakers, 92.679 h. Whole-speaker nesting gives 1 h = 3 speakers, 5.4 h = 14 speakers (each speaker ~25 min), so a size curve would
   also change speaker count 3 -> 14 -> 231. Option B: the same 40 seeded speakers at 1 h (307 utts, 1.073 h) and 5.4 h (1531 utts, 5.436 h),
   utterances nested. Choice of A / B and of the hard test route goes to the 20 Nov freeze (second verifier + the agent).
5. NESTED DATA. The 5.4 h set (40 clean-development speakers) and the 100 h set (251 speakers) share no speaker, so 5.4 h is not a subset.
   Proposal: draw 1 h and 5.4 h as nested whole-speaker subsets of the 100 h set (seeded), so data-size curves change only the amount. Verifier change:
   predeclare duration tolerances, record realised hours / speakers / utterances / fingerprints; adding speakers also changes speaker and
   phone mix, so results are for these nested draws, not hours alone.
### 4c. After Step 0
1. Tranche 1 = Step 0 + Q1 (private vs shared) + Q3 (engineered nuisance handling vs data): needs the author's explicit approval (v5.4: 20 Nov).
2. Tranche 2 (Q2, Q4, Q5, Q6) and Tranche 3 (Q7 scaling): separate approvals, conditional on Tranche 1 (v5.4 sections 4-9).

## 5. Budget and stop points
1. Step 0: target 4-5 GPU-h; no test evaluation. GPU only through the shared queue and lock.
2. Tranche 1: hard cap 40 GPU-h (v5.4). v5.4's legacy subtotal (~31.5-32.5 GPU-h) assumed 2.0-2.5 h per 100 h run; at 3.9 h the fully
   seeded scope does not fit (a reduced scope may):
   the ledger is rebuilt from Step 0's timings before approval is asked.
   Built 20 Nov 01:12: step0/ledger_v61.{py,json}; totals and options in Part C, C5-C6.
3. Stop: any 100 h run > 2.5 GPU-h -> cut or re-approve (v5.4 section 10). Programme review 26 Nov (v5.4): negative or inconclusive
   results redirect, never move locked goalposts.
4. Spent since 19 Nov 23:3x: 0 GPU-h; one review verifier call (done 23:35).

## 6. What may be broken (open list; update in place)
1. Corrections 4b.1-4b.5 (pending verifier).
2. Q1 needs a 100 h mixed-generic arm for any full-data mechanism claim (v5.4 section 11); not in the Tranche 1 ledger.
3. Q2 needs >= 10 + 10 held-out adaptation speakers with enough speech each (v5.4 section 4 allows a single-pool fallback); whether two
   adequate pools plus a harm bank exist is UNKNOWN; decide before 4b.4's split is frozen.
4. Budget (section 5.2).
5. 100 h data path (measured 19 Nov): features ~17.5 GB per speed copy; 31 GB RAM, ~12-14 GB available; the training script loads every
   feature into RAM. A memory-mapped loader with the warp applied per batch is needed before any 100 h run or timing; not built; the warp
   settings script for 100 h is written, not run. Disk free: project drive 112 GB, scratch drive 1.6 TB.
6. Small-data effect risk (the agent's index): the tricks may only pay at small data; row 4c.1 Q3 tests exactly that.
7. No outside model at the same size on the same sets yet.
8. Logging (verifier, 19 Nov; code checked): the scorer returns only a rounded set error and a length ratio; v5.4 section 2 needs
   unrounded edit and reference-phone counts per item, speaker, arm and seed, and its weights must read 'reference-phone weighted'
   (v5.4 line: 'frame/utterance-weighted'). Add the logging before any Step 0 speech baseline.

## 7. Answers to v5.4's [confirm: the agent] items (checked 19 Nov 23:3x at source unless marked)
1. Speech metric: phone error rate, not frame error; greedy output, repeats and blanks collapsed, edit distance to the 39-phone reference;
   set score = total edits / total reference phones. VERIFIED (training script, evaluate()).
2. Corpus: LibriSpeech; 5.4 h = 40 speakers (30 + 10) from the clean development pool; 100 h = train-clean-100, 251 speakers; no shared
   speaker. VERIFIED. v5.4's "data plus corpus changed" label applies to 5.4 h vs 100 h.
3. Selection set 300 utterances / 4 speakers, main test 300 / 6 speakers, both hard development pool, disjoint from training and each
   other. VERIFIED. A third set is from the hard test pool (speaker count not yet checked).
4. Historical 1 h comparator errors, copied 19 Nov 23:5x from the results log (19 Nov 08:5x, 18:09, 18:57 blocks; one model per
   (draw, seed); "hard" = mean of the three old hard sets, 4 + 6 + test-pool speakers, NOT sealed). Shared top (comparator):
   draw 0 seed 2 49.71; draw 1 seed 2 49.92; draw 1 seed 3 50.33. Private at shared's bytes: 48.67 / 48.95 / 48.99. Private at the
   smaller size: 48.47 / 48.95 / 48.43. VERIFIED (copied from the log, not rescored).
5. Size of the historical effect, from those values: shared minus private-at-equal-bytes +1.04 / +0.97 / +1.34; shared minus small
   private +1.24 / +0.97 / +1.90. So the 6 contrasts span +0.97 to +1.90 absolute points (1.9 % to 3.8 % of the comparator's error).
   CORRECTION 19 Nov 23:5x: my earlier "+0.80 to +1.34" (results log 18:57 block; working notes) is not what these six values give;
   0.80 was a different contrast (dense-cut minus small private). The old line stays in the log; this row replaces it here.
6. CIFAR-100 not on this machine; GPU environment has no image library. VERIFIED.
7. Speech models use LayerNorm, not BatchNorm: Q5's BatchNorm rules apply to images only. VERIFIED (model code).

## 8. Change log
1. 19 Nov 23:3x v1: v5.4 read in full; local facts, 5 proposed corrections, Step 0 split, open list. review (second verifier) started.
2. 19 Nov 23:34: 100 h memory fact added to the open list.
3. 19 Nov 23:36 v2: reordered into standard sections (goals, state, methods, plan, budget, change log); content unchanged
   except the added goals, state and methods sections; link placed in the plan
4. 19 Nov 23:37: review verdicts applied (all five ADOPT WITH CHANGE; claims checked at source); Step 0 back to v5.4's strict order;
   no 100 h timing run; logging gap added (6.8). the agent's check requested.
5. 20 Nov 01:0x: Part C v6 written (the author's ideas first); review round 1: READY no (6 blocking).
6. 20 Nov 01:12: Part C v6.1 (all round-1 items answered; per-run ledger step0/ledger_v61; backup .v3_pre_v61_2026-11-20); sent as round 2.
7. 20 Nov 01:23: Part C v6.2 (round-2 items; ledger_v62 from the measured timing table; backup .v4_pre_v62_2026-11-20); round 3: READY no, 8 blocking.
8. 20 Nov 01:5x: Part C v6.3 (round-3 items answered with code: manifests, eval allowlist + sealed dir + score_locked.py, sec2_abs.py, E1 CPU
   harness, ledger_v63, gpu_gate.py; dev-other exposure correction in Part A 4a; Q2 (c) recipe corrected; backup .v6_pre_v63_2026-11-20); round 4.
9. 20 Nov 02:4x: Part C v6.4 (round-4 items: ledger_v64, manifests_v64, sec2_abs adapter + harm calibration, E1 harness v6.4, batchplan_v64,
   train-other-500 md5 verified; unfinished tooling moved into the C9 launch checklist; backup .v7_pre_v64_2026-11-20); round 5 (last).
10. 20 Nov 03:05: round 5 READY no (9 blocking, all enforcement tools; research design accepted); cap reached -> review note; the author: "B is safer.
   Extra call is cheap" -> build the tools, then round 6.
11. 20 Nov 07:4x: Part C v6.5 spliced (backup .v8_pre_v65_2026-11-20): ledger_v65, manifests_v65, gpu_gate2 + queue integration + launch
   audit, reader_v65 (REGISTER / BIND / RELEASE), sec2_q2c_entry, per-item logging in both trainers, E1 v6.5 PASS on five paths, Q2 (c)
   pairing evidence + runner, winner binding, cadence check, hard-panel checks (2)-(5) with a dated amendment of (4); round 6.


---
# Part C. Revision v6.5 (the agent, 20 Nov 03:3x): the author's own ideas move forward. PROPOSED, under review (round 6, the author T23: "B is safer. Extra call is cheap"); Part B stands until it passes.
the author 20 Nov ~01:0x: "feel free to revise it to make my ideas more main just make sure the new version passes review. Making corections without
resending is not a pass". Review rule: findings -> fix -> RESEND the whole file; only a clean round (READY yes, no blocking item) is a pass; at most
5 rounds (researcher.md 6.3); an item still blocking after round 5 goes to the author as a review note. Rounds: r1 (on v6) READY no, 6 blocking; r2 (on
v6.1) READY no, 6; r3 (on v6.2) READY no, 8; r4 (sol_r4.md, on v6.3) READY no, blocking C2.0 C2.4 C4 C5.1 C5.3 C5.4 C5.6 C5.7 C5.8 C6. Every r4
claim was checked at source by the agent (20 Nov 01:56-02:00) and found right. r5 (sol_r5.md, on v6.4) READY no, blocking C2.0 C2.4 C4 C5.1
C5.3 C5.4 C5.7 C5.8 C6 (research design judged right; blockers = enforcement tooling); the open items went to the author (review note); his
answer: "B is safer. Extra call is cheap" = build the tools, then ONE more round. THIS IS ROUND 6 (on v6.5). Round-5 map: C11.
WHAT CHANGED IN SHAPE (since v6.4; v6.5 builds and tests most of C9's tools, C11): earlier revisions described controls as done when the code did less, and each round the code showed it. v6.4
states only what the attached code does now (C2-C5), and puts everything still to be built into ONE LAUNCH CHECKLIST (C9): each item with
an acceptance test, a deadline in the launch order (C5 8), and the rule that no GPU job runs on a path whose C9 items have not passed. The
approval asked of the author (C6) is approval of scope, cap and rules, NOT a statement that the tooling is finished.
Source of the ideas: IDEAS_STATUS_2026-11-18.md (22 ideas from his 14 Nov spec; lines 8-25).
Prices: step0/ledger_v65.py -> ledger_v65.json (ONE ROW PER RUN, row key group|arm|size|seed; full precision; every total is a full-precision
sum, the 10 % reserve applied, then ONE round to 2 dp for display). Every total below is printed by that script.

C1. Owner idea -> the question it motivated (a question tests what its design tests, not the whole idea). UNCHANGED (r3 OK, r4 OK).
   1. Idea 7 (shared trunk, split heads, specialists, one joint decision) motivated Q1 and Q2. Q1 tests only private vs shared ALLOCATION at
      equal bytes, not the specialist architecture as a whole. Q2 tests semantic vs shuffled wiring for editing ((a), (b)) and for low-data
      learning ((c)).
   2. Idea 16 (heads keep their meaning) motivated Q6: does specialisation persist under joint training when another path exists.
   3. Idea 15 (start from knowledge) motivated Q4: written INITIALISATION followed by training, not written weights replacing training.
   4. Idea 12 (per-user layer) motivated Q5, which generalises it to several adaptation shapes and to forgetting, in speech and images.
   5. Idea 1 (remove the speaker first) motivated the WARP part of Q3 only. Q3 asks whether label-free warp+noise substitutes for data at full
      data, plus a CIFAR flip analogue; it is NOT answered by any earlier result. Earlier warp results (speaker-level estimates, none label-free,
      none at full data, none on images): 2-I error lower by 2.84 on the hard set at 5.4 h, 2-V lower by 2.41 at 1 h (Part A 3.2); the older 2a
      figures 3.90 / 2.51 on the 0.66 MB base5 are SOURCE-REPORTED (STAGE1_RESULTS.md:195). Motivation only.
   6. Already in v5.4 as Tranche 3 proposals (Part B 9): the agent repair loop (idea 18) and OPAL-style optimizer rewriting (his 14 Nov link).
   7. Not in v5.4; no matching test in the audited log (IDEAS_STATUS_2026-11-18.md): timing channels (idea 6, line 13), cross links (10, line 17),
      sequence compartment (14, line 21; only its phone 5-gram baseline exists). Partly tested on the old pipeline: named channels (5, line 12),
      accent prototypes with the OR gate (11, line 18). Q7 is downstream scaling.

C2. TRANCHE 1 v6.5 = Step 0 rest + Q2 (c) + Q4 at 5.4 h + Q8 screens + Q1 (part); headroom H1 outside reference, H2 Q2 (c) at 1 h. His ideas
   run first (C5 8). Each part says RETAINED (confirmatory, B design), DEFERRED (named, with the claims it takes along) or EXPLORATORY.
   0. SPLIT LOCK. Corrections kept, dated: (i) the 5.4 h set is 40 DEV-CLEAN speakers, not train-clean-100 (v6.2 error); (ii) the 23 dev-other
      speakers are EXPOSED (whole dev-other was eval column 'devother', base5_next.py:294-295, struct2c.py:197; exposure_audit_v2.json; Part A 4a).
      a. MANIFESTS, built and checked (step0/manifests_v65.py -> manifests_v65.json; v6.5 adds an EXPLICIT required-input list of every cache file the R paths read, incl. speed copies, speaker statistics and both warp-estimate families: a missing file fails the build; asserts cleanS vs every exposed set, clean test vs libriB, libriA exactly 4 speakers, libriB inside exposed dev-other (exposed sub-roles may overlap; that is the stated invariant), no duplicate ids in cache order): per role speaker list, utterance ids in CACHE ORDER, hours,
         counts, sha256 of the sorted and of the ORDERED id list, and sha256 of every cache file read (features base5_fit / cal / cleanS /
         libriA / sealed cleanT, base5_speakers.json, every ph_*.npz label file). Asserted (script exits non-zero otherwise): training,
         validation, clean test and the exposed sets (devother, testother, libriB) pairwise disjoint by speaker, except libriA, which is
         asserted to be 4 dev-other speakers (exposed, validation only); nesting 0.5 h in 1 h in 5.4 h by utterance AND by speaker. TRAINING:
         5.4 h = 40 dev-clean speakers, 2,703 utterances; 1 h = draw 0, 8 speakers, 539 utterances, 1.075 h; 0.5 h = draw 0, 4 speakers, 288
         utterances, 0.538 h. The 92.679 h pool of the deferred package (train-clean-100 minus the 20 cleanS speakers) does not contain the
         5.4 h set: any claim across those sizes says 'data AND corpus part changed'. libriA's cache keeps no utterance ids: its manifest is
         speaker-level plus the cache-file hash (it is validation only). The manifest hashes are copied into the freeze file (C9 1).
      b. VALIDATION: libriA (4 dev-other speakers, 300 utterances) + cleanS (20 train-clean-100 speakers). 'fit' curves are training
         diagnostics, not validation. Validation only, never reported as test results.
      c. CLEAN TEST: cleanT (test-clean, 40 speakers, 320 utterances, 0.757 h), never scored (exposure_audit_v2.json). Its cache was moved to
         cache/sealed/ (sha256 4cff140c...); the trainers' loader no longer finds it. This makes ORDINARY lookup fail; it does not by itself
         give exclusive access. Exclusive access is C9 2 (single controlled reader).
      d. HARD TEST, decided by a rule fixed now, at ONE fixed CUTOFF (v6.5: the calibration freeze, C5 8 c, before the first GPU job of ANY kind; the decision and its
         evidence are recorded there once, irreversibly), using no score of any kind:
         Pool: train-other-500. Archive md5 against the openslr list (https://www.openslr.org/resources/12/md5sum.txt, read 20 Nov 02:31:
         d1a0fd59409feb2c614ce4d30c387708): MATCH, VERIFIED 20 Nov 02:37 (scratch storage AI_scratch/hands-free-mobile/librispeech_other500/md5_check.txt). v6.5 STATUS of (2)-(5) (CPU,
         no model scoring, step0/hardpanel/): (2) PASS: phones.job unchanged reproduces the 20 smallest-hash fit utterances' labels exactly
         in BOTH ph_fit.npz and base5_fit.npz, frame counts equal, a one-phone change detected each time (check2_labels_result.json);
         (3) PASS: recomputed raw log-mel of 3 fit utterances == base5_fit X exactly, extractor code (base5_prep, bb142, frontend) unchanged
         since before the cutoff, sha256 recorded; (4) LITERAL v6.5 rule FAILS (base6.py, base6s.py, base6s_R.py were edited after 01:37,
         and new evaluation tools were created after it); AMENDED 20 Nov 07:4x, before the hard-panel cutoff and before any panel item was
         selected or scored (dated amendment, reviewed in round 6): model-path code edited after the cutoff passes by (a) a diff against
         its last pre-cutoff backup with no train-other-500 reference and every numeric diff line recording-only (base6 9 lines, base6s 14,
         listed in check345_result.json), or (b) with no pre-cutoff backup (base6s_R) the diff vs its pre-edit copy (docstring +
         RWIRE_SEED, default 0) plus a behavioural known answer (r=0 == the 18 Nov sigma; reader parity 29.44 vs 29.43); evaluation tooling
         (gate, reader, scorer, decision entries) is not an artifact a model reads and needs only (5); every data artifact (C2 0 a caches,
         trk31_fit, ph_fit, ph_cal) has mtime before the cutoff: AMENDED (4) PASS (check345_verdict.json); (5) PASS: static import closure of
         14 Tranche 1 entry scripts (62 local modules, data-path literals listed per module); the only train-other-500 mention is
         score_locked.py:20, a comment. SELECTION DONE 20 Nov 07:5x (custodian, no model, no
         audio, TextGrids only): 148,645 train-other-500 TextGrids extracted into a new folder (scratch storage AI_scratch/hands-free-mobile/
         align_other500, EXTRACT_LOG.txt); step0/hardpanel/select_panel.py applied the rule above verbatim: 40 speakers, 320 utterances,
         1.081 h; the first 40 speakers in sha256 order all qualified (none skipped or replaced); disjoint from all 166 manifest speakers;
         written once: hard_panel_record.json (ids sha256 3396e274...). Choice recorded for review: 'a phone the map cannot place' = a label
         neither in phones.PID nor a label the training pipeline treats as non-phone ("", SIL, SP, SPN); 25 of the 320 utterances contain SPN
         (spoken noise, dropped from references exactly as in training and in cleanS); the stricter reading (replace SPN utterances) is
         offered to the reviewer. Audio of the 320 utterances is extracted only at the calibration freeze. Labels: the Lugosch MFA alignments (zip md5 2bab567d0ace651a4ba254e813629f46 =
         Zenodo 2619474; 148,645 train-other-500 TextGrids). The panel EXISTS only if every check passes by the cutoff: (1) the archive md5
         matches; (2) the label pipeline that made the training labels, unchanged, reproduces the cached labels (ph_fit.npz) of the 20 fit
         utterances with the smallest sha256(utterance id) exactly; (3) features come from the frozen extractor (its code hash in the freeze
         file); (4) PROVENANCE: every artifact an evaluated model reads (frozen tree struct2c_AS_RUN, P1-P7 code and fitted numbers, the 5-gram
         text, normaliser code (NORM=causal, per-utterance, no fitted statistics), warp estimator code, cache files of C2 0 a) has an mtime
         and a content hash recorded BEFORE 20 Nov 01:37 (the start of the first train-other-500 download; before it no train-other-500 byte
         was on this machine), or is rebuilt from training inputs only with its input list recorded; (5) a code audit of every script on the
         Tranche 1 paths lists its data inputs. Selection: speakers ordered by sha256(speaker id); a speaker qualifies with >= 8 utterances of
         2-20 s that have a TextGrid; take the first 40 qualifying (if fewer than 40 qualify, all of them, and the panel exists only with >= 10);
         per speaker the 8 utterances with the smallest sha256(utterance id). An utterance with a phone the map cannot place is replaced by the
         next utterance of that speaker in hash order; a speaker without 8 mappable utterances is replaced by the next qualifying speaker.
         All of this is computed without any model. If ANY check is unresolved at the cutoff, this tranche is irrevocably locked CLEAN-ONLY:
         no later addition of a hard panel to the same headline, no substitution of exposed speakers. Population in every headline: 'clean
         test (test-clean)' alone, or 'test-clean + hash-selected train-other-500 speakers' (never 'test-other' or 'dev-other').
      e. TRAINER ENFORCEMENT (attached code): base6.EVALSETS (base6.py:55-57; base6s.py uses it), default fit, cleanS, libriA; an exposed set
         (libriB, testother, devother (added 20 Nov 02:3x), author2, him) is accepted only with LEGACY_EVAL=1; any other name fails. Every Tranche
         1 runner is started with env -i and a pinned environment (as the E1 runner, step0/e1_cpu.sh), so LEGACY_EVAL and EVALSETS
         cannot be inherited. These two protective defaults (allowlist, sealed directory) are valid under Part B as well (B also seals its
         tests); they are adopted with backups (*.bak_preAllow_2026-11-20) and can be undone by the author.
      f. SEALED SCORING: step0/score_locked.py exists (per-utterance edit and reference counts; refuses no lock, rewritten lock, a second
         scoring of the same model on the same set, a changed model hash), but it CANNOT make the pre-run lock this plan needs (make_lock needs
         trained .pt and run json, score_locked.py:30-37), does not bind config / analysis / panel manifests, and knows only cleanT. Replacing
         it is C9 2 (v6.5: the Q2 (c) decision entry with registered protocol and release digests is DONE and tested; the reader is not);
         until C9 2 passes, no sealed panel is scored by any model (custodian reads of bytes / ids / lengths for manifests are logged, C9 2).
      g. EXPOSED sets are reported in a separate block labelled exposed, never as confirmation; new utterances of exposed speakers are not a holdout.
      h. RESERVED for Tranche 2: train-other-500 speakers not in the hard panel are a CANDIDATE speech pool, not a certified holdout: they become
         eligible only through the same checks as C2 0 d (1)-(5).
      i. IMAGES: CIFAR-100 official test split, one locked pass through a CIFAR seal (C9 2); without it, NO CIFAR confirmation is claimed.
   1. Q1. RETAINED: speech 1 h co-primary P vs Shared: matched 3x2 weight-decay / dropout grid at seed 0 (6 cells per arm), seeds 1-2 on each
      arm's chosen setting, a FUNDED replacement seed 0 per arm (used only if the grid winner is not identical in every field): 18 runs. CIFAR
      10 % and 100 % co-primaries P vs Shared as B (3x2 grid at 10 %, three WD at 100 %, three seeds on the chosen setting, replacement seed
      funded): 27 + 18 runs with G_mix, which is a DESCRIPTIVE arm. Selection: the grid rule and candidate hashes are frozen before the grid
      runs; the resolved winner is recorded before any panel opens.
      DEFERRED, with what each removes: (i) 100 h speech: no full-data speech claim, no 'data-limited benefit' claim, no cross-domain
      full-data gate; its slot stays in the speech Holm family as nonrejectable (C4). (ii) speech 5.4 h grid and seeds: no middle-scale
      three-seed speech comparison, no 5.4 h speech Private bases for Q5. (iii) speech G_mix: no speech class-alignment or mechanism claim.
      (iv) the CIFAR P vs G_mix INFERENTIAL class-alignment contrast and B 12's mechanism pivot: no class-alignment, mechanism or equivalence
      reading from CIFAR. (v) CIFAR 30 %: no intermediate-scale image claim.
      100 h PACKAGE (option C6 3 only): staged two-arm P vs augmented Shared under B 3's proxy rules. NOT launchable until its own freeze
      exists, written and reviewed before any of its runs: B 3's tie-break and proxy-failure rules as code with known answers; the label-free
      augmentation identical for P and augmented Shared (frozen label-free warp estimator and its speaker association rule, one seeded noise
      stream per utterance and pass, the same order, reset per seed, the same cold-start rule); the memory-mapped loader built and timed; the
      realized pool duration recorded. Rows: 5.4 h grid cells for both arms (12; they choose proxy candidates only), four 15 % proxies with
      candidate ids at matched seed 0 and data order, three full seeds per arm (6). Cost 31.59 GPU-h raw, 34.75 loaded.
      SIZE CONTRACT (A 4b.1, his 14 Nov rule 'the higher number as the limit'): speech cap = max(deployment file bytes, peak on-phone memory
      while running) <= 2,000,000 bytes with matched trainable-parameter budgets. Peak memory is UNMEASURED: every speech arm is a FILE-SCREENED
      RESEARCH ARM (file <= 2,000,000 bytes) and no result is called phone-cap compliant. Images keep B's fp32 rule.
   2. Q2. RETAINED: (c) his tree readout vs the same model with SHUFFLED wiring at 0.5 h (primary), five paired seeds. Recipe (corrected in
      v6.3): base6s.py arm S (each specialist reads the heads trained on its own phonetic questions) vs base6s_R.py (arm S with every head
      index g replaced by sigma(g), a derangement of the 18 heads; fan-in, sharing pattern, mask size and bytes unchanged). Both train through
      base6s.train("S", ...); the only code difference is the wiring table X.PATHS (base6s_R.py:11-15). Selftest for RWIRE_SEED 0-4 passed
      (step0/rwire_selftest_2026-11-20.txt); the five permutations differ. Pair r = seed r with RWIRE_SEED = r (the runner asserts it and the
      realized sigma is written into the run record: C9 4). LIMIT on reading: a derangement of indices does not guarantee that a specialist
      loses its whole original head SET, nor that the substituted questions are unrelated; so a WIN supports 'the semantic wiring beats a
      shuffled one at 0.5 h', and an INCONCLUSIVE result is NOT read as 'shape, not phonetics'. Schedule (dry-built in the run's own schedule,
      step0/batchplan_v64.json): 0.5 h draw 0, 288 utterances, 1,836 passes, 9 batches per pass, 16,524 optimizer updates for every seed 0-4
      and both arms; OneCycle over exactly those updates. One fixed configuration per arm; no selection and no validation gate (validation
      curves are logged and choose nothing). A run that crashes or has a non-finite loss is reported; no replacement seed; fewer than 5
      complete pairs = INCONCLUSIVE (enforced in code, C4 1). Price class plain. (c) at 1 h is headroom H2; unfunded it is deferred, no 1 h claim.
      DEFERRED: (a), (b), with what they remove: no new-speaker editing, confusion repair, editable-unit superiority or equivalence, or harm
      certification for those edits. Their panels come from C2 0 h; when funded, their lock fixes the panels, the out-of-fold confusion rule for
      (b), matched edit budgets, and the 0.5-point harm protocol with 95 % selection UCB and a separate 95 % certification.
   3. Q4. RETAINED at 5.4 h, B's full primary family: R, D, W, WH full length x 3 paired seeds (12 runs) + W, R compressed half schedule x 3 (6);
      Holm across W-R, WH-R and speed (intersection-union); W-D under B's equivalence rule. Architecture (the agent's freeze choice): the plain
      Shared base at the 2 MB file screen, the SAME in all four arms; if WH needs specialist heads, the matched architecture changes for ALL arms
      and the family is repriced before the freeze. UPDATE TARGETS (integers, per seed, identical across arms within a seed; batchplan_v64.json):
      full 180 passes = 15,084 / 15,093 / 15,085 updates for seeds 0 / 1 / 2; compressed half 90 passes with its own complete OneCycle =
      7,546 / 7,549 / 7,543. The LR trajectory of each schedule is checked in its first run's log against the planned OneCycle. NO WARMUP:
      W and WH load their written weights and train on the same schedule as R and D. The WH smoke (GPU, device cuda, allowance 0.05 GPU-h,
      ledger row) checks that the CPU-written WH trains; if it does not, a matched warmup is added to every arm and priced before the freeze.
      SPEED: the intersection-union test establishes performance at a smaller update budget; it is called a wall-time speedup only if the
      measured GPU times of the half and full arms support it. P1-P7 provenance: the programmed units read only the 'fit' split
      (d1_punits.py:33 and :127, b.labelled("fit")), inside the 5.4 h training set; the full input audit of W / WH construction is C9 1.
      No WT arm, so no claim specific to written heads (WH-W is composite). DEFERRED: 1 h (secondary scale); no 1 h claim.
   4. Q8, EXPLORATORY screens of the author's ideas: (a) cross links, (b) accent prototypes with the OR gate as a shift on the trunk, (c) sequence
      compartment, at 5.4 h on the WN recipe, 2 seeds each. Comparators: (a), (b) = the Shared base pair; (c) FREEZES the base pair's acoustic
      checkpoints and trains ONLY the sequence compartment on the training transcripts and the frozen acoustic outputs; comparator = the same
      checkpoints decoded with the frozen phone 5-gram (CPU-trained on the training transcripts only). So (c) is an isolated sequence-model
      effect, not a system retrain. PRICE, one rule in prose and code (ledger_v65.py small_h: margin on the WHOLE run): each screen run =
      1.2 x a full 5.4 h plain run = 0.598257 GPU-h (v6.5, full precision). For (c) this is an ALLOWANCE, not a 15,091-update acoustic run:
      it covers generating the frozen acoustic outputs on the GPU (charged ONCE, here) and training the sequence part; (c)'s own update count
      is set at its freeze. Its TIMING JOB is a named GPU smoke row (0.05: frozen outputs for a 5 % slice + one sequence epoch), run before
      seed 0, which reprices (c). The 5-gram is trained and decoded on the CPU (0 GPU-h). Base: the Step 0 Shared baseline seeds
      0-1 if identical in every field; else ONE replacement base pair (0.997095 = 2 x 0.498548), serving (a)-(c). If a fair comparator cannot be
      funded, the screen is cut before scoring. (d) named + timing bundle DEFERRED (bundle claim only when run). Each idea gets a written
      design first (his 6 Nov rule). Frozen before scoring, per idea: one configuration, seeds, validation set, metric, allowance, reuse rule.
      Rule name: 'ELIGIBLE FOR CONFIRMATORY FOLLOW-UP' if the validation relative error falls by >= 4 % in both seeds. Triage, not a
      calibrated screen: clearing it does not show a true >= 4 % gain; missing it neither refutes the idea nor shows equivalence (NOT KEPT,
      OPEN with its reason; never REFUTED without a mechanism, his 7 Nov rule). Every candidate and both seed effects are reported.
   5. OUTSIDE REFERENCE = headroom H1, a DESCRIPTIVE option. H1's 1.55 GPU-h raw (1.70 loaded) funds ONE timed smoke and TWO runs of the
      outside recipe only; it funds no internal pair. Frozen before its first run: source-selection rule (fresh search; a published small
      streaming recogniser with public code and config; eligibility = public code + config + output mappable to our phones; tie-break = file
      size nearest 2 MB, then the most recent release), the candidate, its recipe at published DEFAULTS (no tuning on either side), training
      subset 5.4 h, two seeds, its cost repriced from the smoke, the phone mapping (no mapping = no comparison), no pretraining, no text beyond
      what S sees, greedy decoding without LM. Our side: the Step 0 Shared pair only if identical in every field; else a new pair must be funded
      separately (a priced change) or H1 is cut. Scoring: VALIDATION SETS ONLY by default; sealed scoring only through a verified outside-model
      adapter in C9 2 and a protocol registered before first panel access. A file over 2,000,000 bytes or look-ahead above 992 ms is a
      MISMATCH and is reported as one. No fair comparator = say so and claim no industry-standard win.

C3. OTHER QUESTIONS
   1. Q3: DEFERRED and OPEN: no full-data nuisance-persistence or data-substitution result, no image corroboration. Designs kept for separate
      approval. Anchor reuse only if config, split, seed, data order and augmentation stream were frozen before its training and its warp is
      label-free (C2 1); otherwise a Q3 rerun at a PROVISIONAL 3.88 GPU-h per seed.
   2. Q5 (Tranche 2): no confirmatory reuse of Tranche 1 panels; Q5 builds and prices its own bases. Confirmation by domain: SPEECH from
      reserved train-other-500 speakers only if they pass C2 0 h; IMAGES need fresh source images (corruptions of exposed CIFAR test images are
      not independent); a domain without an eligible fresh pool is EXPLORATORY. Removed from Tranche 1: adaptation / forgetting results, Q5
      pilot, generator check, selection-UCB width work.
   3. Q6 (Tranche 2): topology audit drafted in Step 0 (CPU). Removed from Tranche 1: no learned-specialisation retention claim.
   4. Tranche 3 gates as B (Q7 needs a confirmed scalable Q1/Q2 win; repair loop and OPAL need Tranche 1-2 evidence of a bottleneck caused by
      standard backprop or static graphs, with equal-compute controls). An exploratory lead unlocks nothing.
   5. Step 0 optional items closed out: grid pilots absorbed into the funded Q1 grids (no separate pilot readiness evidence); ResNet seed
      spread dropped (C5 5); Q5 UCB width work moves to Tranche 2.

C4. INFERENCE. B 1-2 and B 11 apply to every confirmatory claim.
   0. (v6.5) Q2 (c) is a FIXED-RECIPE comparison with NO validation gate: an explicit exception to any inherited Part B 2 same-sign screen
      requirement. Its verdict comes only from code/small_net/sec2_q2c_entry.py (C9 2). Unequal panels admitted by C2 0 d (clean 40 + hard
      10 to 39 speakers) are calibrated: 40 + 10 / 15 / 20 / 30 / 39, hard-panel spread equal or 2x, four variance patterns, 4,000 draws
      each: coverage 0.98125 to 1.0, band >= 0.97 written before the run (results/sec2_abs_unequal_sims.json, PASS). A simulation is
      calibration, not empirical proof of coverage.
   1. Q2 (c) (code/small_net/sec2_abs.py). Input: from_counts() takes the per-utterance rows (panel, arm, seed, utterance, speaker, edits,
      reference length) and REFUSES unless every panel has exactly arms S, R x seeds 0-4, identical utterance ids, speakers and reference
      lengths in every cell, finite non-negative counts; it returns speaker-level PER differences. decide() REFUSES anything but exactly 5
      complete pairs. d[c, r] = PER_S - PER_R (points); Delta = sum_c w_c mean_r d[c, r] (absolute, linear; sec2_est.py is B 2's relative-error
      estimator and is NOT used here). Weights: panel weights 0.5 / 0.5 or clean-only 1.0 (C2 0 d), each speaker of a panel p / C_panel.
      Variance: Vc, Vs, Vi; V = max(Vc, Vs, Vc + Vs - Vi); df = min(k - 1, C_eff - 1). WIN = the two-sided (1 - .05/3) interval below 0 AND on
      EACH panel the one-sided 95 % upper bound of that panel's Delta <= +0.5 point; LOSS = interval above 0; else INCONCLUSIVE; no
      equivalence claim. CALIBRATION (bands written before the runs): two-sided coverage 0.9855-1.0 vs nominal 0.983, broken control 0.10 /
      0.16 (results/sec2_abs_sims.json); HARM bound (sec2_abs_harm_test.py -> results/sec2_abs_harm_sims.json, PASS): false certification at
      a true +0.5 at most 0.0498 (band 0.06), at +0.6 at most 0.038 (band 0.05), panels of 40 and of 10 speakers, four variance scenarios;
      power at a true -0.5 from 0.33 (10 speakers, speaker-dominated) to 0.999; known answers: a result that wins on the primary interval but
      has one panel at +0.8 is not WIN; 4 pairs refused; missing slot, changed items and NaN refused. B's checkpoint-conditional reporting
      is kept beside it.
   2. Q1 and Q4 use B's relative-error estimator sec2_est.estimate(eA, eB, w) with an EXPLICIT speaker-equal weight vector: per-speaker
      edit and reference sums from the per-utterance counts, each speaker of a panel weight p / C_panel, panel weights as C2 0 d. The
      scorer's pooled PER is never used for inference.
   3. Families. Q1: Holm within each task; a deferred co-primary keeps its slot as p = 1 now. Q2: three-member Bonferroni at exactly .05/3
      each. Q4: three-member Holm with the intersection-union speed test and NI / TOST at the Holm-matched level; W-D under B's equivalence
      rule. LATER PASSES: a Bonferroni member ((a), (b)) is tested once, in its own locked pass on fresh panels, at .05/3; the deferred Q1 100 h
      member is tested once, later, at .025 (the floor of the two-member speech family). Current decisions are never reopened; no alpha is
      spent twice.
   4. Harm rules: Q2 0.5 point (C4 1). Q5 1.0 point in speech and images (80 % selection UCB, reserve zero, separate 95 % certification).
      Descriptive only: CIFAR P vs G_mix, the Q4 R-half vs R-full diagnostic, Q8, calibration runs, the outside reference. B 11's ladder kept.
   5. A clean-only primary is named in the headline of every affected result.

C5. LEDGER, TRIGGERS, CAP, ORDER (GPU-h from step0/ledger_v65.json; every total = full-precision sum of rows, 10 % on new work, one round)
   1. PRICES. Eligible timing logs (v6.5, audited in code, ledger_v65.py eligibility block): full schedule in one log (180 passes at 5.4 h,
      972 at 1 h) with 31 / 163 evaluations, no .resumed name, AND (v6.5) the training hours pair with the schedule (5.3-5.5 h for 180,
      1.0-1.15 h for 972), total updates inside the dry-built per-seed set (15,084-15,093 for 180; 16,524 for 972), the SOURCE LOG exists in
      logs_base6/ and its last line is a final scoring line ("<set> {'PER': ...}"), and the arm variant is in the explicit PLAIN or TREE
      list (an unknown variant stops the script; nothing defaults to plain). 59 logs pass all checks (41 plain, 18 tree); the eligible 972-pass
      logs are all tree-class, so the 1 h PLAIN price is the per-step rate from 5.4 h plain logs applied to 16,524 updates (an extrapolation,
      labelled so; the first 1 h plain job is its timing run, C5 8). Coefficients: plain step median 0.0843 s (max 0.1344, n 41), tree 0.108 s
      (max 0.2294, n 18); eval per curve point 4.22 / 4.38 s (max 10.34 / 10.67); setup + final scoring 396 s per run (MEASURED-DERIVED).
      EXPECTED = (exact updates x median step + curve points x median eval + 396 s) x margin, margin on the WHOLE run. UPDATES (v6.5): each
      row carries the exact integer of the dry-built schedule for its seed (batchplan_v64.json): Q2 (c) 16,524 per seed; Q1 1 h 16,524;
      Q4 full 15,084 / 15,093 / 15,085 and compressed half 7,546 / 7,549 / 7,543 (seeds 0-2); rows without a dry build (none in R) would show
      updates_source 'UPD x frac'. CURVE POINTS: 30 for every run including the compressed half (EVAL_N=30 is the RECIPE, set explicitly; the
      old six-pass cadence would give 15 events on 90 passes, so the half arms' curves are denser than the old rule would make them; E1
      covers that, C5 4). Per run: plain 0.532 (0.5 h, 1 h), 0.499 (5.4 h), half ~0.32; tree 0.642 / 0.599. CIFAR 100 %: Shared 581.9 s =
      0.162 (measured maximum of three runs), P and G_mix = x 1.2 computed; CIFAR 10 % 0.04 (INFERRED). 100 h 3.88 (MEASURED). Old-rule
      increments: 0.5 h +0.324 per run, 1 h +0.155. Classes not yet timed in their own configuration (1 h plain P / Shared, 0.5 h S / R,
      CIFAR 10 %, CIFAR P and G_mix, Q8 paths, the half schedule): the FIRST full job of each class is its timing job (C5 8 d).
   2. v5.4's listed Q1 scope at v6.1 prices = 47.64: a LEGACY ledger_v61 output for the rows v6.1 listed, not a reconstruction of Part B.
   3. SCOPE R (ledger_v65.json): 115 rows = 102 GPU training runs + 10 GPU calibration rows (8 CUDA E1 arms, WH smoke 0.05, Q8 (c) timing
      smoke 0.05) + 3 CPU rows. New GPU-h by group: Step 0 2.42 (baseline triplet 1.50 + CUDA E1 0.93); Q2 (c) 5.32; Q4 7.96; Q1 13.96;
      Q8 4.64. R new 34.30 + 10 % 3.43 + spent 0.85 = 38.58. SENSITIVITY 63.28 (labelled; not a bound: the cap is held by the gate, C5 7).
      SPENT 0.85, itemized per job (CIFAR S seeds 0 / 1 / 2 = 581.9 / 550.9 / 532.6 s, ResNet-32 seed 1 = 1,261.7 s, smokes 0.037 INFERRED)
      and seeded as 'hist' events in step0/gpu_spend.jsonl, which the gate counts. STEP 0 BOUNDARY: one convention, WH counted in Q4:
      Step 0 = spent 0.85 + planned 2.42 = 3.27 of the 4-5 allowance (with WH it would be 3.32; WH is charged once, in Q4).
      HEADROOM: H1 1.55 raw / 1.70 loaded (one outside smoke + two outside runs; the internal comparison pair is reused only if identical in
      every field, else H1 is cut, C2 5); H2 5.32 / 5.85. R + H1 = 40.28: H1 NO LONGER FITS a 40 cap after the CUDA E1 rows (v6.5);
      R + H1 + H2 = 46.13; R + 100 h package = 73.33; every enumerated row = 98.76.
      FREEZE LEDGER (v6.5, tested, freeze_selftest_2026-11-20/result.txt): FREEZE=<freeze_manifest.json> ledger_v65.py takes the approved
      option from the manifest's '_approved_option' (R, R+H1, R+H; anything else is refused; a FREEZE_SCOPES environment setting is refused),
      requires for every non-spent row of that option config_sha256, config_path, script, updates_or_views, data_order_seed and reuse; it
      recomputes the sha256 of config_path and refuses a mismatch, and refuses updates_or_views != the row's exact updates. Tested: complete
      manifest accepted; a Q4 row frozen at 15,091 vs its 15,084 refused; a wrong config digest refused; option 'ALL' refused; the environment
      override refused. NOT YET in code (C9 1): binding a tuned row's winner (candidate map frozen before the grid, winner bound by a separate
      verified record), and checking row-to-job identity at launch (the gate does this by key, C5 7).
   4. E1 = the EVAL_N option (base6s.py, base6.py): the end point is the model; fewer curve points can change it only if evaluation leaks into
      training (RNG draws, module mode, optimizer or scheduler state, CUDA state). Evidence plan:
      a. CPU harness (step0/e1_cpu.sh, e1_compare.py): CPU, base6s.train via base6sy.py arm TS, 0.5 h draw 0 seed 3, 6-pass prefix of a 60-pass
         schedule. Arms A old rule, B old rule again, C EVAL_N=60 (evaluations after passes 1-6), D = C + BROKEN_E1. Verdict rule written 01:59
         before the runs, AMENDED 20 Nov 02:4x while only run A had started and before any trace was read (tightening only): (8) D == C on
         updates 1-9 (identical before the injected draw); (9) the whole interleaved event order (u / e / p with numbers) equal for A, B and
         for C, D; (10) the comparator exits 1 unless PASS. STATUS (20 Nov 03:56): FAIL AS WRITTEN. All four runs rc 0; conditions
         (0)-(2), (4)-(9) held (counts, evaluation events [6] / 1-6, A == B, A == C, D differs at the pass-1 state hash and later, final
         scores equal, D == C on updates 1-9, event order); condition (3) failed: every 'e' line reads False. Cause (checked at the line):
         the 'e' line is written right after evaluate(), which calls m.eval(), and before the next pass's m.train() (base6s.py:215), so the
         rule demanded a state the code never has at that point; all 54 'u' lines of every run read True (no update ran in eval mode). The
         FAIL stands (step0/e1_compare.py verdict, scratch storage .../e1v64/e1_verdict.json). NEW TEST v6.5 on fresh runs, rule written and hashed
         before they started (step0/e1_compare_v65.py, e1_v65_rule_hash.txt 2a2f5ec5...): (3) becomes 'every u line True AND every e line
         False'; paths TS (A B C D), S, R (RWIRE_SEED 3), P at 1 h, G at 1 h (A C D each), final results compared incl. per-item rows; runner
         step0/e1_cpu_v65.sh (two CPU lanes; config in force read back: 0.5 h 540 planned updates = 9 per pass, 1 h 1,020 = 17 per pass).
         v6.5 RESULT (07:2x): PASS on all five paths (TS, S, R, P 1 h, G 1 h; every check true; D first differs at the pass-1 state hash on
         each path; final results incl. per-item rows identical A == C; step0/e1_verdict_v65_copy.json). Because patch_q2c.py then added
         recording-only code to base6s.train (C9 4), S and R are re-run under the UNCHANGED rule (v6.5b: step0/e1_cpu_v65b.sh,
         e1_compare_v65b.py asserts the v6.5 rule hash): v6.5b PASS on S and R (ALL_PASS, 08:2x; step0/e1_verdict_v65b_copy.json). A PASS covers each path's CPU code with the 6-pass prefix only.
      b. CPU harness on every retained path's model code (S, shuffled R at 0.5 h; P, Shared at 1 h), same rule (C9 3; CPU, 0 GPU-h).
      c. CUDA checks PER TRAINER (priced in R, 8 arms, 0.93 GPU-h expected, 1.86 sensitivity): base6s.py with arm S at 0.5 h (base6s.train and
         evaluate() serve arms S, G and P; base6s_R.py is base6s with the wiring permuted) and, at 1 h, the trainer that runs Q1's Shared arm
         as fixed in the freeze file (C9 1; if that is base6s.py arm G the check uses it, if base6.py, base6.py; same price). Arms per trainer: A old rule (one evaluation at pass 6), B = A again (measures
         the GPU run-to-run spread), C evaluation after every pass, D = C + BROKEN_E1 (positive control). Prefix 6 passes (54 / 102 updates).
         Acceptance, written now: rc 0 for all; event order as (9); all modules in training mode at every u / e line; numpy, torch-CPU and
         CUDA RNG state hashes equal for A, B, C at every pass; per-update loss |A - C| <= max over updates of |A - B| x 1.5 + 1e-6 (declared
         tolerance, from the GPU spread of the SAME prefix); D must exceed that tolerance on some update after the 9th AND show a changed RNG
         hash (else the check is blind and FAILS). Why C uses an evaluation after every pass, not the production cadence: production EVAL_N=30
         first evaluates at pass ~61 (0.5 h) / ~33 (1 h), so a short prefix sees no evaluation; every-pass evaluation exercises the same
         evaluation code more often (a stronger leak test); the CADENCE logic itself (which passes evaluate) is pure arithmetic and
         WILL BE checked on the CPU against the full 1,836- / 972-pass schedules (C9 3, TO DO: the evaluation positions printed by the trainer's
         own schedule code for EVAL_N=30 are 30 distinct passes ending at the last pass, and a CPU trace with STOP_AFTER just past the first
         production evaluation equals the same trace with evaluation disabled).
      d. PER-PATH CUDA alternative (scope F, priced: 16 arms 1.85 GPU-h; R would be 39.60): offered if the reviewer holds that the trainer
         loop is not the only CUDA-specific channel. Reason for per-trainer in R: S, R and P share base6s.train and evaluate(); R differs from
         S only in index arrays (no new modules); model-specific channels (dropout layers, head modes) are deterministic and covered by the
         CPU harness per path (b).
      STOP RULE: a path without both a CPU PASS and its trainer's CUDA PASS keeps the old rule at the old-rule price (+0.324 per 0.5 h run,
      +0.155 per 1 h run); R would then exceed 40 and the scope goes back to the author by review note before that path launches.
   5. Triggers (B 10), neither waived; both responses need the author's acceptance in the review note. (a) 'full-data CIFAR > 20 min/run' FIRED on the
      ResNet-32 comparator (21.0 min; Shared 9-10 min): ResNet is calibration, seed 1 done, seeds 2-3 dropped; P / G_mix are timed in their
      first run and over 20 min the trigger fires again. (b) '100 h speech > 2.5 GPU-h/run' FIRED (3.88): B 12 cut, 100 h deferred. Both
      triggers are applied again to the actual retained runs.
   6. Never-drop list kept: split integrity (C2 0), three-seed baselines (speech triplet once, on the frozen geometry and split, AFTER per-item
      count logging, C9 5), ResNet seed 1 (done), uncertainty simulations (done: sec2_sims, sec2_abs_sims, sec2_abs_harm_sims and, v6.5,
      sec2_abs_unequal_sims). If the final freeze changes the CIFAR geometry, the CIFAR baseline is not reusable and its rerun is repriced.
   7. CAP. step0/gpu_gate2.py (v6.5, replaces gpu_gate.py; selftest step0/gpu_gate2_selftest_2026-11-20.json, 16 checks ALL_OK):
      the cap comes from approved.json (written at the author's approval: cap, option, ledger path and sha256, reserve fraction; absent = every
      job refused); commitments come from the approved ledger, never from the caller (the check takes only the job key):
      need(J) = all charged spend (incl. the 0.85 h history) + J's SENSITIVITY allowance (its hard timeout) + 1.1 x the expected price of
      every approved row not yet charged, J excluded; need <= cap. Refused also: ledger changed after approval; J not an approved row or out
      of scope; J already charged; any begun-and-uncharged job (one GPU job at a time). 'run J -- cmd' = check, begin record (fsync), run
      under `timeout <allowance>`, charge exactly once; a non-zero exit or timeout is charged as INTERRUPTED and is never scoring-eligible;
      'recover' charges every begun-and-uncharged job at min(allowance, elapsed) as interrupted. Tested: no approval, no spend record, at
      the cap exactly (allowed), 0.01 h beyond with history counted (refused), out-of-scope, unknown job, no caller override possible, a
      second job while one is open, crash recovery (charged once, at most its allowance, not eligible), double charge, rerun of a charged
      job, an overrun propagating to the next job's refusal, a clean run (eligible), a failing run (interrupted), a ledger edited after
      approval. NOT YET (C9 6): wiring into the queue runner's take() under GPU.lockdir with process-group kill; the launch-path audit
      (every Tranche 1 runner launches GPU work only via 'gpu_gate2.py run'; test: grep of every runner + a runner without it refused at
      review). Cut order for UNSTARTED work when the gate refuses: (1) headroom never starts; (2) Q8 (c), (b), (a); (3) pause, and the author
      decides by review note.
   8. ORDER (one sequence; no step starts before the one above it is done):
      a. CPU now: round-5 tools (this revision), E1 CPU verdicts, C2 0 d checks (2)-(5) and the hard-panel record, C9 CPU items.
      b. Round 6 (the author chose one more round, T23): if clean, the review note (C6); if not, the open items go to him as decisions.
      c. CALIBRATION FREEZE: the calibration rows (8 CUDA E1 arms, the speech baseline triplet, the WH smoke, the Q8 (c) timing smoke) are
         frozen first (config, script, exact schedule) and approved in the review note; the HARD-PANEL DECISION is recorded at this point, once,
         irreversibly (C2 0 d), before the first GPU job of any kind.
      d. Calibration GPU jobs through the gate. Then the ledger is repriced from their measured times.
      e. RESEARCH FREEZE per comparison, before its first run: its rows frozen (C9 1) and its protocol REGISTERED (C9 2 REGISTER); a comparison
         without both does not launch.
      f. Launch, the author's ideas first: Q2 (c), Q4, Q8 a-c, Q1 CIFAR, Q1 speech; headroom last (and only inside the cap). The FIRST full job of an
         untimed class is its timing job: it runs under its sensitivity allowance; after it, STOP before the next job of that class,
         reconcile the actual charge, reprice the class's unfinished rows; if R still fits the cap, continue; if not, or if a recipe, scope,
         rule or cap would change, stop and ask by review note.
      g. Test pass (target 26 Nov; R = 38.58 GPU-h = about 2.4-3.2 allocation days at 12-16 GPU-h a day, not a guarantee): a comparison is
         scored only if all its runs finished uninterrupted (gate eligibility), its slots were BOUND and its panels RELEASED under C9 2, and
         its protocol was registered before its first run; an unfinished one stays open, unscored.

C6. APPROVALS. Tranche 1 needs the author's explicit approval (B 10) by a review note. The approval BINDS: one option (scope), the cap, the sha256
   of the approved ledger json (the job map), the calibration rows of C5 8 c, the two trigger responses (C5 5), the cut order (C5 7), the
   repricing rule (C5 8 f) and the amendment rule (any change of recipe, scope, rule or cap = stop + new review note); approved.json is written from
   that review note and nothing else. Options from ledger_v65.json (10 % on new work):
   1. R, 38.58 GPU-h, cap 40 (recommended). Buys: Q2 (c) 0.5 h, Q4 5.4 h primary family, Q8 a-c screens, Q1 speech 1 h and CIFAR 10 % / 100 %
      co-primaries, the Step 0 speech baseline, CUDA E1 per trainer, the WH and Q8 (c) smokes. Excludes H1 (R + H1 = 40.28 > 40), Q2 (c) 1 h,
      all 100 h speech, speech 5.4 h Q1, speech G_mix, the CIFAR inferential class-alignment contrast, Q4 1 h, Q8 d, Q2 (a)/(b), Q3, Q5, Q6,
      WT, Tranche 3.
   2. R + H1 = 40.28 (cap 41) or R + H1 + H2 = 46.13 (cap 47): adds the outside reference, and Q2 (c) at 1 h.
   3. R + the 100 h package = 73.33 (cap 74): only after the package's own reviewed freeze (C2 1); not executable now.
   4. INVENTORY ESTIMATE, not a purchase option: every enumerated row = 98.76. It includes deferred rows (Q4 1 h, speech G_mix, Q8 d, the
      per-path E1 alternative) that have no path-specific protocol or launch checks yet; buying any of them needs its own design, freeze and
      review.
   AUTHORIZATION BOUNDARY. CPU code and runs spend no GPU, are reversible (backups of every edited script) and are done under his standing rule
   (24 Oct: reversible and free = the agent decides); they ADOPT nothing of Part C while Part B stands, except the two protective defaults of
   C2 0 e (allowlist, sealed directory), which are recorded here as live default changes affecting Part B, with backups, and which the author can
   undo; all other v6.5 instrumentation (TRACE, STOP_AFTER, BROKEN_E1, EVAL_N, RWIRE_SEED) is off by default and changes nothing in a Part B
   run unless set. NO GPU job runs before the review note is answered, the calibration rows included. A review answer to this round is not a waiver
   of any executable gate in C9. Major-change sign-off (Part A): the review is the second verifier, the agent the author.

C7. ROUND-2 ITEMS: answered in v6.2. C8. ROUND-3 ITEMS: answered in v6.3. C10. ROUND-4 ITEMS: answered in v6.4 (files kept:
   step0/partC_v62.md, partC_v63.md, partC_v64.md). Items round 5 reopened are in C11.

C9. LAUNCH CHECKLIST (each item: what, acceptance test, status, which GPU paths wait for it). All CPU unless marked.
   1. FREEZE FILE: manifests (manifests_v65.json), the hard-panel record (C2 0 d), every row of the approved option with config path and
      sha256, script, exact updates or CIFAR views, data-order seed, reuse source; for tuned rows a candidate map frozen before the grid and a
      winner record bound after it (each winner = one of the frozen candidates, chosen by the pre-written rule from validation only);
      W / WH construction-input audit including utts2()'s trk31_fit.npy producer. Tests: ledger_v65 FREEZE refusals (DONE, C5 3); winner
      binding refuses a non-candidate and a changed rule: v6.5 DONE,
      step0/winner_bind.py (freeze = candidate map + seeds + rule file sha256, written once before the grid; bind = run records only, real
      record keys PER / SMOKE / stop_after, whole schedule required, validation sets only, rule hash unchanged, returned index must be a
      frozen candidate, written once; check_launch = the row's job config must equal the bound winner): selftest 15 checks ALL_OK
      (winner_bind_selftest_2026-11-20.json: second freeze, missing run, non-candidate record, sealed set scored, exposed set scored,
      prefix record, short log, wrong seed, duplicate, changed rule, rule returning a non-candidate, second bind, launch of a non-winner all
      refused; known answer picks the right candidate; launch of the winner accepted). Blocks: each row's own job (calibration rows first, C5 8 c).
   2. SEAL, three stages: (i) REGISTER before a comparison's first run: the analysis entry point and its protocol (panels, item lists with
      speakers and reference counts, slots, weights), hashed into the freeze file; BLOCKS THE FIRST RUN of that comparison. (ii) BIND after
      training: each slot's checkpoint and run record hashes, verified complete and gate-eligible. (iii) RELEASE once per panel: one
      controlled reader scores every bound slot and writes per-utterance rows with an atomic access record holding the panel's item
      digest; BLOCKS PANEL ACCESS. Before a panel's first release, every eligible comparison that will use it is registered. Custodian reads
      (hashing bytes, reading ids and lengths for manifests, as manifests_v65.py does for cleanT) are not model scoring and are logged.
      DONE for Q2 (c)'s decision step: code/small_net/sec2_q2c_entry.py (the only Q2 (c) verdict function: takes the registration record and
      the rows file, nothing else; protocol digest, release digests, exact panel / slot / item coverage, no duplicates, speaker and reference
      counts equal to the protocol, seeds exactly 0-4, weights from the protocol) with 12 refusal and known-answer tests ALL_OK
      (results/sec2_q2c_entry_test.json: duplicates, an item deleted from all ten cells, one missing cell item, an extra panel, a speaker swap,
      a changed reference count, an extra seed, a negative count, a protocol edited after registration, a four-seed protocol, an unreleased
      panel; known answer equal to the direct estimator). v6.5 BUILT: step0/reader_v65.py (REGISTER refuses a panel already accessed; BIND = exactly
      arms x seeds, complete runs, gpu_gate2 eligibility or protocol-listed legacy slots, written once; RELEASE = O_EXCL access record
      BEFORE any read, every bound slot scored in a FRESH process with the run's recipe environment set before import and asserted (RIGHT,
      NORM, budget, frozen-tree / shuffle string), the trainer's own evaluate(items=True), coverage / speaker / reference counts checked
      against the protocol, rows and release record written once; a crash leaves the panel COMPROMISED, never retried; FEAT / WARP / fixed
      NORM recipes refused). Selftest 20 checks ALL_OK (step0/reader_v65_selftest_2026-11-20.json: no registration, no bind, missing slot,
      ineligible job, incomplete run, legacy only where listed, bind twice, changed checkpoint, compromised after refusal, two concurrent
      releases -> one wins, rows known answer, release digest = sec2_q2c_entry digest, second release, registration after access, crashed
      child, no retry after crash, missing item, protocol edited, WARP, FEAT). REAL-checkpoint parity (step0/reader_parity_2026-11-20.jsonl,
      cleanS 300 utts, reader vs the trainer's saved GPU score): S b6ii_S_s0 29.56 vs 29.56; shuffled R b6ii_R_s0 29.44 vs 29.43 (rebuilt in
      a fresh process with RWIRE_SEED from the record, frozen-tree string asserted); P b6ii_P_s0 30.74 vs 30.75 (a few CPU / GPU argmax
      flips). FINDING: the same S checkpoint scored without the run's RIGHT=12 gives 79.04 with no error; score_locked.load_model sets the
      environment after a possible earlier import, so it must not score several recipes in one process (the reader replaces it). Note: the
      wiring mask is a saved buffer (mask differs S vs R in the checkpoints), so scoring R does not depend on import-time wiring. TO DO:
      parity for a Q1 arm and a Q4 arm (WARP recipe needs the panel warp adapter; plain base6.py geometry not rebuilt by the reader yet) (reader counts = trainer validation counts of the same
      checkpoint, incl. the shuffle reconstruction in a fresh process with recipe environment set before import); the CIFAR seal; the H1
      adapter if H1 is bought.
   3. E1 (C5 4): CPU PASS per path (v6.4 TS FAIL as written, C5 4 a; v6.5 PASS on TS, S, R, P 1 h, G 1 h; v6.5b S, R
      re-run on the final base6s code PASS; base6.py, if the
      freeze file picks it for Q1 Shared, has no trace instrumentation yet: TO DO in that case); cadence-position check on the full schedules: v6.5 DONE (step0/cadence/cadence_check.py evaluates each trainer's own trigger expression, read verbatim from base6s.py / base6.py, on 1,836 / 972 / 60 / 90 / 108 passes: EVAL_N=30 gives exactly 30 events, last pass included, gaps within 1 pass of epochs / 30; old rule 306 / 162 / 10 / 15 / 18 events; ALL_OK, cadence_result.json); then the CUDA checks per
      trainer (GPU, calibration rows, after the review note). Blocks: every 0.5 h / 1 h EVAL_N job on that path.
   4. Q2 (c) runner. v6.5 BUILT: step0/q2c/patch_q2c.py (APPLIED 07:3x after the E1 runs, backups *.bak_preQ2C_2026-11-20; tested
      first on a copy): every base6s run records plan_sha256 (speed choices + batch order of all passes), targets_sha256, train_ix_sha256,
      mask_sha256 and RWIRE_SEED; with Q2C_PAIR=1 the trainer asserts Q2C_ARM matches the wiring in force and RWIRE_SEED == seed. Smoke on
      the copy (CPU, SMOKE 20, 1 pass): S and R at seed 1 gave equal plan / targets / index hashes and different mask hashes; R with
      RWIRE_SEED 2 at seed 1 and base6s_R run as Q2C_ARM=S both stopped at the assertion. base6s_R docstring: the old 'shape, not the
      phonetics' reading replaced by the v6.3 limit. step0/q2c/runner_q2c.py: jobs() = ten fresh processes (refuses other than seeds 0-4)
      and pair_check(), required before BIND, refusing incomplete or STOP_AFTER runs, wrong seed, a shuffle in S, an R sigma that is not
      its seed's derangement (recomputed; matches the real 18 Nov R record at seed 0), any difference in plan / targets / indices / steps /
      bytes / budget / recipe, equal masks, missing evidence: selftest 15 checks ALL_OK (runner_q2c_selftest_2026-11-20.json). TO DO:
      launch only through the gate / queue (C9 6). Original item: two fresh processes on the locked budget; asserts RWIRE_SEED == seed; identical paired item / batch order and update
      targets (trace hashes); records the realized mask and permutation; refuses a deliberately mismatched pair; the wrapper docstring's old
      'shape, not phonetics' conclusion removed. Blocks: Q2 (c).
   5. Per-utterance count logging in both trainers' evaluate() (Part A 6.8): utterance id, speaker, edits, reference length, hypothesis length,
      arm, seed, set; UNROUNDED sums. v6.5 DONE (applied 03:5x after the v6.4 E1 runs ended, backups
      *.bak_prePeritem_2026-11-20; patch step0/peritem/patch_peritem.py): final scoring in base6.py and base6s.py also writes e, t, h
      (unrounded integers), n_items and items = [index, e, t, h] per utterance (index = position in the set's cache arrays; utterance ids
      come from that file's 'files' list, bound by its sha256 in manifests_v65; arm, seed, set = the run record's own keys); item sums are
      asserted equal to the totals; curve evaluations unchanged. Tests: a temp copy on b6ii_S_s0 (cleanS 29.56, libriA 46.70, both equal to the
      saved json) and base6.evaluate on b6ii_P_s0 (30.74 vs 30.75; libriA equal); INDEPENDENT per-item known answer (step0/peritem/
      indep_check.py: the reader's rows from the trainer's evaluate vs score_locked.score_items, a separate loop with its own set reader) on
      S, R and P checkpoints, cleanS: 300 / 300 items identical for each (indep_*_cleanS.json). Missing / duplicate / swapped item refusals
      live in the consumers (sec2_q2c_entry, reader_v65; tested). Original wording of the test: an independent item-level known answer on both trainers (a fixed checkpoint, recomputed per item
      by a separate script), refusals for a missing, a duplicate and a swapped item, totals reconciled to the unrounded sums; a verified
      ordered item map for libriA (its ids are not in the cache: rebuilt from its source list, or libriA stays speaker-level and is not a
      scored panel). Code waits until the CPU E1 harness finishes (it imports these files). Blocks: the speech baseline and every speech job.
   6. ENFORCING GPU GATE: gpu_gate2.py DONE (C5 7). v6.5 queue integration DONE: gpu_gate2.run refuses unless the caller holds the shared
      GPU.lockdir (owner file's first token == GATE_OWNER, exported by take()), runs the command in its OWN process group, on the allowance
      timeout sends TERM to the whole group then KILL after 60 s, treats a surviving group member as a fault, charges exactly once
      (backup gpu_gate2.py.bak_preQueue_2026-11-20; selftest 21 checks ALL_OK, gpu_gate2_selftest_queue_2026-11-20.json, incl. no lock,
      another owner's lock, no owner, refusals charge nothing, timeout kills a grandchild and charges interrupted). step0/gpu_runner_lib.sh
      = the only launch path (take / rel copied from the launch protocol, gjob <row> <cmd> = gate run); smoke: take exported the owner,
      gjob refused on the real (unheld) lock, rel cleaned up (gpu_runner_lib_smoke_2026-11-20.txt). Launch-path audit step0/audit_launch.py:
      a runner passes only if it sources the library, defines no take / rel / gjob, starts every GPU python (venv path or any variable
      derived from it, transitively) on a gjob line, no nohup / setsid / eval / background GPU launch, no direct gate call, GPU 0 only;
      selftest 9 cases ALL_OK (audit_launch_selftest_2026-11-20.json); the old small_net/run_base.sh FAILS it, as it must. Blocks: every
      GPU job (every Tranche 1 runner is audited before launch).
   7. 100 h package freeze (C2 1): only if option 3 is approved.

C11. ROUND-5 ITEMS AND WHERE THEY ARE ANSWERED (claims checked at source 20 Nov 02:4x: FREEZE_SCOPES env, one 15,091 for every Q4 seed,
   missing spend = 0 h, comparator without fail exit and without the D-prefix check: all right)
   1. C2.0 (BLOCKING): REGISTER blocks the first run, BIND / RELEASE block access, all comparisons on a panel registered before its first
      release (C9 2); manifests_v65 asserts (explicit required-input list, missing file = failure; cleanS vs exposed; clean test vs libriB;
      libriA exactly 4 speakers; libriB inside exposed dev-other as the stated invariant; duplicate ids) C2 0 a; custodian reads named C9 2.
   2. C2.4 (BLOCKING): Q8 unit 0.598257 / pair 0.997095 GPU-h; the (c) timing job named (a GPU smoke row); GPU logits charged once, inside
      the (c) allowance; (c) is an allowance, its own updates set at its freeze C2 4.
   3. C4 (BLOCKING): sec2_q2c_entry.py with registered protocol and release digests, exact coverage, no duplicates, no caller overrides,
      tested end to end (C9 2); Q2 (c) has NO validation gate (C4 1); unequal panels 40 + 10 / 15 / 20 / 30 / 39, equal and 2x hard-panel
      spread: coverage 0.98125-1.0 >= 0.97 band, PASS (results/sec2_abs_unequal_sims.json, band written before the run) C4 1.
   4. C5.1 (BLOCKING): audited eligibility (duration, updates, completed source log, known class) in code; exact per-seed updates; half arms
      at 30 curve points as the explicit recipe, priced so C5 1.
   5. C5.3 (BLOCKING): freeze validation from the approved option, digest and update checks, no environment scopes, tested C5 3; winner
      binding C9 1.
   6. C5.4 (BLOCKING): CUDA checks per trainer with B spread, declared tolerance, RNG and mode checks and a positive control, priced in R;
      per-path alternative priced; cadence logic checked on CPU; comparator amended (D prefix, event order, fail exit) C5 4.
   7. C5.7 (BLOCKING): gpu_gate2.py with approval-derived cap, ledger-derived commitments, history counted, single charge, crash recovery,
      tested at and beyond the cap C5 7; queue integration and launch-path audit C9 6.
   8. C5.8 (BLOCKING): one sequence: calibration freeze and hard-panel decision first, calibration jobs, reprice, per-comparison freeze and
      registration, launch with stop-and-reconcile after each class's timing job C5 8.
   9. C6 (BLOCKING): option 4 relabelled an inventory estimate; approval binds the ledger digest and the listed items; live default changes
      affecting Part B recorded C6.
   10. C2.1, C2.2, C2.3, C2.5, C5.6 (CHANGE): C2.1 freeze publications listed in C9 1; C2.2 runner items C9 4; C2.3 exact half integers in the
      rows and the WH smoke frozen as calibration C5 8 c; C2.5 H1 reuse-or-cut unchanged, H1 now outside R's cap; C5.6 item-level test C9 5.

# Part B. the author's v5.4, unchanged (only the 'Do not read yet' line removed)

# What makes a small network good?

**Executable Living Research Programme — v5.4 | 19 November 2026**

*Full multi-tranche plan with predeclared falsification triggers and protocol-amendment rules. Incorporates reviewer, reviewer, and technical peer-review adjudications (v4.4 → v5.4). Assumes cited external facts and prior measurements are accurate; local run and model facts still require `[confirm: the agent]` checks.*

**Authorization status:** Step 0 only is eligible under the existing midnight fallback. If no answer arrives by 20 November 00:00, execute Step 0 alone, subject to the existing authorization, with a target cost of 4–5 GPU-hours. Do not start Phase 1, a 100-hour run, an optional run, or a later tranche without explicit approval. This document is a proposal and does not itself authorize additional spending.

**Working funding structure:** This document defines the **entire Q1–Q7 research programme** (including downstream scaling and training-rule extensions) so every question is scientifically coherent and falsifiable from day one, while enforcing strict tranche-by-tranche spending caps:

* **Tranche 1 (current funding decision):** Hard, all-inclusive **40 GPU-hour ceiling** covering Step 0 + Q1 + Q3, including full-data tuning proxies, reruns, conditional third seeds, and timing overruns. A 35-hour cap would fund a more provisional screen, not the same confirmation promises.
* **Tranche 2 (separate approval):** Q2, Q4, Q5, and Q6 (~10–12 GPU-hours core estimate, plus CPU work, calibration, buffer, and conditional controls). Total Q1–Q6 programme is roughly 45–48 GPU-hours core (~52–55 with a 15% buffer), to be replaced by measured Step 0 costs.
* **Tranche 3 (conditional on Tranche 1–2 outcomes):** Q7 scaling (~33–38 GPU-hours provisional) and equal-compute training/repair proposals (OPAL; automated edit/link/grow/split).

**Core design principle:** A narrow result with the right control beats a broad claim from an underpowered comparison—*and* a full research plan should evolve when evidence disproves its premises. Every hypothesis, architectural assumption, and control below carries an explicit **falsification condition** and a **predeclared pivot** (Section 12). Inconclusive or negative findings on 26 November are valid outputs that redirect the programme rather than excuses to move locked goalposts.

---

## 0. Questions, hypotheses, and living-plan philosophy

We study what makes a roughly 2 MB network effective across seven connected questions:

1. **Q1 (Weight allocation):** Private per-output capacity versus a shared dense layer.
2. **Q2 (Wiring meaning):** Semantically meaningful versus shuffled wiring for editing and low-data learning.
3. **Q3 (Invariance vs. data):** Engineered nuisance handling versus scaling raw training data.
4. **Q4 (Informed initialization):** Human-written versus data-driven and randomly initialized weights.
5. **Q5 (Adaptation shape):** Adapting to new speakers or corruptions without catastrophic forgetting.
6. **Q6 (Persistence of specialization):** Whether locally trained specialized parts retain their roles under joint optimization.
7. **Q7 (Scaling):** Whether a validated 2 MB structural advantage grows, persists, or shrinks from 0.25 MB to 4 MB.

Speech recognition over 39 sounds is the primary domain; CIFAR-100 is an independent image task. A speech-only result is never described as cross-domain. Historical measurements are motivations, not new evidence or part of the new multiplicity family; `[confirm: the agent]` flags require verification of local model files, metrics, and run logs.

**How this plan handles disproven premises:** The plan distinguishes **locked confirmatory runs** (never edited mid-flight or after test access) from **living programme design** (downstream hypotheses, controls, and architectures that update cleanly through versioned amendments whenever an upstream measurement disproves an assumption—see Section 12).

---

## 1. Common experimental contract

* **Size and accounting:** 2 MB means 2,000,000 bytes of learned fp32 weights (about 500,000 parameters). Log buffers, BatchNorm state, adaptation parameters, replay storage, and total deployment footprint separately. Match trained parameter budgets across arms. Also report multiply-adds, training/backprop FLOPs, CPU batch-one latency on one fixed machine, GPU throughput, and test-time inference cost. Never call equal weights equal compute.
* **Datasets and splits:** Freeze the precise speech corpus, preprocessing, vocabulary, and metric in Step 0; state whether the old 39-class result is frame error or phone error rate, the unit of aggregation, and how speakers and utterances are weighted. Use speaker-disjoint train/validation/test. Record hours, speakers, and utterances for 1, 5.4, and 100 hours. Prefer nested data subsets from the same speaker pool; otherwise label results “data plus corpus changed” rather than claiming an isolated data effect. CIFAR-100 uses a single stratified 45,000/5,000 split of the official training images, with nested fixed 10% and 30% subsets; the official 10,000 test images are sealed. Every arm within a comparison uses the same examples and split fingerprints.
* **Validation and test:** All tuning, early stopping, arm selection, edit-strength selection, and resource gates happen without test results. A validation estimate after model or hyperparameter selection is a screen, not confirmation. Lock a question's models, analysis code, primary family, and selection procedure before its single test evaluation. No modification, rerun for selection, or new hypothesis after test access.
* **Shared-test dependency rule:** Q5 reuses Q1's Private models and CIFAR's official test images. Therefore its base identity, datasets, adaptation/selection/scoring splits, adaptation shapes, reference method, compute ceiling, harm definition, comparison, and scoring code must be preregistered and locked **before the first Q1 test access** if reuse of that holdout for later Q5 confirmation is intended. The 20 November checkpoint freezes Q1/Q3 scope and Q5's compute-ceiling formula, not a fabricated numeric value if the optional Step 0 pilot is deferred. Before Q1's first test access, complete and record a Q5 validation-only end-to-end dry run (all methods, safety calculations, both scoring pipelines, normalization mode, reproducibility, and failure handling) within an approved allowance. If that dry run cannot be completed before Q1 test access, Q1 still receives its locked one-time test pass, and Q5 is downgraded to exploratory on the reused holdout unless it moves to an entirely new untouched holdout. Any substantive post-access Q5 fix requires a fresh holdout for independent confirmation. Q1 test results must never steer Q5, and Q1 is not delayed until Q5 finishes training.
* **Training fairness:** Pair arm seeds, shuffles, data order where feasible, and evaluation items. Architecture changes share the same trunk and a prespecified trunk/head weight allocation wherever the comparison requires it. Tuning effort is equal within each pair unless explicitly labelled an asymmetric engineering screen. Record full configs, optimizer, schedules, regularization grids, all train/validation loss traces, code revision, and seed. Fix extra test-time costs (such as flip averaging and speech warping) in the configuration, not after inspecting scored results.

---

## 2. Estimands, uncertainty, and decisions

### Error and practical magnitude

For intervention $A$ versus comparator $B$, report both errors, the absolute percentage-point difference, and relative error reduction:

$$\Delta = \frac{e_B - e_A}{e_B}.$$

The $+4\%$ relative-error threshold is a prespecified resource-screen threshold fixed in both domains (not four absolute percentage points). Convert all historical speech gains to relative units using their original comparator errors, and record baseline comparator error plus the implied absolute-point threshold for every regime so class/sample scarcity differences remain visible. A point estimate $\ge 4\%$ with evidence against zero does not establish a true $\ge 4\%$ effect; only a lower interval bound above $4\%$ supports that stronger claim.

### Two distinct inference targets

1. **Checkpoint-conditional test effect:** The paired difference for these particular trained checkpoints over fresh independent data clusters, estimated by cluster-aware uncertainty.
2. **Training-procedure effect:** The expected difference across new training seeds, shuffles, and data orders as well as new data clusters.

A small p-value for (1) cannot certify (2). Report both when possible, never treat cluster sample size as the number of independently trained networks, and always publish per-seed effects and sign consistency. With two or three paired seeds, seed-population inference is intrinsically fragile; a five-seed extension improves but does not automatically solve it.

### Primary combined uncertainty for training-procedure claims

Save per-item/per-speaker 0/1 or error-rate outcomes for every seed; do not average logits across seeds (which estimates an ensemble rather than a single-model procedure). Let normalized fixed cluster weights satisfy $\sum_{c=1}^C w_c = 1$ (unequal speaker-size weights for frame/utterance-weighted speech; equal $w_c = 1/C$ for Q2/Q5 equal-speaker and CIFAR equal-source-image analyses). Define the grand aggregate errors $\hat{e}_A = \sum_{c=1}^C w_c \, k^{-1}\sum_{r=1}^k e_{A,cr}$ (likewise $\hat{e}_B$) and the weight-scaled linearised ratio influence score for cluster $c$ and paired seed $r$:

$$u_{cr} = \frac{w_c}{\hat{e}_B}\Big[(1 - \hat{\Delta})(e_{B,cr} - \hat{e}_B) - (e_{A,cr} - \hat{e}_A)\Big].$$

**Critical axis-scaling rule:** Because $w_c$ is already folded into $u_{cr}$, the linearised grand contrast **sums over clusters $c$** and **averages over seeds $r$** (so $\sum_{c=1}^C \frac{1}{k}\sum_{r=1}^k u_{cr} = 0$). Never average $u_{cr}$ over $c$. Define per-cluster seed averages $\bar{u}_{c\cdot} = \frac{1}{k}\sum_{r=1}^k u_{cr}$, per-seed cluster totals $U_{\cdot r} = \sum_{c=1}^C u_{cr}$, and the finite-cluster weight-centering factor:

$$d_w = 1 - \frac{2\sum_{c=1}^C w_c^3}{\sum_{c=1}^C w_c^2} + \sum_{c=1}^C w_c^2 \quad \Big(\text{note: } d_w = \frac{C - 1}{C} \text{ when } w_c = 1/C\Big).$$

Compute the three crossed components directly on $u_{cr}$:

$$V_{\mathrm{cluster}} = \frac{1}{d_w}\sum_{c=1}^C \bar{u}_{c\cdot}^2, \qquad V_{\mathrm{seed}} = \frac{1}{k(k - 1)}\sum_{r=1}^k U_{\cdot r}^2, \qquad V_{\mathrm{interaction}} = \frac{1}{d_w \, k(k - 1)}\sum_{c=1}^C \sum_{r=1}^k \big(u_{cr} - \bar{u}_{c\cdot} - w_c U_{\cdot r}\big)^2.$$

Freeze the nonnegative variance approximation:

$$V_{\mathrm{primary}} = \max\big\{V_{\mathrm{cluster}},\ V_{\mathrm{seed}},\ V_{\mathrm{cluster}} + V_{\mathrm{seed}} - V_{\mathrm{interaction}}\big\}.$$

*Note on mathematical properties:* Taking the maximum changes the estimator's expectation, so never call $V_{\mathrm{primary}}$ unbiased. For equal weights ($w_c = 1/C$), the untruncated expression recovers the exact crossed ANOVA grand-mean variance; for unequal speaker weights, $d_w$ corrects finite-sample centering under homoscedastic cell errors, while heterogeneous speaker-level error variance remains an approximation. Unit-test the weighted implementation against paired resampling on synthetic unequal-size clusters in Step 0, and record $C_{\mathrm{eff}} = (\sum_c w_c)^2 / \sum_c w_c^2$ strictly as a concentration diagnostic.

### Degrees-of-freedom safeguard

For training-procedure inference, cap degrees of freedom at $\nu_{\mathrm{train}} \le k - 1$ (and no larger than independent-cluster df), even when $V_{\mathrm{primary}}$ selects $V_{\mathrm{cluster}}$ because a noisy $V_{\mathrm{seed}}$ happens to be tiny. Report positive-component Welch–Satterthwaite df as a sensitivity diagnostic only; it cannot promote a two- or three-seed training-procedure claim to thousands of df. Checkpoint-conditional item uncertainty may use cluster df $C_{\mathrm{eff}} - 1$ (documented as an approximation) or a paired cluster bootstrap, explicitly labelled conditional on the frozen checkpoints. Illustrative two-sided Holm first-step thresholds under seed-capped df: at $\mathrm{df}=1$ (2 contrasts), $|t| \approx 25.45$; at $\mathrm{df}=2$, $|t| \approx 6.21$ (2 contrasts) and $|t| \approx 7.65$ (3 contrasts).

### Diagnostics, screens, equivalence, and confirmation

* **Calibration:** Use a paired cluster bootstrap (recomputing the aggregate error ratio) for checkpoint-conditional distributional checks and simple quadrature ($V_{\mathrm{cluster}} + V_{\mathrm{seed}}$) as a conservative sensitivity alternative—never competing post-hoc definitions of the primary SE. Step 0 runs preregistered CPU simulations at true $\Delta \in \{0\%, 4\%\}$, varying seed/cluster/interaction variance, speaker-weight skew, and seed-variance misspecification, plus a known-null sanity check. Record simulation counts and Monte Carlo uncertainty. If plausible simulated settings materially undercover, present affected training-procedure conclusions as exploratory, report the conservative sensitivity interval, or fund additional seeds.
* **Screens and equivalence:** Separate statistical direction (positive / negative / unsupported) from practical magnitude ($\ge 4\%$ point estimate / equivalent within $\pm 4\%$ / unresolved). A positive validation resource screen requires $\hat{\Delta} \ge +4\%$ and a multiplicity-adjusted positive test against zero; a negative screen is symmetric. Equivalence requires a full multiplicity-adjusted 90% CI inside $[-4\%, +4\%]$ (TOST at one-sided 5% per boundary, adjusted for any equivalence family); non-inferiority requires the adjusted lower bound $\ge -4\%$. All other outcomes are inconclusive, never automatically equivalent. A statistically positive $+2\%$ effect can simultaneously be practically equivalent within $\pm 4\%$: report both truths rather than forcing a contradiction.
* **Seeds and one-time test confirmation:** Initial CIFAR and $\le 5.4$-hour speech runs use three paired seeds; 100-hour speech starts at two provisional seeds and requests a third paired seed before supporting a positive resource gate, test claim, or decision-critical equivalence/negative branch (competing within the same capped reserve). Promote a training-procedure claim only when low-$k$ uncertainty is defensible under the calibration sensitivity and the corrected test succeeds; otherwise report *checkpoint-conditional finding; training-seed generalization unresolved*. For every frozen comparison report validation estimate/bucket, test effect/sign, checkpoint-conditional and combined uncertainty, adjusted p-values, and per-seed replication. A test-confirmed positive effect requires the same positive sign and a corrected positive test against zero (test $\hat{\Delta}$ need not again exceed $4\%$). Significant opposite effects contradict. Holm/Bonferroni control applies within each question's declared primary family, not study-wide across Q1–Q7; disclose study-wide multiplicity explicitly.

### Safety constraints (Q2 and Q5)

Collateral-harm caps are **0.5 absolute error points** for Q2 and **1.0 absolute error point** for Q5 on their prespecified non-target distributions.

* **Test certification (confirmatory):** *Feasible* requires the one-sided 95% test upper confidence bound on harm $\le \text{cap}$; *infeasible* means the corresponding one-sided lower bound exceeds the cap; anything between is *uncertain*.
* **Setting selection (eligibility filter, not safety certification):** For Q2, freeze selection UCB at 95% unless revised by pre-scoring amendment. For Q5 (both CIFAR 2,000-image selection and the speech non-target selection bank), fix a **one-sided 80% upper confidence bound on aggregate harm $\le 1.0$ point with `reserve = 0**` as the candidate eligibility filter, preventing a noisy development pool from mechanically eliminating safe active settings while keeping the strict 95% test UCB unchanged for final certification. Apply the identical filter to all methods/strengths, clustering by original source image or speaker, with no post-scoring relaxation or double-margining. The original 0.35/0.7 point-estimate margins are descriptive fallback diagnostics only. "No eligible active adaptation" remains a valid scientific output.

---

## 3. Q1 — Private capacity versus a shared layer (Tranche 1)

* **Motivation:** Private `P` beat equally sized Shared `S` by $+0.8$ to $+1.3$ absolute points in six of six prior 1-hour speech runs; an earlier 5.4-hour run tied. Shared's lower training loss is consistent with overfitting, not proof. Step 0 converts all historical gains to relative units.
* **Arms and geometry:** `P` = separate class-aligned hidden heads; `S` = shared dense hidden representation; `G` = structurally distinct class-alignment control (never relabelled `P` neurons). At the 20 November freeze publish layer equations, per-layer trainable parameter counts, trunk/head fraction, active hidden-unit counts, fan-in/out, head depth, and FLOPs for all three arms in both domains. Hold the convolutional trunk architecture and trainable trunk budget identical across `P`/`S`/`G` within the 2 MB limit. Compare `S` against an on-split compact ResNet to diagnose trunk starvation before interpreting any `P` win.
* **The width trap and `G_mix` control:** With one hidden layer from $D$ trunk features to $C$ outputs, equal head budgets can give `P` more active hidden units than dense `S` (for $D = 256, C = 100$, ratio $(D + C)/(D + 1) \approx 1.385$). Freeze `G_mix` as the sole primary `G` control: apply a fixed, dense orthogonal $C \times C$ mixing matrix $M$ to `P`'s $C$ scalar head outputs (`P` uses identity readout). Construct $M$ deterministically via QR decomposition of a fixed-seed Gaussian matrix with a frozen QR-sign convention; assert numerically that $M^\top M = I$, $\det(M) \ne 0$, and entries are delocalized (neither near-permutation nor small-block diagonal). Keep $M$ fixed across seeds. Each `P`/`G_mix` head emits one scalar before $M$; place learned scalar-head biases before $M$ with zero post-$M$ trainable bias or projection, and enforce identical bias/norm weight-decay exclusions across `P`, `S`, and `G_mix`. Log the extra fixed inference arithmetic, and note in interpretation that coordinate-wise optimizers (e.g., Adam) are not rotation-invariant on logits, so `P` versus `G_mix` tests this exact readout alignment including its optimization dynamics rather than proving generic privacy.
* **Data scales and training:** Speech at 1, 5.4, and 100 hours (warp+noise); CIFAR-100 at 10%, optionally 30%, and 100% (crop+flip). At low/middle data, run matched $3 \times 2$ weight-decay/dropout grids including `(0, 0)` and train three seeds on validation-selected settings. For CIFAR `P`/`S`/`G_mix`, apply activation dropout at the shared $D$-dimensional pre-head trunk tensor (not inside narrow 6–8 unit private heads), holding location and candidate rates identical across arms. On 100% CIFAR, search three weight-decay values per arm (dropout fixed at zero if 30% is omitted).
* **100-hour speech fairness (mandatory equal-cost proxy search):** Carrying 5.4-hour regularization symmetrically to 100 hours is confounded. Before launching full 100-hour training, run an equal-cost two-candidate proxy search on the actual 100-hour pool for `P`, augmented `S`, and unaugmented `S` (Q3): candidates are `(WD=0, drop=0)` and each arm's 5.4-hour winner (if identical, predeclare a distinct nearby low-regularization candidate before inspecting proxy results). Run all six arm-candidate proxies for a frozen **15% of full optimizer updates** using a **compressed complete LR trajectory** (proportionally scaled warmup and decay milestones, never truncating the full schedule at high LR). At 2.0–2.5 GPU-hours per full run, this costs ~1.8–2.25 GPU-hours before overhead. Freeze tie-break and stopping rules in advance and log full learning curves. If the proxy cannot distinguish candidates, underfits severely, or is unaffordable, downgrade 100-hour claims to a matched-carry engineering screen rather than claiming full-data architecture superiority; full-length matched tuning requires separate reapproval. The 100-hour augmented-`S` model serves as a single frozen anchor across Q1 and Q3 only when its entire config, seed, and data order match both questions identically.
* **Primary contrasts and interpretation:** Co-primary per task: `P` versus `S` at low data (1h speech / 10% CIFAR) and full data (100h speech / 100% CIFAR), Holm-adjusted within each task. `P` versus `G_mix` is a prespecified mechanism contrast, not an extra primary gate. `P` beating `S` and `G_mix` supports a class-aligned readout advantage in this architecture; `P` beating `S` while equivalent to `G_mix` suggests class alignment is unnecessary here (not proof of generic sparsity without a matched sparsity/width control). A data-limited benefit requires small-data `P` win *plus* demonstrated full-data equivalence (`P` $\equiv$ `S`), not merely a nonsignificant full-data result. Report samples per class explicitly; never equate 1h speech and 10% CIFAR scarcity.

---

## 4. Q2 — Does the meaning of the wiring help? (Tranche 2; speech first)

* **Motivation:** Meaningful phonetic heads and shuffled heads tied on overall error, while semantic consonant heads took interpretable roles.
* **Design:** Compare semantic named-part editing (`N`), semantic gradient-selected editing (`SG`), and shuffled gradient-selected editing (`RG`) across five paired semantic/shuffled seeds with independently drawn shuffled graphs. `N` vs. `RG` tests practical editability; `SG` vs. `RG` isolates editable-unit quality under identical selection rules. Freeze part counts and edit-strength grids before seeing scored utterances.
* **(a) New-speaker adaptation:** Target $\ge 10$ held-out adaptation speakers for validation and $\ge 10$ distinct sealed speakers for confirmation (excluding the previously studied named speaker). Split each speaker's utterances into enrolment/adaptation, selection, and untouched scoring; choose the target sound group from enrolment/selection only. Measure collateral harm on a separate fixed non-target speaker/sound bank (split into selection and scoring, disjoint from adaptation). Equal-weight target-speaker error is primary. If only one adequate speaker pool exists, reserve its scoring utterances for a single final pass rather than faking validation/test independence.
* **(b) Confusion repair:** Define the five largest systematic $A \leftrightarrow B$ confusions from out-of-fold training diagnostics before scoring. Score combined error across true $A$ and true $B$ tokens so collapsing predictions onto one class cannot game the metric. Match active trainable parameters, gradient FLOPs, and maximum adaptation steps across `N`, `SG`, and `RG`, permitting `SG`/`RG` to distribute that budget across multiple heads when confused classes span shuffled heads. Report head counts, label coverage, compute, and sibling harm separately. Keep sibling sounds inside the collateral-harm metric.
* **(c) Low-data learning:** Semantic versus shuffled heads at 0.5 hour (primary) and 1 hour (robustness), five seeds with independent shuffle draws.


* **Primary family and interpretation:** `N` vs. `RG` in (a), `N` vs. `RG` in (b), and semantic vs. shuffled at 0.5h in (c), using Bonferroni-adjusted 98.3% intervals. `SG` vs. `RG` is a fixed mechanism check. Wins must satisfy the 0.5-point harm cap. `N` beating `RG` *with* `SG` beating `RG` supports superior semantic units; `N` beating `RG` without `SG` beating `RG` supports easier targeting only; equal unit quality requires demonstrated `SG` $\equiv$ `RG`. Provisional budget: ~4–5 GPU-hours plus CPU editing.

---

## 5. Q3 — Engineered nuisance handling versus data (Tranche 1)

* **Motivation:** Speech warp+noise at 5.4 hours previously matched an unaugmented 100-hour run (pending corpus-comparability check).
* **Speech:** One 2 MB Shared architecture reusing Q1's 100-hour augmented-`S` anchor versus an unaugmented (no-warp/no-noise) `S` model. Matched $3 \times 2$ weight-decay/dropout grids at 5.4 hours (three seeds); equal-cost two-candidate 15% proxy search at 100 hours (two initial seeds; third seed required for confirmation). If the full-data bundle screens positively and separate funding is approved, add warp-only and noise-only (~8 GPU-hours) to complete the $2 \times 2$; otherwise components remain untested, not disproved. Evaluate standard condition first; seen-noise and unseen-noise are secondary. Warping uses zero oracle speaker information.
* **CIFAR-100:** Same frozen Shared architecture at 10% and 100% with random crop active. Four arms: (`i`) no flip; (`ii`) flip augmentation during training; (`iii`) frozen `ii` weights with two-view **final-logit averaging** at test time (never probability averaging); (`iv`) two-view **final-logit averaging** in both training and test. Freeze `iv` to identical random crops for both left/right views, $B$ unique source images ($2B$ views) per optimizer step, fixed optimizer-update count matching `ii`, and a pre-frozen choice of pooled-$2B$ BatchNorm statistics during training; at test time, evaluate `iv` in `.eval()` mode with two-view final-logit averaging identical to `iii` so `iv` vs. `iii` cleanly isolates training-time invariance. Charge `iv`'s ~2× view-level forward/backward FLOPs and `iii`'s 2× test views explicitly in the run ledger.
* **Primary contrasts and interpretation:** At 100% CIFAR, Holm-adjusted primary contrasts are `ii`, `iii`, and `iv` each versus `i`; sharper mechanism contrasts (`iii` vs. `ii`, `iv` vs. `iii`) are predeclared secondary comparisons. An image flip win over no-flip is expected benchmark corroboration, never proof that flip, warp, and noise share a causal mechanism. A "data-substitute" conclusion requires both a small-data win and demonstrated full-data equivalence, not a nonsignificant 100-hour contrast. Legacy core estimate: ~11h speech + ~3.5h images (unverified pending run ledger).

---

## 6. Q4 — Written weights versus trained weights (Tranche 2; speech first)

* **Motivation:** Seven small written parts approached trained-part accuracy with fewer numbers, while a fully written recognizer failed where context and sequential structure were absent.
* **Design:** Match architecture and 2 MB budget across `R` (random), `D` (whitened k-means informed layer-1 filters), `W` (written layer-1 filters), and `WH` (written full trunk + written heads). Construct `D`/`W`/`WH` using training inputs and documented domain knowledge only; match first-layer output RMS on a training batch (not weight norm). Run three paired seeds at 1h and 5.4h; use a jointly preregistered warmup only if `WH` requires stabilization (never add a `W`-only warmup post-hoc).
* **Schedules and primary family:** Freeze a full schedule of $T$ optimizer updates and a **compressed half schedule** of $T/2$ updates (proportionally scaled LR milestones/decay) for both `R` and `W`; raw truncation of $T$ at $T/2$ is an exploratory diagnostic only. At 5.4h, the three-member primary Holm family is:
1. `W` vs. `R` full-length (two-sided p-value; positive claim requires positive effect).
2. `WH` vs. `R` full-length (two-sided p-value).
3. Training speed via the intersection–union p-value combining two one-sided tests without internal $\alpha$-splitting:

$$p_{\mathrm{speed}} = \max\big\{p_{\mathrm{NI}}(W_{1/2}, R_{\mathrm{full}}),\ p(W_{1/2} \text{ beats } R_{1/2})\big\}.$$



Apply Holm across all three p-values; any reported NI/TOST confidence interval must match its Holm-adjusted stepwise significance level (never an unadjusted 90% CI). Report $R_{1/2}$ vs. $R_{\mathrm{full}}$ as a schedule-saturation diagnostic.


* **Mechanism controls:** `W` vs. `D` tests written construction against a data-driven start (`W` $\equiv$ `D` requires valid equivalence). Because `W` writes only layer 1 while `WH` writes upper trunk plus heads, `WH` vs. `W` is a composite comparison; isolating written heads requires adding and preregistering `WT` (`WH`'s fully written trunk + random heads) and comparing `WH` vs. `WT` (separately budgeted). Provisional budget: ~3 GPU-hours plus writing time (~1h optional warmup).

---

## 7. Q5 — Adapt without forgetting (Tranche 2; locked before Q1 test)

* **Motivation:** Speaker adaptation improves target performance while risking collateral harm; replay previously reduced harm ~4×, and error-targeted edits beat random selection.
* **Arms and normalization:** On Q1's saved three-seed Private models (5.4h speech; 100% CIFAR-100), compare four supervised adaptation shapes: `A` whole-network adaptation with replay (fixed reference); `B` plain L2 pullback to base weights; `C` fixed-rank add-on; `D` Private output parts only. Repeat `A`–`C` on Shared as a secondary architecture-dependence check. Include `0` (no adaptation) and `BN-only` as reference rows, not ranked supervised shapes.
* **Strict BatchNorm hygiene:** For `A`–`D`, set every frozen trunk BatchNorm module to `.eval()` during adaptation steps so running statistics and normalization behavior remain base-model fixed; separately freeze or explicitly list affine $\gamma/\beta$ parameters, and programmatically assert BN state before and after every update.
* **`BN-only` reference:** With all learned weights/affines frozen, overwrite running mean/variance once from the 1,000 corrupted adaptation images (default: full replacement; freeze epsilon/variance conventions pre-test; zero test images or labels used) and evaluate strictly in `.eval()` mode including batch-one latency. High clean-image harm from `BN-only` is a measured failure of the 1-point cap, not a bug.


* **Compute fairness and eligibility:** Match grid size and enforce a common ceiling on total backprop FLOPs including replay (`D` may execute more cheap head-only steps; report passes, wall-clock GPU time, extra weights, weight copies, and replay storage separately). Freeze the ceiling formula on 20 November and lock its numeric value, replay ratio (`A`), rank (`C`), and penalty family (`B`) **before first Q1 test access** if Q5 will reuse that holdout confirmatorily. Select each shape's strength on the selection split by minimizing target error among candidates passing the **one-sided 80% selection UCB $\le 1.0$ point (`reserve = 0`)**; break ties by lower compute. If no active setting is eligible, fall back to `0` (no adaptation) and report *no eligible active shape*.
* **Speech protocol:** Adapt each held-out target speaker from labelled enrolment utterances; use disjoint within-speaker selection and scoring utterances (following Q2's two-pool or single-final-pass rule). Primary target error is equal-weight target-speaker error; collateral harm is error increase on a separate fixed non-target speaker bank (split into selection and scoring). Apply the same 80% selection UCB / 95% test UCB framework.
* **Image protocol:** Partition CIFAR's 5,000 validation source images (stratified by class, disjoint by source image) into **1,000 adapt / 2,000 select / 2,000 score**. Generate five severity-3 corruptions (Gaussian noise, defocus blur, contrast, pixelate, JPEG) reproducibly with frozen libraries (use official CIFAR-100-C strictly for pixel/PSNR generator unit tests, zero model evaluation). Adapt per corruption on the 1,000 adapted images; select shape strengths on the 2,000 selection images; score on the remaining 2,000. Target error is mean corrupted error weighted equally across the five corruptions, clustered by source image. Harm is clean-image error increase on those same source images, averaged equally across the five corruption-adapted models and clustered by source image. The 1.0-point primary cap governs this equal-weight aggregate; always report per-corruption gain, per-corruption harm, and worst-corruption harm so a localized breach is never disguised as uniform safety. Prominently label the 2k score a **development screen** (because base checkpoints were selected on the full 5k validation pool); the untouched 10k official test set (evaluated by frozen validation-adapted models with zero adaptation/tuning on test) is the first independent confirmation.
* **Primary comparison and dry run:** `A` is the fixed reference. On the selection split, pick the single strongest eligible challenger among `B`/`C`/`D` by aggregate target error, freeze its identity and strength, and score that challenger once against `A` on the scoring/test split (report all four curves descriptively). A confirmed challenger win requires $\ge 4\%$ relative target-error reduction on the validation screen, a corrected positive test effect, and feasible 95% test harm $\le 1.0$ point. Cross-domain confirmation requires the *same named challenger* to beat `A` in both domains. Complete the full validation-only Q5 dry run (including infeasible-setting fallback, test-call blocking, and BN assertions) prior to Q1 test access if confirmatory reuse is intended. Provisional budget: ~1 GPU-hour or 6–10 CPU-hours.

---

## 8. Q6 — Do specialised parts retain their meaning? (Tranche 2; speech first)

* **Motivation:** Earlier locally trained heads retained their roles in four of four runs; consonant heads showed clear specialization.
* **Non-tautology prerequisite (topological ablation audit):** Q6 is informative only if, during joint optimization, the network possesses trainable cross-head routing, a shared bypass, or another genuine alternative path capable of classifying a sound group when its original head is ablated. Diagram every group's paths before freezing: if ablation specificity is hard-wired by connectivity, relabel Q6 as *continued reliance on a hard-wired branch* or amend the architecture pre-freeze (Section 12).
* **Training, Panel E/R split, and specificity:** Pretrain semantic and shuffled networks with comparable local goals (three paired seeds, independent shuffle draws). Fix an entry threshold $\tau$ above measurement noise on a separate calibration sample. Define specificity $S$ as own-group error increase under mean ablation minus other-sound error increase (mean activations estimated from training data only). Split held-out validation utterances into **Panel E** (used solely for entry qualification and pre-joint $S_{\mathrm{pre}}$ stratum/matching assignment) and disjoint **Panel R** (used for paired pre- and post-joint scoring of $S_{\mathrm{pre}}$ and $S_{\mathrm{post}}$). Never use Panel-R $S_{\mathrm{pre}}$ to determine entry or strata; report Panel-E vs. Panel-R reliability to quantify residual regression-to-the-mean attenuation. If shuffled heads rarely pass entry $\tau$, that entry gap—not selective retention—is the primary finding.
* **Joint optimization and inference:** Remove local goals and optimize jointly at a frozen standard LR to validation convergence (0.3× optional), logging parameter movement and validation-loss change. Primary conditional contrast: semantic versus shuffled Panel-R $S_{\mathrm{post}}$, stratified or matched by Panel-E $S_{\mathrm{pre}}$ over the prespecified overlapping range, aggregated to **model-level outcomes** before between-structure inference (never run unconstrained head-level ANCOVA treating correlated heads as independent across only three models per arm; if regression adjustment is used, freeze a single pooled slope on independent development data and flag the six-model limitation). Report raw change and $S_{\mathrm{post}}/S_{\mathrm{pre}}$ as descriptive sensitivities only. A local-goals-kept-on control (~2 GPU-hours) requires separate approval before recommending maintained auxiliary losses. Provisional budget: ~2–3 GPU-hours plus CPU.

---

## 9. Q7 and later directions (Tranche 3; conditional on Tranche 1–2)

* **Q7 — Scaling a validated structure:** Triggered only by a Q1 structural win or a Q2 scalable-wiring win. Compare the winning structure against a matched plain network at **0.25, 1, 2 (reused eligible pair), and 4 MB**, holding trunk-to-structured-part share fixed. Three paired seeds at $\le 2$ MB; two at 4 MB (third seed if slope inference requires it). Regress paired relative-error lead against $\log(\text{size})$, reporting error and hardware cost at every point. A slope CI excluding zero supports lead growth or decay over 0.25–4 MB only, never a universal scaling law. Provisional budget: ~33–38 GPU-hours (separate approval).
* **OPAL training-rule rewrite and automated repair (edit/link/grow/split):** Equal-compute Tranche 3 proposals. Triggered only if Tranche 1–2 results show that structural benefits (Q1/Q2) or adaptation/retention gaps (Q5/Q6) are bottlenecked by standard backprop or static graphs. Must obey the Section 1–2 contract (equal backprop/wall-clock compute ceilings, paired seeds, sealed test).

---

## 10. Execution, budgets, ledger, and hard stops

### Full programme map

| Tranche | Questions | Core Estimate | Authorization Status | Hard Entry / Unlock Gate |
| --- | --- | --- | --- | --- |
| **Step 0** | Calibration, audits, uncertainty sims | 4–5 GPU-h | Eligible under midnight fallback (20 Nov 00:00) | Existing Step 0 authorization |
| **Tranche 1** | Step 0 + Q1 + Q3 | **Hard cap: 40 GPU-h** (all-inclusive) | Pending explicit approval on 20 Nov | Step 0 baselines + verified run ledger $\le 40$h |
| **Tranche 2** | Q2, Q4, Q5, Q6 | ~10–12 GPU-h core (+ CPU/buffer) | Separate approval required | Q5 dry run pre-Q1-test for reused holdout; Q6 topology audit |
| **Tranche 3** | Q7 scaling; OPAL; automated repair | ~33–38 GPU-h (Q7) + TBD | Separate approval required | Confirmed Q1/Q2 win (Q7) or Tranche 1–2 bottleneck evidence |

### Step 0 — Night of 19 November (target 4–5 GPU-hours; zero test evaluation)

Execute strictly in order:

1. **Data, splits, and freeze prep:** Confirm speech metric/corpus/splits and historical 1h comparator errors (`[confirm: the agent]`); fingerprint CIFAR and speech splits; record exact MB/FLOPs accounting; establish provisional `P`/`S` and Q3-Shared geometry before running baselines (if the 20 Nov freeze alters geometry, rerun affected baselines rather than mixing numbers).
2. **Mandatory calibration baselines:** Three-seed 100%-CIFAR Shared baseline; at least one on-split compact ResNet comparator (CIFAR $\ge 69\%$ top-1 is an aspiration judged primarily against this same-split ResNet); three-seed 5.4h speech Shared baseline with per-cluster predictions and loss traces.
3. **Mandatory CPU audits and simulations:** Implement and unit-test the Section 2 weighted influence-score estimator ($d_w$ centering, $\nu_{\mathrm{train}} \le k-1$ cap) on synthetic unequal clusters; run coverage simulations at $\Delta \in \{0\%, 4\%\}$; complete Q1's parameter/active-width/`G_mix` orthogonality audit; draft Q6's topological ablation audit; cost the six 15%-length 100-hour tuning proxies.
4. **If GPU time remains:** Run one-seed `P`/`S` small-data WD/dropout pilots (1h speech, 10% CIFAR); add compact-ResNet seeds 2 and 3.
5. **If CPU/time remains:** Time one Q5 adaptation case, test the CIFAR corruption generator (image-only pixel/PSNR check), and measure development-only 2,000-image selection UCB widths. If deferred, Q5's numeric ceiling and dry run remain open until before first Q1 test access (never fabricated on 20 Nov).

**Explicit Step 0 drop order on overrun:** (1) ResNet seeds 2–3 (keep seed 1); (2) Q5 adaptation pilot and generator check; (3) optional small-data grid pilots (moved to Phase 1). Never drop split integrity, three-seed baselines, ResNet seed 1, or uncertainty simulations. Stop at the Step 0 allowance; never bill overruns silently to Phase 1.

### 20 November go/no-go checkpoint

Publish corpus/metric definitions, baseline errors, on-split ResNet comparison, measured per-arm timings, empirical variance components, expected interval widths, and a **line-item per-run GPU ledger** with columns: `arm`, `domain`, `data_size`, `config_hash`, `seed`, `stage`, `optimizer_updates`, `unique_images_or_views`, `expected_gpu_hours`, `reuse_source`. Reuse a Step 0 Shared baseline as a Phase 1 arm *only* when architecture, geometry, augmentation, schedule, search status, and seed match identically; otherwise treat it as sunk Step 0 calibration spend.

Freeze on 20 November: Q1 `P`/`S`/`G_mix` equations, dense orthogonal $M$, pre-$M$ biases, pre-head CIFAR dropout, and 15%-length compressed-LR 100h proxy spec; Q3 Shared, `iii`/`iv` final-logit pooling, and `iv` $B$-unique/$2B$-view BN/compute protocol; Q2/Q4/Q6 design drafts and Q5 compute-ceiling formula, 1k/2k/2k split, and 80% selection UCB (Q5 numeric ceiling and dry run freeze before first Q1 test access); Section 2 analysis code and unit tests. If full-data CIFAR exceeds 20 minutes/run, 100h speech exceeds 2.5 GPU-hours/run, or the ledger exceeds 40 GPU-hours / 26 November, trigger Section 12 cuts or pause for scope reapproval.

### Tranche 1 budget reconciliation (legacy estimates; replaced by 20 Nov run ledger)

| Item | Expected Cost (GPU-h) | Verification Status |
| --- | --- | --- |
| Step 0 | 4–5 | Measured tonight |
| Q1 speech | ~10.5 | Pending Step 0 timings |
| Q1 images (no 30%; $3 \times 2$ grids) | ~2.5 (down from ~6) | **Unverified:** must verify full `G_mix`, 3-WD search, and Step 0 reuse |
| Q3 speech (incl. shared 100h `S` anchor) | ~11 | Pending Step 0 timings |
| Q3 images | ~3.5 | **Unverified:** must include `iv` $2B$-view ~2× training compute |
| **Legacy core subtotal** | **~31.5–32.5** | **Not a validated ledger until 20 Nov** |
| Remaining capacity for proxies, 3rd seeds, reruns, overruns | **~7.5–8.5 max** | Six 15% proxies (~1.8–2.25h) + both 3rd seeds (~6h) = ~7.8–8.25h before overhead |
| **Hard all-inclusive ceiling** | **40.0** | **Never cross without explicit approval** |

*Shared-anchor seed arithmetic:* One paired 100h third seed is ~4h; if both Q1 and Q3 need a third seed and share an identical augmented-`S` anchor, the incremental cost is ~6h (one `P`, one shared `S`, one unaugmented `S`). Because proxies + both third seeds (~7.8–8.25h) already exhaust the nominal reserve before any unbudgeted `G_mix`/`iv` cost, **do not assume both third seeds fit inside 40h** until the 20 November ledger confirms `measured_spent + planned_spent + reserve ≤ 40`.

**Phase 1 cost-preservation order:** (1) Step 0 integrity and baselines; (2) equal-cost 100h tuning proxies; (3) primary contrasts; (4) decision-relevant third seeds; (5) indispensable funded mechanism controls; (6) secondary scales/sweeps (drop Q3 10% CIFAR and nonessential small-data `G` runs before primary full-data comparisons). Never sacrifice tuning fairness for a nominally significant 100h result, and never retain a mechanistic claim after dropping its required control.

### 26 November review and one-time test pass

Present frozen validation decisions first, separating checkpoint-conditional from training-procedure inference. Run the one-time test pass only for locked comparisons whose dependent protocols (including Q5's pre-test dry run, if confirmatory reuse is desired) are verified. Disclose unfunded third seeds, approximate intervals, and underpowered equivalence branches. No post-test retraining; unfinished Tranche 2–3 questions remain explicitly open rather than receiving invented outcomes.

---

## 11. Gates and reporting standard

* **Q1 cross-domain gate:** `P` beats `S` in the same named regime (low-data or full-data) in both domains under within-task Holm control. A class-aligned readout mechanism claim additionally requires width-/trunk-matched `P` beating `G_mix`; without 100h speech `G_mix`, full-data mechanisms remain unresolved.
* **Q3 cross-domain gate:** Nuisance handling screens positively at full data in both domains, explicitly disclosing that CIFAR flip vs. no-flip corroborates only the broad engineering principle, not a shared causal mechanism with speech warp+noise.
* **Q5 cross-domain gate (Tranche 2):** The same named `B`/`C`/`D` challenger beats fixed `A` in both domains with feasible 95% test harm $\le 1.0$ point. Q2/Q4/Q6 speech wins justify image replications rather than automatic cross-domain claims.

For every question report: (1) candidate network-level lesson; (2) absolute and relative effect; (3) interval, per-seed outcomes, multiplicity family, decision bucket, and calibration status; (4) selection and freeze provenance; (5) validation screen and one-time test confirmation/contradiction; (6) control-based mechanism assessment; (7) parameters, fp32 MB, MACs/FLOPs, CPU latency, GPU throughput, and storage; (8) negative, inconclusive, and untested findings; and (9) actual spend plus cost of further confirmation. Use the strict five-tier headline ladder—*observed* → *positive validation screen* → *test-confirmed* → *mechanism supported* → *replicated in both domains*—never silently promoting one tier to the next.

**Predeclared expected outcome:** Several branches—especially `P` $\equiv$ `G_mix`, full-data `P` $\equiv$ `S`, Q3 data-substitute, `W` $\equiv$ `D`, Q6 retention, and tight harm-cap certification—are likely to be underpowered or unidentifiable at $k = 2\text{–}3$ seeds. The programme succeeds by accurately mapping what is established, what is disproven, and where to pivot next.

---

## 12. Falsification triggers, predeclared pivots, and living-plan amendment rules

Because this is a full programme designed to adapt when premises fail, every major assumption has a predeclared trigger and next action. Pivots **never** alter an already-locked test comparison; they either adjust an unfrozen design before its freeze or open a cleanly versioned follow-on arm/tranche.

### Predeclared Falsification & Pivot Matrix

| Area | Falsification Trigger (Measured Evidence) | Immediate Claim Boundary | Predeclared Programme Pivot |
| --- | --- | --- | --- |
| **Step 0 Baselines** | CIFAR Shared trails on-split compact ResNet by a large margin (trunk starvation) | Do not interpret any Q1 `P` win as evidence of privacy working | Before 20 Nov freeze, shift parameter share from head back to trunk and rerun Step 0 baseline; if 2 MB still starves 100 heads, reframe CIFAR Q1 as a capacity-allocation limit and pivot Q7/Q1b to trunk-preserved heads |
| **Uncertainty Calibration** | Step 0 simulations show severe undercoverage at $k = 2\text{–}3$ under realistic variance ratios | Downgrade training-procedure inference to *exploratory*; rely on checkpoint-conditional claims | Use conservative quadrature sensitivity intervals in Tranche 1; budget targeted 5-seed replications only on the single highest-value Tranche 1 lead before scaling to Q7 |
| **Q1 Full-Data Proxy** | 15% compressed proxy fails to separate candidates or curves remain clearly underfit | Label 100h comparison a *provisional matched-carry engineering screen* | Do not claim full-data superiority; log exact failure mode and propose a focused, full-length matched tuning follow-up (or reallocate Tranche 2/3 spend to the cleaner low-data regime) |
| **Q1 Mechanism (`G_mix`)** | `P` beats `S`, but `P` $\equiv$ `G_mix` (or `G_mix` beats `P`) | Reject class-aligned readout as the sole driver in this geometry | Pivot follow-on mechanism study to isolate **hidden width / block-diagonal input sparsity / optimizer gradient scaling** using a separately preregistered width-matched sparse control |
| **Q1 Scale Decay** | `P` beats `S` at 1h/10%, but `P` $\equiv$ `S` (or `S` wins) at 100h/100% | Classify private capacity as a *data-limited* or regularization-like benefit | Halt Q7 scaling of `P` at full data; pivot Q1 follow-up to test whether explicit regularization on `S` closes the small-data gap and retain `P` only for low-data/adaptation (Q5) |
| **Q3 Persistence** | Speech warp+noise fails to beat unaugmented `S` at 100h (or achieves equivalence) | Drop "persistent full-data gain" claim; test data-substitute equivalence | Do not fund the 8h $2 \times 2$ warp/noise ablation; pivot Q3 conclusions to regime-dependent nuisance handling (strong at 5.4h, saturating at 100h) |
| **Q2 Wiring Meaning** | `N` beats `RG`, but `SG` fails to beat `RG` (or all three tie) | Reject intrinsic unit superiority; claim easier targeting only (or null) | Do not scale semantic wiring into Q7; pivot practical editing work toward gradient-selected or automated localization rather than manual phonetic partitioning |
| **Q4 Written Weights** | `W` matches `D` (k-means) or fails speed test $p_{\mathrm{speed}}$ | Attribute gain to generic informed initialization, not human-written superiority | Retire manual filter writing; adopt data-driven informed starts (`D`) and test whether `WH` vs. `WT` still adds value in upper layers/heads |
| **Q5 Safety / Adaptation** | Selection UCB eliminates all active shapes `A`–`D`, or `BN-only` breaches 1.0-point clean harm | Report *no eligible active shape* (valid finding); never relax bounds post-hoc | Diagnose whether bottleneck is variance (selection split too small) or genuine forgetting; if genuine forgetting, pivot Tranche 3 toward lower-rank / selective-layer / replay-augmented shapes or relaxed-cap operating curves (Pareto frontier) on a fresh holdout |
| **Q6 Topology / Entry** | Topological audit shows hard-wired specificity (no alternative path), or shuffled heads fail entry $\tau$ | Relabel as hard-wired reliance or entry-rate gap; prohibit learned-retention claims | Before Q6 freeze, introduce a trainable cross-head bypass/routing layer (logging changed comparability with Q1/Q2); if shuffled entry stays near zero, reframe Q6 around *why semantic local goals induce specialization more reliably* |
| **Q7 Gate** | Neither Q1 nor Q2 produces a confirmed scalable structural win | Do not launch the 33–38h Q7 scaling sweep | Redirect Tranche 3 budget to whichever positive mechanism emerged (e.g., Q3 invariance scaling, Q5 safe adaptation, or OPAL equal-compute training rules) |

### Protocol Amendment Rules (How to Change the Plan Cleanly)

1. **Immutable boundary:** Once a question's freeze timestamp passes (20 Nov for Q1/Q3; pre-Q1-test for Q5 shared-holdout rules; pre-run freeze for Q2/Q4/Q6/Q7), its primary estimands, primary family, selection rules, and analysis code cannot be altered for that confirmatory pass.
2. **Pre-freeze updates:** Findings from Step 0 or an upstream tranche may freely reshape any *unfrozen* question via a dated version increment (e.g., v5.4 → v5.5) that states: (a) what measured evidence falsified the prior assumption, (b) the exact change to geometry, control, or hypothesis, and (c) the updated run ledger.
3. **Post-falsification redesigns:** If a locked confirmatory run disproves a core premise and suggests a better architecture or hypothesis, close and publish the original locked outcome first, then open the revised design as a **newly named, separately preregistered follow-on experiment** (e.g., `Q1b`, `Q5b`) using an untouched holdout or fresh evaluation split if any test data were previously exposed.

---

## Appendix — Consolidated Version Lineage (v4.4 → v5.4)

* **v4.4 → v5.0 (reviewer adjudication):** Raised Tranche 1 hard cap from 35h to 40h all-inclusive with triggers inside the cap and shared-anchor seed savings; added strict pre-Q1-test lock for Q5 shared-holdout reuse; made Q1 low- and full-data co-primary with relative-error conversion; added `BN-only`, `0`, and fixed-`A` structure to Q5; distinguished clean/non-target harm from target error; separated Step 0 drafting from 20 Nov freezing with an explicit drop order; disclosed lack of study-wide Q1–Q7 multiplicity control.
* **v5.0 → v5.1 (reviewer adjudication):** Replaced symmetric 5.4h carry-forward with a mandatory equal-cost 100h proxy search; formally separated checkpoint-conditional from training-procedure estimands and exposed $k = 2\text{–}3$ t/Satterthwaite fragility; added Q6 topological non-tautology audit and baseline-adjusted $S_{\mathrm{post}}$; added Q1 width-trap audit and `G_mix` control; flagged Q5 validation-split overlap with base selection; enforced `.eval()` BN hygiene during `A`–`D` adaptation; separated magnitude screens from statistical confirmation; identified `WH` vs. `W` as composite (requiring `WT` to isolate heads); matched Q2 confusion-repair active parameter/FLOP budgets; required pre-Q1-test Q5 dry run.
* **v5.1 → v5.2 (Pre-freeze edge cases):** Eliminated double-margining on Q5 selection (`reserve = 0` default); mandated compressed complete warmup-and-decay LR trajectories for 100h proxies; locked `G_mix` pre-$M$ scalar heads and pre-$M$ biases; replaced unconstrained Q6 head-level ANCOVA with model-level baseline strata/matching.
* **v5.2 → v5.3 (Technical audit):** Rebalanced CIFAR Q5 validation split to 1k/2k/2k and separated 80% selection eligibility UCB from 95% test certification; decoupled Q5's numeric ceiling freeze from 20 Nov (binding it to pre-Q1-test); capped training-procedure df at $\nu_{\mathrm{train}} \le k - 1$; enforced deterministic dense orthogonal $M$ and pre-head CIFAR dropout in Q1; standardized Q3 `iii`/`iv` final-logit pooling and `iv` $B$-unique/$2B$-view accounting; separated Q6 Panel E (entry/strata) from Panel R (scoring); aligned Q4 IUT one-sided and two-sided Holm thresholds; required a per-run GPU ledger before Phase 1.
* **v5.3 → v5.4 (Living-programme upgrade & mathematical fix):** Fixed the axis-scaling ambiguity in Section 2's weighted influence-score estimator by writing explicit cluster-summed ($\sum_c$) and seed-averaged ($k^{-1}\sum_r$) formulas with finite-cluster weight-centering factor $d_w$; aligned Q5 speech selection with the 80% eligibility UCB default; specified test-time parity for Q3 arm `iv`; consolidated conflicting historical appendices; and added **Section 12 (Falsification Triggers, Predeclared Pivots, and Living-Plan Amendment Rules)** plus the multi-tranche programme map so the entire Q1–Q7 plan can evolve rigorously whenever evidence disproves a premise.
5. 19 Nov 23:54: the agent integrity check PASS; 'the agent's check' removed from the rule (his correction); corrections marked
   ADOPTED; 5.4 h speech baseline changed from WAITS to PROVISIONAL after the author's 'do not be over cautious' review note. 1 h comparator
   errors copied (7.4-7.5) with a correction of my old range.
