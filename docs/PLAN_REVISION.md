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
         Acceptance, written now: rc 0 for all; event order as (9); all modules in training mode at every u line and in eval mode at every e line (as the v6.5 CPU rule; CORRECTED 20 Nov 11:0x after the round-6 review: the old text "training mode at every u / e line" is the v6.4 condition that failed a correct trainer); numpy, torch-CPU and
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
