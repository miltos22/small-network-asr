#!/usr/bin/env python3
"""Per-run GPU ledger for MASTER_PLAN Part C v6.1 (the agent, 20 Nov 2026). Every sum is computed here, never typed.
Prices (seconds) and their sources:
  LOOP_STEP_S   = 0.0863 s/optimizer step  MEASURED: b6iiWN180_GT_s3.log, 1,302 s loop / 15,091 steps (5.4 h, 180 passes)
  EPOCH_OVH_S   = 1.37 s/epoch fixed        MEASURED-DERIVED: b6t1h_TS_d1_s3.log 2.85 s/epoch at 17 steps/epoch minus 17 x 0.0863
  SETUP_S       = 396 s/run                 MEASURED-DERIVED: 1 h batch 200 min / 4 runs = 50 min wall vs 43.4 min loop (2,605 s)
  STEPS_SMALL   = 16,524 updates            MEASURED: b6t1h_TS_d1_s3.log (1 h x 972 passes; 5.4 h x 180 has 15,091)
  H100          = 3.88 h/run                MEASURED (Part A 4b: 13,981 s, 1.7 M params, 20 passes)
  CIFAR100_S    = 0.19 h/run                MEASURED S 532-582 s; x1.2 margin for P / G_mix (UNMEASURED) + setup
  CIFAR10pc     = 0.04 h/run                INFERRED: 1/10 of the updates at 200 epochs + setup; UNMEASURED
  E1 (loader fix: several passes per loop epoch, eval every 36 passes) removes the per-epoch overhead: INFERRED, must be timed.
"""
import json, sys
LOOP_STEP_S, EPOCH_OVH_S, SETUP_S, STEPS = 0.0863, 1.37, 396, 16524
def small(hours_data, e1, frac=1.0):
    passes = {0.5: 1944, 1.0: 972, 5.4: 180}[hours_data]        # equal-update convention of 2-V (0.5 h: frozen before data)
    epochs = passes * frac
    ovh = 0 if (e1 or hours_data == 5.4) else epochs * EPOCH_OVH_S   # 5.4 h overhead is already inside its measured loop rate
    return (STEPS * frac * LOOP_STEP_S + ovh + SETUP_S) / 3600
H100, PROXY, C100, C10 = 3.88, 0.15 * 3.88, 0.19, 0.04

def items(e1):
    s05, s1, s54, s54h = small(0.5, e1), small(1.0, e1), small(5.4, e1), small(5.4, e1, 0.5)
    L = []
    def add(group, arm, domain, size, n, price, stage, reuse="", scope="R"):
        L.append(dict(group=group, arm=arm, domain=domain, data_size=size, runs=n, gpu_h_each=round(price, 3),
                      gpu_h=round(n * price, 3), stage=stage, reuse_source=reuse, scope=scope))
    # Step 0 (B never-drop: split integrity, 3-seed baselines, ResNet seed 1, simulations)
    add("Step0", "spent CIFAR S x3 + ResNet-32 s1 + smokes", "both", "-", 1, 0.85, "spent", "MEASURED ~51 GPU-min")
    add("Step0", "Shared baseline (WN recipe), seeds 0-2", "speech", "5.4h", 3, s54, "calibration")
    add("Step0", "E1 loader fix: timing smoke (3 arms x 30 passes)", "speech", "1h", 1, 0.05, "calibration")
    add("Step0", "E1 known answer: one full 1 h run with E1 vs old-loader seed spread", "speech", "1h", 1, small(1.0, True), "calibration")
    # Q2(c): his tree (semantic) vs shuffled, 5 paired seeds, independent shuffle draws
    add("Q2c", "semantic + shuffled", "speech", "0.5h", 10, s05, "primary (98.3 %)")
    add("Q2c", "semantic + shuffled", "speech", "1h", 10, s1, "robustness", scope="H")
    # Q4 at 5.4 h: primary Holm family (W-R, WH-R, speed) + W-D mechanism
    add("Q4", "R, D, W, WH full, 3 seeds", "speech", "5.4h", 12, s54, "primary")
    add("Q4", "W, R compressed half, 3 seeds", "speech", "5.4h", 6, s54h, "primary (speed)")
    add("Q4", "R, D, W, WH full, 3 seeds", "speech", "1h", 12, s1, "secondary", scope="F")
    add("Q4", "W, R compressed half, 3 seeds", "speech", "1h", 6, small(1.0, e1, 0.5), "secondary", scope="F")
    # Q1 low data speech: 3x2 WD/dropout grid at seed 0 + seeds 1-2 on the chosen setting (seed 0 reused exactly)
    add("Q1", "P, Shared grid 3x2", "speech", "1h", 12, s1, "co-primary low data")
    add("Q1", "P, Shared seeds 1-2", "speech", "1h", 4, s1, "co-primary low data", "seed 0 = grid winner (exact)")
    add("Q1", "G_mix grid + seeds", "speech", "1h", 8, s1, "mechanism", scope="F")
    # Q1 CIFAR: 10 % grid 3x2 + seeds; 100 % three WD per arm + seeds; P / Shared / G_mix
    add("Q1", "P, Shared, G_mix grid 3x2 + seeds 1-2", "cifar", "10%", 24, C10, "co-primary low data")
    add("Q1", "P, Shared, G_mix 3 WD + seeds 1-2", "cifar", "100%", 15, C100, "co-primary full data")
    # Q1 100 h speech (staged two-arm P vs augmented Shared) and what it needs
    add("Q1", "P, Shared grid 3x2 (proxy candidates)", "speech", "5.4h", 12, s54, "calibration", scope="F")
    add("Q1", "15 % proxies P, aug Shared, 2 candidates", "speech", "100h", 4, PROXY, "calibration", scope="F")
    add("Q1", "P, aug Shared, seeds 0-1", "speech", "100h", 4, H100, "co-primary full data (provisional)", scope="F")
    add("Q1", "P, aug Shared, seed 2 (B: required for a gate/test claim)", "speech", "100h", 2, H100, "co-primary full data", scope="F")
    # Q8 screens and the outside reference (descriptive)
    add("Q8", "ideas a-c x 2 seeds", "speech", "5.4h", 6, s54, "exploration", "base = Step0 baseline seeds 0-1 if config identical")
    add("Q8", "idea d (named + timing bundle) x 2 seeds", "speech", "5.4h", 2, s54, "exploration", scope="F")
    add("Q8", "paired base if exact reuse fails", "speech", "5.4h", 2, s54, "exploration", scope="C")
    add("REF", "outside recipe: smoke + 2 seeds (x1.5 loop margin)", "speech", "5.4h", 2, (STEPS * 1.5 * LOOP_STEP_S + SETUP_S) / 3600 + 0.05, "descriptive")
    add("REF", "outside recipe 2 seeds", "speech", "1h", 2, (STEPS * 1.5 * LOOP_STEP_S + SETUP_S) / 3600, "descriptive", scope="F")
    return L

out = {}
for e1 in (True, False):
    L = items(e1)
    r = sum(x["gpu_h"] for x in L if x["scope"] == "R"); f = sum(x["gpu_h"] for x in L if x["scope"] != "C")
    new_r = sum(x["gpu_h"] for x in L if x["scope"] == "R" and x["stage"] != "spent")
    hh = sum(x["gpu_h"] for x in L if x["scope"] == "H"); cc = sum(x["gpu_h"] for x in L if x["scope"] == "C")
    new_f = sum(x["gpu_h"] for x in L if x["stage"] != "spent" and x["scope"] != "C")
    out["E1" if e1 else "noE1"] = dict(items=L, recommended=round(r, 2), recommended_plus_10pc_contingency=round(r + 0.1 * new_r, 2),
                                       headroom_item=round(hh, 2), contingency_covered_item=round(cc, 2), full=round(f, 2), full_plus_10pc_contingency=round(f + 0.1 * new_f, 2),
                                       price_small_h={"0.5h": round(small(0.5, e1), 3), "1h": round(small(1.0, e1), 3), "5.4h": round(small(5.4, e1), 3)})
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=1)
for k, v in out.items():
    print(k, "prices", v["price_small_h"], "| recommended", v["recommended"], "+10%", v["recommended_plus_10pc_contingency"],
          "| headroom", v["headroom_item"], "cont-item", v["contingency_covered_item"], "| full", v["full"], "+10%", v["full_plus_10pc_contingency"])
    for g in ("Step0", "Q2c", "Q4", "Q1", "Q8", "REF"):
        print("   ", g, "R", round(sum(x["gpu_h"] for x in v["items"] if x["group"] == g and x["scope"] == "R"), 2),
              "all", round(sum(x["gpu_h"] for x in v["items"] if x["group"] == g and x["scope"] != "C"), 2))
