from os.path import join
from time import perf_counter

import numpy as np
from metrics import SERIES_ANALYSIS_LABELS, Compare, Mask, Region, prettyPrintTable

if __name__ == "__main__":
    t = perf_counter()
    OBJ = ["bladder", "prostate", "rectum", "femoral_head_l", "femoral_head_r"]

    load = lambda x: [
        Mask(np.load(join("data", f"{x}_{obj}.npy")).transpose(2, 1, 0)) for obj in OBJ
    ]

    masksU = load("prostate_upright")
    masksS = load("prostate_supine")

    anchorU = Mask(masksU[3].mask | masksU[4].mask).center_of_mass
    anchorS = Mask(masksS[3].mask | masksS[4].mask).center_of_mass

    regionU = Region(*masksU[0:3], target="prostate", anchor=anchorU, labels=OBJ[:3])
    regionS = Region(*masksS[0:3], target="prostate", anchor=anchorS, labels=OBJ[:3])

    comp = Compare(regionU, regionS)
    print(f"\n0. loading ============================> (t = {perf_counter() - t:.3f}s)")

    if True:
        t = perf_counter()
        V = list(comp.getVolDiff(diff_only=False).values())
        labels = ["vol diff", "upright Vol", "supine Vol"]
        print(f"\n1. volume diff ========================> (t = {perf_counter() - t:.3f}s)")
        prettyPrintTable(OBJ[:3], V, labels)

    if True:
        t = perf_counter()
        SA = list(comp.getSADiff(diff_only=False).values())
        labels = ["sa diff", "upright sa", "supine sa"]
        print(f"\n2. surface area diff ==================> (t = {perf_counter() - t:.3f}s)")
        prettyPrintTable(OBJ[:3], SA, labels)

    if True:
        t = perf_counter()
        DISP = list(comp.getROIDisplacementDiff().values())
        labels = ["L(-ve)/R(+ve)", "A(-ve)/P(+ve)", "I(-ve)/S(+ve)", "Magnitude"]
        print(f"\n3. roi displacement ===================> (t = {perf_counter() - t:.3f}s)")
        prettyPrintTable(OBJ[:3], DISP, labels)

    if True:
        t = perf_counter()
        ASD = list(comp.getBSDDiff("asd").values())
        HD95 = list(comp.getBSDDiff("hd95").values())
        HD = list(comp.getBSDDiff("hd").values())
        print(f"\n4. bidirectional surface discrepancy ==> (t = {perf_counter() - t:.3f}s)")
        prettyPrintTable(OBJ[:3], [ASD, HD95, HD], ["ASD", "HD95", "HD"])

    if True:
        t = perf_counter()
        data = comp.getSeparationDistanceDiff("volume")
        print(f"\n5. separation distance (vol) ===========> (t = {perf_counter() - t:.3f}s)")
        for loc, dat in data.items():
            print(f"\n\t[{loc}]")
            prettyPrintTable(["diff", "upright", "supine"], dat, SERIES_ANALYSIS_LABELS)

    if True:
        t = perf_counter()
        data = comp.getSeparationDistanceDiff("surface")
        print(f"\n6. separation distance (surface) =======> (t = {perf_counter() - t:.3f}s)")
        for loc, dat in data.items():
            print(f"\n\t[{loc}]")
            prettyPrintTable(["diff", "upright", "supine"], dat, SERIES_ANALYSIS_LABELS)

    if True:
        t = perf_counter()
        data = comp.getSeparationDistanceDiff("rcvs")
        print(f"\n7. separation distance (rcvs) ==========> (t = {perf_counter() - t:.3f}s)")
        for loc, dat in data.items():
            print(f"\n\t[{loc}]")
            prettyPrintTable(["diff", "upright", "supine"], dat, SERIES_ANALYSIS_LABELS)

    if True:
        t = perf_counter()
        data = comp.getPercentageOverlapDiff("volume")
        print(f"\n8. percentage overlap (volume) =========> (t = {perf_counter() - t:.3f}s)")
        for loc, dat in data.items():
            print(f"\n\t[{loc}]")
            prettyPrintTable(["diff", "upright", "supine"], dat, SERIES_ANALYSIS_LABELS)
