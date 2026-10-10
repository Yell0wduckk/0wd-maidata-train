"""Explicit local oracle capture; never imported by production inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--names", nargs="*", help="capture named additions, preserving the pinned existing cases")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/"src/maimai_baseline/assets/miacode-slide-runtime.json").read_text("utf-8"))
    cases = {f"single-{k}": f"(120){{4}}{k}[4:4],,,,,E"
             for family in ("slides", "wifi") for k in data[family] if k.startswith("1")}
    cases.update({
        "same-path": "(120){4}1-5[4:4],1-5[4:1],,,,E",
        "different-path": "(120){4}1-5[4:4],3-5[4:1],,,,E",
        "wifi": "(120){4}1w5[4:4],1-5[4:1],,,,E",
        "wifi-pair": "(120){4}1w5[4:4],1w5[4:1],,,,E",
        "hold": "(120){4}1-5[4:4],5h[4:4],,,,E",
        "touch": "(120){4}1-5[4:4],C/B5/A5,,,,E",
        "branch": "(120){4}1-5[4:4]*-3[4:1],1-5[4:1],,,,E",
        "outer": "(240){16}8>3[4:1],,,,,8,E",
        "protected": "(240){16}8>3[4:1],,,,,8x,E",
        "outer-star": "(240){16}8>3[4:1],,,,,8-4[4:1],E",
        "aligned": "(120){4}1-5[4:4],1,5h[4:1],E",
        "bpm": "(120){4}1-5[4:4],(240)1w5[4:2],C/5h[4:1],E",
        "tail": "(120){4}1-5[4:1],,,5-1[4:1],E",
        "touch-ring": "(120){4}1-5[4:4],B1/E1/B2/E2/B3/E3/B4/E4/B5/E5/B6/E6/B7/E7/B8/E8,E",
        "outer-hold": "(240){16}8>3[4:1],,,,,8h[4:4],E",
        "hold-zero": "(120){4}1-5[4:4],5h[4:0],E",
        "touch-mixed": "(120){4}1-5[4:4],A1/B1/E8/A5,E",
        "rotate90": "(120){4}3-7[4:4],3-7[4:1],E",
        "rotate180": "(120){4}5-1[4:4],5-1[4:1],E",
        "rotate270": "(120){4}7-3[4:4],7-3[4:1],E",
        "mirror": "(120){4}8-4[4:4],8-4[4:1],E",
        "guard50": "(120){4}1-5[4:4],{40},1,E",
        "guard150": "(120){4}1-5[4:4],{40},,,1,E",
        "release-tail": "(120){4}5h[4:1]/1-5[4:4],5-1[4:1],E",
    })
    records = []
    if args.names:
        previous = json.loads(args.output.read_text("utf-8"))
        if previous["executable_sha256"] != hashlib.sha256(args.exe.read_bytes()).hexdigest():
            raise ValueError("cannot combine captures from different executables")
        records = [c for c in previous["cases"] if c["name"] not in args.names]
        cases = {n: cases[n] for n in args.names}
    for name, text in cases.items():
        print(f"Capturing {name}", flush=True)
        proc = subprocess.run([str(args.exe.resolve()), "--with-markers"], input=text,
                              encoding="utf-8", capture_output=True, check=True, timeout=60)
        result = json.loads(proc.stdout)
        if not result["ok"]:
            raise ValueError((name, result["parse"]))
        records.append(dict(name=name, simai=text, result=result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(executable_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        resource_manifest=json.loads((root/"src/maimai_baseline/assets/miacode-slide-runtime-manifest.json").read_text("utf-8")),
        cases=records), ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
    print(f"Captured {len(records)} independent oracle cases")


if __name__ == "__main__":
    main()
