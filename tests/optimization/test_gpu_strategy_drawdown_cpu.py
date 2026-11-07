"""Execute the actual shader risk blocks on CPU; no torch/GPU required."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNELS = sorted((ROOT / "passivbot-rust/src/gpu").glob("*.metal"))
BLOCK = re.compile(r"if \(equity > run_peak\) \{.*?max_dd = fmax\(max_dd, (?:dd|drawdown)\);", re.S)

@pytest.mark.parametrize("kernel", KERNELS, ids=lambda p: p.name)
def test_raw_strategy_drawdown_matches_exact_formula(kernel, tmp_path):
    blocks = BLOCK.findall(kernel.read_text())
    if not blocks:
        pytest.skip("not a portfolio drawdown kernel")
    assert shutil.which("g++"), "CPU shader regression requires g++"
    # Identical candidate equity paths, including terminal gaps beyond liquidation.
    paths = [[100, 120, 110, 90], [100, 120, 110, 4],
             [100, 120, 110, -2], [100, 120, 110, -20]]
    for index, block in enumerate(blocks):
        source = tmp_path / f"risk{index}.cpp"
        source.write_text("#include <cmath>\n#include <iostream>\nusing namespace std;\nint main(){\n"
            + "float paths[4][4]={{100,120,110,90},{100,120,110,4},{100,120,110,-2},{100,120,110,-20}};\n"
            + "for(auto &path: paths){float run_peak=-INFINITY, max_dd=0, last_high_k=-1,recovery_max_min=0;\n"
            + "for(int k=0;k<4;k++){float kf=k,equity=path[k],eqf=fmax(equity,5.0f),effective_equity=eqf;\n"
            + block + "}cout.precision(9);cout<<max_dd<<std::endl;} }\n")
        binary = tmp_path / f"risk{index}"
        subprocess.run(["g++", "-std=c++17", str(source), "-o", str(binary)], check=True, capture_output=True)
        actual = list(map(float, subprocess.check_output([str(binary)], text=True).split()))
        expected = []
        for path in paths:
            peak = -float("inf")
            worst = 0
            for value in path:
                peak = max(peak, value)
                worst = max(worst, (peak-value)/max(abs(peak),1e-12))
            expected.append(worst)
        assert actual == pytest.approx(expected, abs=1e-6)
        assert actual[1] < actual[2] < actual[3]  # no liquidation-floor rank ties


def test_all_portfolio_drawdown_blocks_use_raw_equity():
    count = 0
    for kernel in KERNELS:
        text = kernel.read_text()
        assert not re.search(r"if \((?:eqf|effective_equity) > run_peak\)", text)
        count += len(BLOCK.findall(text))
    assert count == 7  # includes chunked replay variants
