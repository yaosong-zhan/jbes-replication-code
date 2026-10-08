"""Data-free startup tests for empirical scripts and local data-path handling."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class EmpiricalSmokeTest(unittest.TestCase):
    def test_imports_and_loads_synthetic_utf8_named_archives(self):
        with tempfile.TemporaryDirectory() as data_root:
            env = os.environ.copy()
            env['JBES_DATA_ROOT'] = data_root
            script = r'''
import os
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

from empirical import build_earnings_panel as panel
from empirical import compute_alt_measures as alt

root = Path(os.environ['JBES_DATA_ROOT'])
day = date(2024, 1, 2)
archive = root / '2024' / '202401SH股票五档分笔.zip'
archive.parent.mkdir(parents=True)
with zipfile.ZipFile(archive, 'w') as zf:
    zf.writestr('000001_20240102.csv', 'stock,time,value\n000001,09:35:00,1\n'.encode('gbk'))

assert panel.DATA_ROOT == str(root)
assert alt.DATA_ROOT == str(root)
assert panel.find_zip_for_date(day, 'SH') == str(archive)
assert panel.load_stock_day('000001', day, 'SH').iloc[0]['value'] == 1
assert alt.load_stock_day('000001', day, 'SH').iloc[0]['value'] == 1
assert callable(panel.run_full_pipeline)
repo_root = Path.cwd()
for filename in ('build_earnings_panel.py', 'compute_alt_measures.py'):
    result = subprocess.run(
        [sys.executable, str(repo_root / 'empirical' / filename), '--help'],
        cwd=root,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'usage:' in result.stdout.lower()
'''
            subprocess.run(
                [sys.executable, '-c', script],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )


    def test_requires_an_explicit_local_data_path(self):
        env = os.environ.copy()
        env.pop('JBES_DATA_ROOT', None)
        script = r"""
from datetime import date
from empirical import build_earnings_panel as panel
from empirical import compute_alt_measures as alt

calls = (
    (panel.find_zip_for_date, (date(2024, 1, 2), 'SH')),
    (alt.load_stock_day, ('000001', date(2024, 1, 2), 'SH')),
)
for reader, args in calls:
    try:
        reader(*args)
    except RuntimeError as exc:
        assert 'JBES_DATA_ROOT' in str(exc)
    else:
        raise AssertionError('A local licensed-data path must be set explicitly.')
"""
        subprocess.run(
            [sys.executable, '-c', script],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )


if __name__ == '__main__':
    unittest.main()
