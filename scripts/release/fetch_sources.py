"""Download the original source files that the paper-version materials are built from.

    python -m scripts.release.fetch_sources [--raw DIR]

Files go to data/raw/ (or COGBATTERY_RAW, or --raw) at the paths listed in
scripts/release/paper_materials/sources.json. Existing files with the recorded SHA-256 are kept.
A file whose SHA-256 differs from the record is reported, because the built materials would then
differ from the paper. Each source keeps its own terms; see the license field of each entry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).resolve().parent / 'paper_materials' / 'sources.json'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, target, attempts=4):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'CogBattery-fetch'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as handle:
                shutil.copyfileobj(response, handle)
            partial.replace(target)
            return None
        except OSError as error:  # OSF intermittently answers 503
            if attempt == attempts:
                return str(error)
            time.sleep(10 * attempt)


def extract(spec, raw):
    archive, target = raw / spec['archive'], raw / spec['target']
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir() or not member.filename.startswith(spec['member_prefix']):
                continue
            path = target / member.filename[len(spec['member_prefix']):]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.read(member))


def fetch(raw):
    raw = Path(raw)
    sources = json.loads(SOURCES.read_text())
    changed, failed = [], []
    for entry in sources['files']:
        target = raw / entry['path']
        if target.is_file() and sha256(target) == entry['sha256']:
            print(f'ok        {entry["path"]}')
            continue
        error = download(entry['url'], target)
        if error:
            failed.append(entry)
            print(f'FAILED    {entry["path"]}: {error}')
            continue
        status = 'fetched' if sha256(target) == entry['sha256'] else 'CHANGED'
        if status == 'CHANGED':
            changed.append(entry['path'])
        print(f'{status:9} {entry["path"]} ({entry["license"]})')
    for spec in sources['extract']:
        if (raw / spec['archive']).is_file():
            extract(spec, raw)
    for entry in sources['user_supplied']:
        if not (ROOT / entry['path']).is_file():
            print(f'missing   {entry["path"]}: {entry["reason"]} The other tasks build without it.')
    for entry in failed:
        print(f'Download {entry["url"]} by hand and save it as {raw / entry["path"]}, then rerun.')
    if changed:
        raise SystemExit('These files differ from the versions the paper used: ' + ', '.join(changed))
    if failed:
        raise SystemExit(f'{len(failed)} file(s) could not be downloaded.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=Path(os.environ.get('COGBATTERY_RAW', ROOT / 'data' / 'raw')))
    args = parser.parse_args()
    fetch(args.raw)


if __name__ == '__main__':
    main()
