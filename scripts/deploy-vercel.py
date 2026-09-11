"""Deploy only the built static site. Credentials stay local and out of the upload."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--secrets', type=Path, default=Path.home() / 'Documents' / 'mysecrets')
parser.add_argument('--team', required=True)
parser.add_argument('--name', default='llm-anatomy-lab')
args = parser.parse_args()
match = re.search(r'^\s*Vercel_key\s*[:=]\s*(.+?)\s*$', args.secrets.read_text(), re.M)
if not match:
    raise SystemExit('Vercel_key is missing from the local secrets file')
token = match.group(1).strip().strip('"\'')
root = Path(__file__).resolve().parents[1]
dist = root / 'dist'
if not (dist / 'index.html').is_file():
    raise SystemExit('Run npm run build first')
scope = '?teamId=' + urllib.parse.quote(args.team)

def request(path, data=None, headers=None, method=None):
    hdr = {'Authorization': 'Bearer ' + token, **(headers or {})}
    req = urllib.request.Request('https://api.vercel.com' + path + scope, data=data, headers=hdr, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        message = error.read().decode(errors='replace').replace(token, '[redacted]')
        raise RuntimeError(f'Vercel HTTP {error.code}: {message[:1200]}') from None

files = []
for path in sorted(dist.rglob('*')):
    if not path.is_file():
        continue
    if path.is_symlink() or not path.resolve().is_relative_to(dist.resolve()):
        raise SystemExit('Refusing a file outside the static build')
    data = path.read_bytes()
    if token.encode() in data:
        raise SystemExit('Credential detected in build; upload aborted')
    relative = path.relative_to(dist).as_posix()
    if relative.startswith('.') or 'mysecrets' in relative.lower():
        raise SystemExit('Unexpected private file in static build')
    files.append((relative, data, hashlib.sha1(data).hexdigest()))
print(f'Uploading {len(files)} static files / {sum(len(f[1]) for f in files)/1e6:.2f} MB', flush=True)

def upload(item):
    name, data, sha = item
    request('/v2/files', data, {'Content-Type': 'application/octet-stream', 'x-vercel-digest': sha, 'Content-Length': str(len(data))}, 'POST')
    return {'file': name, 'sha': sha, 'size': len(data)}

try:
    with ThreadPoolExecutor(max_workers=4) as pool:
        uploaded = list(pool.map(upload, files))
    body = {'name': args.name, 'target': 'production', 'files': uploaded,
            'projectSettings': {'framework': None, 'buildCommand': '', 'installCommand': '', 'outputDirectory': '.'}}
    deployment = request('/v13/deployments', json.dumps(body).encode(), {'Content-Type': 'application/json'}, 'POST')
    receipt = {k: deployment.get(k) for k in ['id', 'url', 'readyState', 'alias', 'projectId']}
    receipt['teamId'] = args.team
    (root / '.vercel').mkdir(exist_ok=True)
    (root / '.vercel' / 'deployment.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
except Exception as error:
    print(str(error).replace(token, '[redacted]'), file=sys.stderr)
    sys.exit(1)
