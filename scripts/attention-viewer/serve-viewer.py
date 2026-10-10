"""Serve the adjacent attention viewer on a free localhost port."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import re
from urllib.request import Request, urlopen
import webbrowser


class RangeHandler(SimpleHTTPRequestHandler):
    """Single byte ranges let browsers seek without downloading the whole MP3."""

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def send_head(self):
        self.byte_range = None
        value = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not value or not path.is_file() or self.headers.get('If-Range'):
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', value.strip())
        start, end = 0, size - 1
        if match and any(match.groups()):
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            else:
                start = max(0, size - int(right))
        else:
            start = size
        if start > end or start >= size:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        stream = path.open('rb')
        stream.seek(start)
        self.byte_range = (start, end)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Last-Modified', self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        if self.byte_range is None:
            return super().copyfile(source, outputfile)
        remaining = self.byte_range[1] - self.byte_range[0] + 1
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name in ['index.html', 'viewer-data.js', 'audio.mp3']:
        if not (root/name).is_file():
            raise FileNotFoundError(f'Keep this launcher beside the viewer files: {root/name}')
    handler = partial(RangeHandler, directory=str(root))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    url = f'http://127.0.0.1:{server.server_port}/'
    print('Attention viewer: '+url, flush=True)
    print('Keep this window open. Close it or press Ctrl+C to stop.', flush=True)
    if args.check:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for name in ['', 'viewer-data.js', 'audio.mp3', 'attention-over-30.html', 'threshold-data.js']:
                with urlopen(url+name, timeout=10) as response:
                    assert response.status == 200
                    response.read()
            print('HTTP checks passed for main page, table, data and audio.', flush=True)
            audio = (root/'audio.mp3').read_bytes()
            for header, expected in [('bytes=100-199', audio[100:200]),
                                     ('bytes=-100', audio[-100:]),
                                     (f'bytes={len(audio)-100}-', audio[-100:])]:
                with urlopen(Request(url+'audio.mp3', headers={'Range': header}), timeout=10) as response:
                    assert response.status == 206
                    assert response.read() == expected
                    assert response.headers['Accept-Ranges'] == 'bytes'
            print('Audio byte-range seek checks passed.', flush=True)
        finally:
            server.shutdown();server.server_close()
        return
    if not args.no_browser:
        threading.Timer(.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':main()
