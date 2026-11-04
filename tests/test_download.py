import hashlib
import http.client
import io

from sentinelml.data import download
from sentinelml.data.download import calculate_sha256


def test_hash_reads_all_chunks(tmp_path):
    content = b"a" * 4096 + b"b" * 4096 + b"end"
    path = tmp_path / "file.bin"
    path.write_bytes(content)
    assert calculate_sha256(path) == hashlib.sha256(content).hexdigest()


def test_interrupted_download_retries_without_caching_partial_file(tmp_path, monkeypatch):
    content = b"verified archive"
    attempts = 0

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            raise http.client.IncompleteRead(b"partial")

    def fake_urlopen(url, timeout):
        nonlocal attempts
        attempts += 1
        return InterruptedResponse() if attempts == 1 else io.BytesIO(content)

    monkeypatch.setattr(download, "RAW_DIRECTORY", tmp_path)
    monkeypatch.setattr(download, "ZIP_PATH", tmp_path / "archive.zip")
    monkeypatch.setattr(download, "EXPECTED_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)

    download.download_verified_archive()

    assert attempts == 2
    assert (tmp_path / "archive.zip").read_bytes() == content
    assert list(tmp_path.glob("*.part")) == []
