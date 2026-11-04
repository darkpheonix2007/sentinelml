import hashlib

from sentinelml.data.download import calculate_sha256


def test_hash_reads_all_chunks(tmp_path):
    content = b"a" * 4096 + b"b" * 4096 + b"end"
    path = tmp_path / "file.bin"
    path.write_bytes(content)
    assert calculate_sha256(path) == hashlib.sha256(content).hexdigest()
