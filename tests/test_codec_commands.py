"""Validate encoder/decoder subprocess argument and temporary-file contracts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scope.evaluation.codec_transmissions import real_bpg_transmission, real_jpeg2000_transmission


CODEC_STUB = '''#!/usr/bin/env python3
import pathlib
import shutil
import sys
args = sys.argv[1:]
def get(flag):
    return args[args.index(flag) + 1]
if '-o' not in args:
    raise SystemExit('missing -o')
out = pathlib.Path(get('-o'))
if '-i' in args:
    source = pathlib.Path(get('-i'))
else:
    source = pathlib.Path(args[-1])
if not source.exists():
    raise SystemExit('source missing')
shutil.copyfile(source, out)
'''


class CodecCommandIntegrationTests(unittest.TestCase):
    def test_fake_bpg_encode_and_decode_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / "bpgenc"
            decoder = Path(tmp) / "bpgdec"
            for executable in (encoder, decoder):
                executable.write_text(CODEC_STUB)
                executable.chmod(0o755)
            image = np.full((32, 32, 3), 92, dtype=np.uint8)
            result = real_bpg_transmission(image, bpg_qp=30, encoder_bin=str(encoder), decoder_bin=str(decoder))
            self.assertEqual(result["reconstructed_bgr"].shape, image.shape)
            np.testing.assert_array_equal(result["reconstructed_bgr"], image)
            self.assertGreater(result["estimated_bytes"], 0)

    def test_fake_jpeg2000_openjpeg_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / "opj_compress"
            decoder = Path(tmp) / "opj_decompress"
            for executable in (encoder, decoder):
                executable.write_text(CODEC_STUB)
                executable.chmod(0o755)
            image = np.full((32, 32, 3), 135, dtype=np.uint8)
            result = real_jpeg2000_transmission(
                image, jp2_rate=48, encoder_bin=str(encoder), decoder_bin=str(decoder), backend="openjpeg",
            )
            np.testing.assert_array_equal(result["reconstructed_bgr"], image)
            self.assertGreater(result["estimated_bytes"], 0)

    def test_missing_binary_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            real_bpg_transmission(np.zeros((32, 32, 3), dtype=np.uint8), 30,
                                  encoder_bin="/not/a/real/bpg/encoder", decoder_bin="/not/a/real/bpg/decoder")


if __name__ == "__main__":
    unittest.main()
