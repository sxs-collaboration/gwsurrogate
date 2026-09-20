"""Check that failed downloads preserve existing model files and clean up partial files."""

import hashlib
from unittest.mock import MagicMock, Mock

import pytest
import requests
from gwsurrogate import catalog


# pytest runs this function twice, once for each value of failure.
@pytest.mark.parametrize('failure', ['checksum', 'interrupted'])
def test_pull_failed_download_preserves_existing_file(tmp_path, monkeypatch, failure):
  # pytest supplies tmp_path, a temporary directory unique to this test run.
  # monkeypatch makes temporary changes to objects and undoes them after the test.
  # The b prefix makes these byte strings, matching the binary files used by pull().
  path = tmp_path / 'TestModel.h5'
  path.write_bytes(b'previous model')

  # Register a temporary model whose expected contents are b'new model'. Its MD5
  # differs from the existing file, so pull() will attempt to download a replacement.
  info = catalog.surrogate_info('https://example.com/TestModel.h5', '', '', hashlib.md5(b'new model').hexdigest())
  monkeypatch.setitem(catalog._surrogate_world, 'TestModel', info)

  def download_chunks(chunk_size):
    # yield supplies one chunk to pull()'s download loop, then execution resumes here.
    # Either the transfer stops with an error, or it finishes with the wrong contents.
    # chunk_size is accepted because pull() passes it to iter_content().
    yield b'incomplete download'
    if failure == 'interrupted':
      raise requests.exceptions.ChunkedEncodingError('Transfer interrupted')

  # MagicMock stands in for the HTTP response. It also supports special methods
  # such as __enter__, which Python calls when entering a "with" block.
  response = MagicMock()
  # Make "with requests.get(...) as r" assign this fake response to r.
  response.__enter__.return_value = response
  # side_effect makes iter_content(...) call our download_chunks function.
  response.iter_content.side_effect = download_chunks
  # Replace requests.get with a function-like mock that returns the fake response.
  # No network request is made; raise_for_status() on the fake response does nothing.
  monkeypatch.setattr(catalog.requests, 'get', Mock(return_value=response))

  # pytest.raises requires pull() to raise the expected exception. The test fails
  # if pull() returns normally or raises a different kind of exception.
  error = requests.exceptions.ChunkedEncodingError if failure == 'interrupted' else ValueError
  with pytest.raises(error):
    catalog.pull('TestModel', tmp_path)

  # The original bytes must survive, and the directory must contain only that file:
  # no partial download or backup directory should remain after either failure.
  assert path.read_bytes() == b'previous model'
  assert set(tmp_path.iterdir()) == {path}
