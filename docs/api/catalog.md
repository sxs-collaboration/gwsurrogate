# `catalog` module

`catalog.pull(model)` reuses a local file when its MD5 matches the checksum in the installed catalog. Missing or mismatched files are downloaded and verified before installation. Use `catalog.pull(model, force=True)` to download again even when the local file matches; the new download must still pass verification.

Existing files are backed up only after a replacement download passes verification. Failed downloads leave existing files untouched. For `.tar.gz` models, the verified archive is retained and extracted on every call, and `pull()` returns the extracted directory. Other models return the data file path. The optional `sdir` argument selects a download directory; when omitted, it defaults to `catalog.download_path()`.

::: gwsurrogate.catalog
