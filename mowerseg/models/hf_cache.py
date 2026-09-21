"""Use installed artifacts without an unnecessary network metadata round trip."""


def from_pretrained_cached(factory, source, **kwargs):
    try:
        return factory.from_pretrained(source, local_files_only=True, **kwargs)
    except OSError:
        return factory.from_pretrained(source, **kwargs)
