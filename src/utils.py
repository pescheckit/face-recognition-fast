import gc


def cleanup():
    """
    Perform any cleanup needed.
    Right now, we’re just forcing garbage collection.
    """
    gc.collect()
