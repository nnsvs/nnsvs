import tempfile

import pytest
from nnsvs.logger import getLogger


@pytest.mark.parametrize("verbose", [0, 1, 100])
def test_logger(verbose):
    logger = getLogger(verbose=verbose)

    logger.info(f"verbose={verbose}")


@pytest.mark.parametrize("verbose", [0, 1, 100])
def test_logger_filename(verbose):
    with tempfile.NamedTemporaryFile() as tmp:
        logger = getLogger(verbose=verbose, filename=tmp.name)

    logger.info(f"verbose={verbose}")
