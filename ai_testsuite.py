import logging
import os.path
import shutil
import tarfile
import tempfile
import urllib.request
from os import mkdir
from os.path import dirname, exists, getsize, join
from pathlib import Path
from shutil import rmtree
from typing import Tuple

import pytest

__TEST_DATA_FILENAME = "test_data.tar.gz"
__TEST_DATA_URL = "https://github.com/NVIDIA/NeMo/releases/download/v1.0.0rc1/"
__TEST_DATA_SUBDIR = ".data"


def pytest_addoption(parser):
    parser.addoption(
        '--cpu', action='store_true', help="Use CPU during testing (DEFAULT: GPU)"
    )
    parser.addoption(
        '--use_local_test_data',
        action='store_true',
        help="Use local test data/skip downloading from URL/GitHub (DEFAULT: False)",
    )
    parser.addoption(
        '--with_downloads',
        action='store_true',
        help="Activate tests which download models from the cloud.",
    )
    parser.addoption(
        '--relax_numba_compat',
        action='store_false',
        help="Relax numba compatibility checks to just availability of cuda.",
    )
    parser.addoption(
        "--nightly",
        action="store_true",
        help="Activate tests marked as nightly for QA.",
    )


@pytest.fixture
def device(request):
    if request.config.getoption("--cpu"):
        return "CPU"
    else:
        return "GPU"


@pytest.fixture(autouse=True)
def run_only_on_device_fixture(request, device):
    marker = request.node.get_closest_marker('run_only_on')
    if marker and marker.args[0] != device:
        pytest.skip(f'skipped on this device: {device}')


@pytest.fixture(autouse=True)
def downloads_weights(request):
    if request.node.get_closest_marker('with_downloads'):
        if not request.config.getoption("--with_downloads"):
            pytest.skip('To run this test, pass --with_downloads option. It will download (and cache) models from the cloud.')


@pytest.fixture(autouse=True)
def run_nightly_test_for_qa(request):
    if request.node.get_closest_marker('nightly'):
        if not request.config.getoption("--nightly"):
            pytest.skip('To run this test, pass --nightly option. It will run tests marked as "nightly".')


@pytest.fixture(autouse=True)
def cleanup_local_folder():
    assert not Path("./lightning_logs").exists()
    assert not Path("./NeMo_experiments").exists()
    assert not Path("./nemo_experiments").exists()

    yield

    cleanup_dirs = ["./lightning_logs", "./NeMo_experiments", "./nemo_experiments"]
    for dir in cleanup_dirs:
        if Path(dir).exists():
            rmtree(dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_data_dir():
    return join(dirname(__file__), __TEST_DATA_SUBDIR)


def download_and_extract_data(test_dir, test_data_archive, url):
    urllib.request.urlretrieve(url, test_data_archive)
    extract_tar(test_data_archive, test_dir)


def extract_tar(tar_path, extract_to):
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)


def extract_data_from_tar(test_dir, test_data_archive, url=None, local_data=False):
    if exists(test_dir):
        handle_existing_data(test_dir, test_data_archive, local_data)
    else:
        mkdir(test_dir)

    if url and not local_data:
        download_and_extract_data(test_dir, test_data_archive, url)
    else:
        extract_tar(test_data_archive, test_dir)


def handle_existing_data(test_dir, test_data_archive, local_data):
    if not local_data:
        rmtree(test_dir)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Copying local tarfile to temporary storage..")
            shutil.copy2(test_data_archive, temp_dir)
            print("Deleting test dir to cleanup old data")
            rmtree(test_dir)
            mkdir(test_dir)
            print("Restoring local tarfile to test dir")
            shutil.copy2(os.path.join(temp_dir, os.path.basename(test_data_archive)), test_data_archive)


@pytest.fixture(scope="session")
def k2_is_appropriate() -> Tuple[bool, str]:
    try:
        from nemo.core.utils.k2_guard import k2
        return True, "k2 is appropriate."
    except ImportError as e:
        logging.exception(e)
        return False, "k2 is not available or does not meet the requirements."


@pytest.fixture(scope="session")
def k2_cuda_is_enabled(k2_is_appropriate) -> Tuple[bool, str]:
    if not k2_is_appropriate[0]:
        return k2_is_appropriate

    import torch
    from nemo.core.utils.k2_guard import k2

    if torch.cuda.is_available() and k2.with_cuda:
        return True, "k2 supports CUDA."
    elif torch.cuda.is_available():
        return False, "k2 does not support CUDA. Consider using a k2 build with CUDA support."
    else:
        return False, "k2 needs CUDA to be available in torch."


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "run_only_on(device): runs the test only on a given device [CPU | GPU]",
    )
    config.addinivalue_line(
        "markers",
        "with_downloads: runs the test using data present in tests/.data",
    )
    config.addinivalue_line(
        "markers",
        "nightly: runs the nightly test for QA.",
    )

    test_dir = join(dirname(__file__), __TEST_DATA_SUBDIR)
    test_data_archive = join(test_dir, __TEST_DATA_FILENAME)

    local_size, remote_size = get_test_data_sizes(config, test_data_archive)

    if config.option.use_local_test_data:
        if local_size == -1:
            pytest.exit(f"Test data `{test_data_archive}` is not present in the system")
        else:
            print(f"Using the local `{__TEST_DATA_FILENAME}` test archive ({local_size}B) found in `{test_dir}`.")

    if not config.option.use_local_test_data:
        url = __TEST_DATA_URL + __TEST_DATA_FILENAME
        if remote_size != local_size:
            download_and_extract_data(test_dir, test_data_archive, url)
        else:
            print(f"A valid `{__TEST_DATA_FILENAME}` test archive ({local_size}B) found in `{test_dir}`.")
    else:
        extract_tar(test_data_archive, test_dir)

    if config.option.relax_numba_compat is not None:
        from nemo.core.utils import numba_utils
        numba_utils.set_numba_compat_strictness(strict=config.option.relax_numba_compat)


def get_test_data_sizes(config, test_data_archive):
    try:
        local_size = getsize(test_data_archive)
    except OSError:
        local_size = -1

    remote_size = -1
    if not config.option.use_local_test_data:
        try:
            url = __TEST_DATA_URL + __TEST_DATA_FILENAME
            u = urllib.request.urlopen(url)
            remote_size = int(u.info()["Content-Length"])
        except Exception as e:
            logging.error(f"Failed to retrieve remote test data size: {e}")
            if local_size == -1:
                pytest.exit(f"Test data not present and cannot access the '{url}' URL")
            else:
                print(f"Cannot access the '{url}' URL, using the test data ({local_size}B) found in `{test_dir}`.")
    return local_size, remote_size
