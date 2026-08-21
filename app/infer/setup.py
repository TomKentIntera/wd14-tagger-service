import os
import pathlib
import tempfile
from urllib.error import URLError

import httpx
from loguru import logger
from robust_downloader import download


def _unlink_quietly(path: str) -> None:
    try:
        pathlib.Path(path).unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Could not remove temporary download file {path}: {e}")


def _create_part_path(file_dir: str, file_name: str) -> str:
    """
    Create a unique .part path in file_dir.

    tempfile.mkstemp() leaves the file open. Windows cannot unlink or replace an
    open handle, so the descriptor must be closed before the path is used.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f"{file_name}.",
        suffix=".part",
        dir=file_dir,
    )
    os.close(tmp_fd)
    return tmp_path


async def _httpx_stream_download(file_name: str, file_url: str, file_dir: str):
    """
    Stream-download via httpx with redirect support and atomic writes.
    """
    file_path = pathlib.Path(file_dir).joinpath(file_name)
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)

    tmp_path = _create_part_path(file_dir, file_name)

    max_attempts = 3
    try:
        for attempt in range(1, max_attempts + 1):
            total_size = 0
            try:
                async with httpx.AsyncClient(
                    timeout=300.0, follow_redirects=True
                ) as client:
                    async with client.stream("GET", file_url) as response:
                        response.raise_for_status()
                        with open(tmp_path, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    total_size += len(chunk)
                pathlib.Path(tmp_path).replace(file_path)
                logger.success(
                    f"Download file {file_name} from {file_url} success ({total_size} bytes)"
                )
                return
            except (httpx.HTTPError, OSError) as e:
                if attempt == max_attempts:
                    raise
                logger.warning(
                    f"httpx download attempt {attempt}/{max_attempts} failed for {file_name}: {e}"
                )
    finally:
        _unlink_quietly(tmp_path)


async def download_file(file_name: str, file_url: str, file_dir: str):
    """
    Download file from url
    :param file_name: File name
    :param file_url: File url
    :param file_dir: The directory to store the file
    :return: None
    :raises: DownloadError, FileSizeMismatchError
    """
    # Try robust_downloader first, but fall back to httpx when needed.
    try:
        if download(url=file_url, folder=file_dir, filename=file_name):
            logger.success(f"Download file {file_name} from {file_url} success")
            return
    except KeyError as e:
        # KeyError.args[0] contains the missing key
        missing_key = str(e.args[0] if e.args else '').lower()
        if 'content-length' in missing_key or 'content-length' in str(e).lower():
            logger.warning(f"Content-Length header missing, using httpx for {file_name}")
            await _httpx_stream_download(file_name, file_url, file_dir)
            return
        else:
            raise
    except Exception as e:
        # robust_downloader can fail for missing headers/redirects depending on host.
        logger.warning(f"robust_downloader failed for {file_name}, using httpx fallback: {e}")
        await _httpx_stream_download(file_name, file_url, file_dir)
        return


async def download_model(model_name: str, file_dir: str = "models") -> str:
    """
    Download model from url
    :param model_name: Model name
    :param file_dir: The directory to store the file
    :return: None
    """
    # Get Current Path / Create models folder
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
    # Check if model already exists
    ab_path = pathlib.Path(file_dir).joinpath(f"{model_name}.onnx").absolute()
    if ab_path.exists():
        logger.info(f"Model {model_name} already exists at {ab_path}, skipping download")
        return str(ab_path)
    
    # Download
    model_url = (
        f"https://huggingface.co/SmilingWolf/{model_name}/resolve/main/model.onnx"
    )
    try:
        await download_file(f"{model_name}.onnx", model_url, file_dir=file_dir)
    except URLError:
        if ab_path.exists():
            logger.warning(
                f"Model {model_name} download failed, but file exists, skip download"
            )
            return str(ab_path)
        raise URLError(f"Model {model_name} download failed")
    except Exception as e:
        raise e
    else:
        logger.info(f"Model {model_name} downloaded at {ab_path}")
    return str(ab_path)


async def download_csv(model_name: str, file_dir: str = "models") -> str:
    """
    Download csv from url
    :param model_name: Model name
    :param file_dir: The directory to store the file
    :return: None
    """
    # Get Current Path / Create models folder
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
    # Check if CSV already exists
    ab_path = pathlib.Path(file_dir).joinpath(f"{model_name}.csv").absolute()
    if ab_path.exists():
        logger.info(f"Tagger CSV {model_name} already exists at {ab_path}, skipping download")
        return str(ab_path)
    
    # Download
    csv_url = f"https://huggingface.co/SmilingWolf/{model_name}/resolve/main/selected_tags.csv"
    try:
        await download_file(f"{model_name}.csv", csv_url, file_dir=file_dir)
    except URLError:
        if ab_path.exists():
            logger.warning(
                f"Tagger CSV {model_name} download failed, but file exists, skip download"
            )
            return str(ab_path)
        raise URLError(f"Model {model_name} download failed")
    except Exception as e:
        raise e
    else:
        logger.info(f"Tagger CSV {model_name} downloaded at {ab_path}")
    return str(ab_path)
