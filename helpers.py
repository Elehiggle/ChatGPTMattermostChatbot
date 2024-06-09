import logging
import traceback
import urllib.parse
import re
import socket
import ssl
from io import BytesIO
from time import monotonic_ns
from functools import lru_cache, wraps
import certifi
from PIL import Image
import validators
from config import log_level

logger = logging.getLogger(__name__)
logger.setLevel(log_level)

REGEX_LOCAL_LINKS = re.compile(
    r"(?:^|\b)(127\.|192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|::1|[fF][cCdD]00::|\blocalhost\b)(?:$|\b)",
    re.IGNORECASE,
)

YOUTUBE_VALID_LINKS = re.compile(
    r"(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/|youtube\.com/shorts/)([^\"&?/\s]{11})",
    re.IGNORECASE,
)

# save the original create_default_context function so we can call it later
create_default_context_orig = ssl.create_default_context


# define a new create_default_context function that sets purpose to ssl.Purpose.SERVER_AUTH
def cdc(*args, **kwargs):
    kwargs["cafile"] = certifi.where()  # Use certifi's CA bundle
    kwargs["purpose"] = ssl.Purpose.SERVER_AUTH
    return create_default_context_orig(*args, **kwargs)


# monkey patching ssl.create_default_context to fix SSL error
ssl.create_default_context = cdc


def timed_lru_cache(_func=None, *, seconds: int = 600, maxsize: int = 128, typed: bool = False):
    """Extension of functools lru_cache with a timeout

    Parameters:
    seconds (int): Timeout in seconds to clear the WHOLE cache, default = 10 minutes
    maxsize (int): Maximum Size of the Cache
    typed (bool): Same value of different type will be a different entry

    """

    def wrapper_cache(f):
        f = lru_cache(maxsize=maxsize, typed=typed)(f)
        f.delta = seconds * 10 ** 9  # fmt: skip
        f.expiration = monotonic_ns() + f.delta

        @wraps(f)
        def wrapped_f(*args, **kwargs):
            if monotonic_ns() >= f.expiration:
                f.cache_clear()
                f.expiration = monotonic_ns() + f.delta
            return f(*args, **kwargs)

        wrapped_f.cache_info = f.cache_info
        wrapped_f.cache_clear = f.cache_clear
        return wrapped_f

    # To allow decorator to be used without arguments
    if _func is None:
        return wrapper_cache

    return wrapper_cache(_func)


@lru_cache(maxsize=1000)
def sanitize_username(username):
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", username):
        username = re.sub(r"[.@!?]", "", username)[:64]
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", username):
        username = "".join(re.findall(r"[a-zA-Z0-9_-]", username))[:64]
    return username


@lru_cache(maxsize=100)
def is_valid_url(url):
    if isinstance(validators.url(url), validators.ValidationError):
        logger.debug(f"Skipping invalid URL: {url}")
        return False

    if re.search(REGEX_LOCAL_LINKS, url):
        logger.info(f"Skipping local URL: {url}")
        return False

    parsed_url = urllib.parse.urlparse(url)
    ipv4_addresses, ipv6_addresses = resolve_hostname(parsed_url.hostname)

    for ip in ipv4_addresses:
        if re.search(REGEX_LOCAL_LINKS, ip):
            logger.info(f"Skipping local IPv4 {ip} from URL: {url}")
            return False

    for ip in ipv6_addresses:
        if re.search(REGEX_LOCAL_LINKS, ip):
            logger.info(f"Skipping local IPv6 {ip} from URL: {url}")
            return False

    return True


def resolve_hostname(hostname):
    addr_info = socket.getaddrinfo(hostname, None)

    ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
    ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]

    return ipv4_addresses, ipv6_addresses


# Bugged sometimes, if anyone with some brain cells can do this proper, I owe you a coffee
def split_message(msg, max_length=4000):
    """
    Split a message based on a maximum character length, ensuring Markdown code blocks
    and their languages are preserved. Avoids unnecessary linebreaks at the end of the last message
    and when closing a code block if the last line is already a newline.

    Args:
    - msg (str): The message to be split.
    - max_length (int, optional): The maximum length of each split message. Defaults to 4000.

    Returns:
    - list of str: The message split into chunks, preserving Markdown code blocks and languages,
                   and avoiding unnecessary linebreaks.
    """
    if len(msg) <= max_length:
        return [msg]

    if len(msg) > 40000:
        raise Exception(f"Response message too long, length: {len(msg)}")

    current_chunk = ""  # Holds the current message chunk
    chunks = []  # Collects all message chunks
    in_code_block = False  # Tracks whether the current line is inside a code block
    code_block_lang = ""  # Keeps the language of the current code block

    # Helper function to add a chunk to the list
    def add_chunk(chunk, in_code, code_lang):
        if in_code:
            # Check if the last line is not just a newline itself
            if not chunk.endswith("\n\n"):
                chunk += "\n"
            chunk += "```"
        chunks.append(chunk)
        if in_code:
            # Start a new code block with the same language
            return f"```{code_lang}\n"
        return ""

    lines = msg.split("\n")
    for i, line in enumerate(lines):
        # Check if this line starts or ends a code block
        if line.startswith("```"):
            if in_code_block:
                # Ending a code block
                in_code_block = False
                # Avoid adding an extra newline if the line is empty
                if current_chunk.endswith("\n"):
                    current_chunk += line
                else:
                    current_chunk += "\n" + line
            else:
                # Starting a new code block, capture the language
                in_code_block = True
                code_block_lang = line[3:].strip()  # Remove the backticks and get the language
                current_chunk += line + "\n"
        else:
            # If adding this line exceeds the max length, we need to split here
            if len(current_chunk) + len(line) + 1 > max_length:
                # Split here, preserve the code block state and language if necessary
                current_chunk = add_chunk(current_chunk, in_code_block, code_block_lang)
                current_chunk += line
                if i < len(lines) - 1:  # Avoid adding a newline at the end of the last line
                    current_chunk += "\n"
            else:
                current_chunk += line
                if i < len(lines) - 1:  # Avoid adding a newline at the end of the last line
                    current_chunk += "\n"

    # Don't forget to add the last chunk
    if current_chunk:
        add_chunk(current_chunk, in_code_block, code_block_lang)

    return chunks


def wrapper_function_call(func, call_input_arguments, *args, **kwargs):
    try:
        result = func(call_input_arguments, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error calling function call function: {str(e)} {traceback.format_exc()}")
        result = f"An error occurred: {str(e)}"
        return result
    return result


def yt_extract_video_id(url):
    match = re.search(YOUTUBE_VALID_LINKS, url)
    return match.group(1) if match else None


def yt_is_valid_url(url):
    # Pattern to match various YouTube URL formats including video IDs
    match = re.search(YOUTUBE_VALID_LINKS, url)
    return bool(match)


def compress_image(image, max_size_mb):
    buffer = BytesIO()
    image.save(buffer, format=image.format, optimize=True)
    return compress_image_data(buffer.getvalue(), max_size_mb)


def compress_image_data(image_data, max_size_mb):
    max_size = max_size_mb * 1024 * 1024
    buffer = BytesIO(image_data)
    image = Image.open(buffer)

    quality = 90

    # Compress the image until the size is within the target
    while len(image_data) > max_size:
        if quality <= 0:
            raise Exception("Image too large, can't compress any further")

        buffer = BytesIO()
        image.save(
            buffer,
            format=image.format,
            optimize=True,
            quality=quality,
        )
        image_data = buffer.getvalue()
        quality -= 5

    return image_data


def resize_image_data(image_data, max_dimensions, max_size_mb):
    image = Image.open(BytesIO(image_data))

    width = max(max_dimensions)
    height = min(max_dimensions)

    image.thumbnail((width, height) if image.width > image.height else (height, width), Image.Resampling.LANCZOS)

    compressed_image_data = compress_image(image, max_size_mb)

    return compressed_image_data
