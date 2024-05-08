import time
import ssl
import traceback
import json
import os
import threading
import re
import datetime
import logging
import concurrent.futures
import base64
from time import monotonic_ns
from functools import lru_cache, wraps
from io import BytesIO
from defusedxml import ElementTree
import certifi

# noinspection PyPackageRequirements
import fitz
import httpx
from PIL import Image
from mattermostdriver.driver import Driver
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from openai import OpenAI
from helpers.pymupdf_rag import to_markdown

log_level_root = os.getenv("LOG_LEVEL_ROOT", "INFO").upper()
logging.basicConfig(level=log_level_root)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# fix ssl certificates for compiled binaries
# https://github.com/pyinstaller/pyinstaller/issues/7229
# https://stackoverflow.com/questions/55736855/how-to-change-the-cafile-argument-in-the-ssl-module-in-python3
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# save the original create_default_context function so we can call it later
create_default_context_orig = ssl.create_default_context


# define a new create_default_context function that sets purpose to ssl.Purpose.SERVER_AUTH
def cdc(*args, **kwargs):
    kwargs["cafile"] = certifi.where()  # Use certifi's CA bundle
    kwargs["purpose"] = ssl.Purpose.SERVER_AUTH
    return create_default_context_orig(*args, **kwargs)


# monkey patching ssl.create_default_context to fix SSL error
ssl.create_default_context = cdc


def timed_lru_cache(
        _func=None, *, seconds: int = 600, maxsize: int = 128, typed: bool = False
):
    """Extension of functools lru_cache with a timeout

    Parameters:
    seconds (int): Timeout in seconds to clear the WHOLE cache, default = 10 minutes
    maxsize (int): Maximum Size of the Cache
    typed (bool): Same value of different type will be a different entry

    """

    def wrapper_cache(f):
        f = lru_cache(maxsize=maxsize, typed=typed)(f)
        f.delta = seconds * 10 ** 9
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


# AI parameters
api_key = os.environ["AI_API_KEY"]
model = os.getenv("AI_MODEL", "gpt-4-turbo")
ai_api_baseurl = os.getenv("AI_API_BASEURL", None)
timeout = int(os.getenv("AI_TIMEOUT", "120"))
max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
temperature = float(os.getenv("TEMPERATURE", "1"))
system_prompt_unformatted = os.getenv(
    "AI_SYSTEM_PROMPT",
    """
You are a helpful assistant used in a Mattermost chat. The current UTC time is {current_time}. 
Whenever users asks you for help you will provide them with succinct answers formatted using Markdown. Do not unnecessarily greet people with their name, 
do not be apologetic. 
For tasks requiring reasoning or math, use the Chain-of-Thought methodology to explain your step-by-step calculations or logic before presenting your answer. 
Messages sent to you might contain XML tags, these XML tags are only sent to YOU exclusively for your own understanding. 
A list of example XML tags that can be sent to you: <youtube_video_details> <url> <title> <description> <uploader> <transcript> <chatbot_error> 
<website_data> <exception> <url_content> <file_data> <name> <file_content>
If a user sends a link, use the extracted content provided in the XML tags, do not assume or make up stories based on the URL alone.
If a user sends a YouTube link, primarily focus on the transcript and do not unnecessarily repeat the title, description or uploader of the video. 
In your answer DO NOT contain the link to the video/website the user just provided to you as the user already knows it, unless the task requires it. 
If your response contains any URLs, make sure to properly escape them using Markdown syntax for display purposes.
If an error occurs, provide the information from the <chatbot_error> tag to the user along with your answer.""",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generates an image based on a textual prompt",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "Text prompt for generating the image"}},
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rates",
            "description": "Retrieve the latest exchange rates from the ECB, base currency: EUR",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cryptocurrency_data_by_id",
            "description": "Fetches cryptocurrency data by ID (ex. ethereum) or symbol (ex. BTC)",
            "parameters": {
                "type": "object",
                "properties": {
                    "crypto_id": {
                        "type": "string",
                        "description": "The identifier or symbol of the cryptocurrency"
                    }
                },
                "required": ["crypto_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cryptocurrency_data_by_market_cap",
            "description": "Fetches cryptocurrency data for the top N currencies by market cap",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_currencies": {
                        "type": "integer",
                        "description": "The number of top cryptocurrencies to retrieve. Optional",
                        "default": 15,
                        "max": 20
                    }
                },
                "required": []
            }
        }
    },
]

image_size = os.getenv("IMAGE_SIZE", "1024x1024")
image_quality = os.getenv("IMAGE_QUALITY", "standard")
image_style = os.getenv("IMAGE_STYLE", "vivid")

# Mattermost server details
mattermost_url = os.environ["MATTERMOST_URL"]
mattermost_scheme = os.getenv("MATTERMOST_SCHEME", "https")
mattermost_port = int(os.getenv("MATTERMOST_PORT", "443"))
mattermost_basepath = os.getenv("MATTERMOST_BASEPATH", "/api/v4")
mattermost_cert_verify = os.getenv("MATTERMOST_CERT_VERIFY", True)  # pylint: disable=invalid-envvar-default
mattermost_token = os.getenv("MATTERMOST_TOKEN", "")
mattermost_ignore_sender_id = os.getenv("MATTERMOST_IGNORE_SENDER_ID", "").split(",")
mattermost_username = os.getenv("MATTERMOST_USERNAME", "")
mattermost_password = os.getenv("MATTERMOST_PASSWORD", "")
mattermost_mfa_token = os.getenv("MATTERMOST_MFA_TOKEN", "")

flaresolverr_endpoint = os.getenv("FLARESOLVERR_ENDPOINT", "")

# Maximum website/file size
max_response_size = 1024 * 1024 * int(os.getenv("MAX_RESPONSE_SIZE_MB", "100"))

keep_all_url_content = os.getenv("KEEP_ALL_URL_CONTENT", "TRUE").upper() == "TRUE"

# For filtering local links
REGEX_LOCAL_LINKS = (
    r"(?:127\.|192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|::1|(?<![:.\w])[fF][cCdD](?![:.\w])|localhost)"
)

# Create a driver instance
driver = Driver(
    {
        "url": mattermost_url,
        "token": mattermost_token,
        "login_id": mattermost_username,
        "password": mattermost_password,
        "mfa_token": mattermost_mfa_token,
        "scheme": mattermost_scheme,
        "port": mattermost_port,
        "basepath": mattermost_basepath,
        "verify": mattermost_cert_verify,
    }
)

# Chatbot account username, automatically fetched
CHATBOT_USERNAME = ""
CHATBOT_USERNAME_AT = ""

# Create an AI client instance
ai_client = OpenAI(api_key=api_key, base_url=ai_api_baseurl)

# Create a thread pool with a fixed number of worker threads
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

compatible_image_content_types = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
]


def get_system_instructions():
    current_time = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return system_prompt_unformatted.format(current_time=current_time, CHATBOT_USERNAME=CHATBOT_USERNAME)


@lru_cache(maxsize=1000)
def sanitize_username(username):
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", username):
        username = re.sub(r"[.@!?]", "", username)[:64]
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", username):
        username = "".join(re.findall(r"[a-zA-Z0-9_-]", username))[:64]
    return username


@lru_cache(maxsize=1000)
def get_username_from_user_id(user_id):
    try:
        user = driver.users.get_user(user_id)
        return sanitize_username(user["username"])
    except Exception as e:
        logger.error(f"Error retrieving username for user ID {user_id}: {str(e)} {traceback.format_exc()}")
        return f"Unknown_{user_id}"


def send_typing_indicator(user_id, channel_id, parent_id):
    """Send a "typing" indicator to show that work is in progress."""
    options = {
        "channel_id": channel_id,
        # "parent_id": parent_id  # somehow bugged/doesnt work
    }
    driver.client.make_request("post", f"/users/{user_id}/typing", options=options)  # id may be substituted with "me"


def send_typing_indicator_loop(user_id, channel_id, parent_id, stop_event):
    while not stop_event.is_set():
        try:
            send_typing_indicator(user_id, channel_id, parent_id)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error sending busy indicator: {str(e)} {traceback.format_exc()}")


def handle_typing_indicator(user_id, channel_id, parent_id):
    stop_typing_event = threading.Event()
    typing_indicator_thread = threading.Thread(
        target=send_typing_indicator_loop,
        args=(user_id, channel_id, parent_id, stop_typing_event),
    )
    typing_indicator_thread.start()
    return stop_typing_event, typing_indicator_thread


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


def handle_image_generation(prompt, is_raw, channel_id, root_id):
    if is_raw:
        # Removing a leading '#' and any whitespace following it
        prompt = re.sub(r"^#(\s*)", "", prompt)
        prompt = f"I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: {prompt}"

    response = ai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=image_size,  # type: ignore
        quality=image_quality,  # type: ignore
        style=image_style,  # type: ignore
        n=1,
        response_format="b64_json",
        timeout=timeout,
    )

    # Extract the base64-encoded image data from the response
    image_data = response.data[0].b64_json
    revised_prompt = response.data[0].revised_prompt

    # Decode the base64-encoded image data
    decoded_image_data = base64.b64decode(image_data)

    file_id = driver.files.upload_file(
        channel_id=channel_id,
        files={"files": ("image.png", decoded_image_data)},
    )[
        "file_infos"
    ][0]["id"]

    # Send the API response back to the Mattermost channel as a reply to the thread or as a new thread
    driver.posts.create_post(
        {
            "channel_id": channel_id,
            "message": f"_{revised_prompt}_",
            "root_id": root_id,
            "file_ids": [file_id],
        }
    )


def handle_text_generation(current_message, messages, channel_id, root_id):
    # Send the messages to the AI API
    response = ai_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": get_system_instructions()}, *messages],
        timeout=timeout,
        temperature=temperature,
        tools=tools,
        tool_choice="auto",  # Let model decide to call the function or not
    )

    initial_message_response = response.choices[0].message

    tool_messages = []

    # Check if tool calls are present in the response
    if initial_message_response.tool_calls:
        tool_calls = initial_message_response.tool_calls
        prompt_is_raw = current_message.startswith("#")
        for index, call in enumerate(tool_calls):
            if index >= 15:  # Limit the number of function calls
                raise Exception("Maximum amount of function calls reached")

            if call.function.name == "get_cryptocurrency_data_by_market_cap":
                arguments = json.loads(call.function.arguments)
                num_currencies = arguments["num_currencies"] if "num_currencies" in arguments else 15
                crypto_data = get_cryptocurrency_data_by_market_cap(num_currencies)
                func_response = {
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": call.function.name,
                    "content": json.dumps(crypto_data),
                }

                tool_messages.append(func_response)
            elif call.function.name == "get_cryptocurrency_data_by_id":
                arguments = json.loads(call.function.arguments)
                crypto_id = arguments["crypto_id"]
                crypto_data = get_cryptocurrency_data_by_id(crypto_id)
                func_response = {
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": call.function.name,
                    "content": json.dumps(crypto_data),
                }

                tool_messages.append(func_response)
            elif call.function.name == "get_exchange_rates":
                exchange_rates = get_exchange_rates()
                func_response = {
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": call.function.name,
                    "content": json.dumps(exchange_rates),
                }

                tool_messages.append(func_response)
            elif call.function.name == "generate_image":
                arguments = json.loads(call.function.arguments)
                image_prompt = arguments["prompt"]
                handle_image_generation(
                    current_message if prompt_is_raw else image_prompt, prompt_is_raw, channel_id, root_id
                )

        # If all tool calls were image generation, we do not need to continue here. Refactor this sometime
        image_gen_calls_only = all(call.function.name == "generate_image" for call in tool_calls)
        if image_gen_calls_only:
            return

        # Remove all generate_image tool calls from the message for API compliance, as we handle images differently
        response.choices[0].message.tool_calls = [call for call in tool_calls if call.function.name != "generate_image"]

    # Requery in case there are new messages from function calls
    if tool_messages:
        # Add the initial response to the messages array as it contains infos about tool calls
        messages.append(initial_message_response)

        messages.extend(tool_messages)

        response = ai_client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": get_system_instructions()}, *messages],
            timeout=timeout,
            temperature=temperature,
            tools=tools,
            tool_choice="none",
        )

    response_text = response.choices[0].message.content

    if response_text is None:
        raise Exception("Empty AI response, likely API error or mishandling")

    # Split the response into multiple messages if necessary
    response_parts = split_message(response_text)

    # Send each part of the response as a separate message
    for part in response_parts:
        # Send the API response back to the Mattermost channel as a reply to the thread or as a new thread
        driver.posts.create_post({"channel_id": channel_id, "message": part, "root_id": root_id})


def handle_generation(current_message, messages, channel_id, root_id):
    try:
        logger.info("Querying AI API")
        handle_text_generation(current_message, messages, channel_id, root_id)
    except Exception as e:
        logger.error(f"Error: {str(e)} {traceback.format_exc()}")
        driver.posts.create_post({"channel_id": channel_id, "message": f"Error occurred: {str(e)}", "root_id": root_id})


# Function to fetch and parse EUR exchange rates
@timed_lru_cache(seconds=7200, maxsize=100)
def get_exchange_rates():
    ecb_url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"

    with httpx.Client() as client:
        response = client.get(ecb_url, timeout=4)
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)
        namespace = {
            "gesmes": "http://www.gesmes.org/xml/2002-08-01",
            "ecb": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref",
        }

        rates = root.find(".//ecb:Cube/ecb:Cube", namespaces=namespace)
        exchange_rates = {"base_currency": "EUR"}
        for rate in rates.findall("ecb:Cube", namespaces=namespace):
            exchange_rates[rate.get("currency")] = rate.get("rate")

        return exchange_rates


@timed_lru_cache(seconds=180, maxsize=100)
def get_cryptocurrency_data_by_market_cap(num_currencies):
    num_currencies = min(num_currencies, 20)  # Limit to 20

    url = 'https://api.coingecko.com/api/v3/coins/markets'  # possible alternatives: coincap.io, mobula.io
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': num_currencies,
        'page': 1,
        'sparkline': 'false',
        'price_change_percentage': '24h,7d'
    }

    with httpx.Client() as client:
        response = client.get(url, timeout=15, params=params)
        response.raise_for_status()

        data = response.json()
        return data


@timed_lru_cache(seconds=180, maxsize=100)
def get_cryptocurrency_data_by_id(crypto_id):
    crypto_id = crypto_id.lower()

    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 500,
        'page': 1,
        'sparkline': 'false',
        'price_change_percentage': '24h,7d'
    }

    with httpx.Client() as client:
        response = client.get(url, timeout=15, params=params)
        response.raise_for_status()

        data = response.json()
        # Filter data to find the cryptocurrency with the matching id or symbol
        matched_crypto = next((item for item in data if crypto_id in (item['id'], item['symbol'])), None)
        if matched_crypto:
            return matched_crypto

        return {"error": "No data found for the specified cryptocurrency ID/symbol."}


def process_message(event_data):
    post = json.loads(event_data["data"]["post"])
    if should_ignore_post(post):
        return

    current_message, channel_id, sender_name, root_id, post_id, channel_display_name = extract_post_data(
        post, event_data
    )

    stop_typing_event = None
    typing_indicator_thread = None

    try:
        messages = []

        # Chatbot is invoked if it was mentioned, the chatbot has already been invoked in the thread or its a DM
        if is_chatbot_invoked(post, post_id, root_id, channel_display_name):
            # Start the typing indicator
            stop_typing_event, typing_indicator_thread = handle_typing_indicator(
                driver.client.userid, channel_id, root_id
            )

            # Retrieve the thread context if there is any
            thread_messages = []

            if root_id:
                thread_messages = get_thread_posts(root_id, post_id)

            # If we don't have any thread, add our own message to the array
            if not root_id:
                thread_messages.append((post, sender_name, "user", current_message))

            for index, thread_message in enumerate(thread_messages):
                thread_post, thread_sender_name, thread_role, thread_message_text = thread_message

                # We don't want to extract information from links the assistant sent
                if thread_role == "assistant":
                    messages.append(construct_text_message(thread_sender_name, thread_role, thread_message_text))
                    continue

                # If keep content is disabled, we will skip the remaining code to grab content unless its the last message
                is_last_message = index == len(thread_messages) - 1
                if not keep_all_url_content and not is_last_message:
                    messages.append(construct_text_message(thread_sender_name, "user", thread_message_text))
                    continue

                links = re.findall(r"(https?://\S+)", thread_message_text, re.IGNORECASE)  # Allow http and https links
                website_content_all = ""
                image_messages = []

                for link in links:
                    if re.search(REGEX_LOCAL_LINKS, link):
                        logger.info(f"Skipping local URL: {link}")
                        continue

                    website_content = ""

                    try:
                        website_content, link_image_messages = request_link_content(link)
                        image_messages.extend(link_image_messages)
                    except Exception as e:
                        logger.error(f"Error extracting content from link {link}: {str(e)} {traceback.format_exc()}")
                        website_content = f"<chatbot_error>fetching website caused an exception, warn the chatbot user<exception>{str(e)}</exception></chatbot_error>"
                    finally:
                        if website_content:
                            website_content_all += f"<website_data><url>{link}</url><url_content>{website_content}</url_content></website_data>"

                files_text_content, files_image_messages = get_files_content(thread_post)
                image_messages.extend(files_image_messages)

                content = f"{files_text_content}{website_content_all}{thread_message_text}"

                if image_messages:
                    image_messages.append({"type": "text", "text": content})
                    messages.append({"name": thread_sender_name, "role": "user", "content": image_messages})
                else:
                    messages.append(construct_text_message(thread_sender_name, "user", content))

            # If the message is not part of a thread, reply to it to create a new thread
            handle_generation(current_message, messages, channel_id, post_id if not root_id else root_id)
    except Exception as e:
        logger.error(f"Error inner message handler: {str(e)} {traceback.format_exc()}")
    finally:
        get_raw_thread_posts.cache_clear()
        if stop_typing_event is not None:
            stop_typing_event.set()
        if typing_indicator_thread is not None:
            typing_indicator_thread.join()


def should_ignore_post(post):
    sender_id = post["user_id"]

    # Ignore own posts
    if sender_id == driver.client.userid:
        return True

    if sender_id in mattermost_ignore_sender_id:
        logger.debug("Ignoring post from an ignored sender ID")
        return True

    if post.get("props", {}).get("from_bot") == "true":
        logger.debug("Ignoring post from a bot")
        return True

    return False


def extract_post_data(post, event_data):
    # Remove the "@chatbot" mention from the message
    message = post["message"].replace(CHATBOT_USERNAME_AT, "").strip()
    channel_id = post["channel_id"]
    sender_name = sanitize_username(event_data["data"]["sender_name"])
    root_id = post["root_id"]
    post_id = post["id"]
    channel_display_name = event_data["data"]["channel_display_name"]
    return message, channel_id, sender_name, root_id, post_id, channel_display_name


def construct_text_message(name, role, message):
    message = {
        "name": name,
        "role": role,
        "content": [
            {
                "type": "text",
                "text": f"{message}",
            }
        ],
    }

    return message


def construct_image_content_message(content_type, image_data_base64):
    message = {
        "type": "image_url",
        "image_url": {"url": f"data:{content_type};base64,{image_data_base64}"},
    }

    return message


# We pass post_id here so cache contains results for the most recent message
@lru_cache(maxsize=100)
def get_raw_thread_posts(root_id, _post_id):
    return driver.posts.get_thread(root_id)


def get_thread_posts(root_id, post_id):
    messages = []
    thread = get_raw_thread_posts(root_id, post_id)

    # Sort the thread posts based on their create_at timestamp as the "order" prop is not suitable for this
    sorted_posts = sorted(thread["posts"].values(), key=lambda x: x["create_at"])
    for thread_post in sorted_posts:
        thread_sender_name = get_username_from_user_id(thread_post["user_id"])
        thread_message = thread_post["message"].replace(CHATBOT_USERNAME_AT, "").strip()
        role = "assistant" if thread_post["user_id"] == driver.client.userid else "user"
        messages.append((thread_post, thread_sender_name, role, thread_message))
        if thread_post["id"] == post_id:
            break  # To prevent it answering a different newer post that we might have occurred during our processing

    return messages


def is_chatbot_invoked(post, post_id, root_id, channel_display_name):
    # We directly access the message here as we filter the mention earlier
    if CHATBOT_USERNAME_AT in post["message"]:
        return True

    # It is a direct message
    if channel_display_name.startswith("@"):
        return True

    if root_id:
        thread = get_raw_thread_posts(root_id, post_id)

        for thread_post in thread["posts"].values():
            if thread_post["user_id"] == driver.client.userid:
                return True

            # Needed when you mention the chatbot and send a fast message afterward
            if CHATBOT_USERNAME_AT in thread_post["message"]:
                return True

    return False


@lru_cache(maxsize=100)
def get_file_content(file_details_json):
    file_details = json.loads(file_details_json)
    file_id = file_details["id"]
    file_size = file_details["size"]
    content_type = file_details["mime_type"].lower()
    image_messages = []

    if file_size / (1024**2) > max_response_size:
        raise Exception("File size exceeded the maximum limit for the chatbot")

    file = driver.files.get_file(file_id)
    if content_type.startswith("image/"):
        if content_type not in compatible_image_content_types:
            raise Exception(f"Unsupported image content type: {content_type}")
        image_data_base64 = process_image(file.content)
        image_messages.append(construct_image_content_message(content_type, image_data_base64))
        return "", image_messages

    if content_type == "application/pdf":
        return extract_pdf_content(file.content)

    return file.content, image_messages


def extract_pdf_content(stream):
    pdf_text_content = ""
    image_messages = []

    with fitz.open(None, stream, "pdf") as pdf:
        pdf_text_content += to_markdown(pdf).strip()

        for page in pdf:
            # Extract images
            for img in page.get_images():
                xref = img[0]
                pdf_base_image = pdf.extract_image(xref)
                pdf_image_extension = pdf_base_image["ext"]
                pdf_image_content_type = f"image/{pdf_image_extension}"
                if pdf_image_content_type not in compatible_image_content_types:
                    continue
                pdf_image_data_base64 = process_image(pdf_base_image["image"])

                image_messages.append(construct_image_content_message(pdf_image_content_type, pdf_image_data_base64))

    return pdf_text_content, image_messages


def get_files_content(post):
    files_text_content_all = ""
    image_messages = []

    try:
        if "metadata" in post and post["metadata"]:
            metadata = post["metadata"]
            if "files" in metadata and metadata["files"]:
                metadata_files = metadata["files"]
                files_text_content_all = ""

                for file_details in metadata_files:
                    file_name = file_details["name"]
                    file_text_content = ""
                    try:
                        file_text_content, file_image_messages = get_file_content(
                            json.dumps(file_details)
                        )  # JSON to make it cachable
                        image_messages.extend(file_image_messages)
                    except Exception as e:
                        logger.error(
                            f"Error extracting content from file {file_name}: {str(e)} {traceback.format_exc()}"
                        )
                        file_text_content = f"<chatbot_error>fetching file content caused an exception, warn the chatbot user<exception>{str(e)}</exception></chatbot_error>"
                    finally:
                        if file_text_content:
                            files_text_content_all += f"<file_data><name>{file_name}</name><file_content>{file_text_content}</file_content></file_data>"
    except Exception as e:
        logger.error(f"Error get_files_content: {str(e)} {traceback.format_exc()}")

    return files_text_content_all, image_messages


async def message_handler(event):
    try:
        event_data = json.loads(event)
        logger.debug(f"Received event: {event_data}")
        if event_data.get("event") == "hello":
            logger.info("WebSocket connection established.")
        elif event_data.get("event") == "posted":
            # Submit the task to the thread pool. We do this because Mattermostdriver-async is outdated
            thread_pool.submit(process_message, event_data)
        else:
            # Handle other events
            pass
    except json.JSONDecodeError:
        logger.error(f"Failed to parse event as JSON: {event} {traceback.format_exc()}")
    except Exception as e:
        logger.error(f"Error message_handler: {str(e)} {traceback.format_exc()}")


def yt_find_preferred_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Define the preferred order of transcript types and languages
    preferred_order = [
        ("manual", "en"),
        ("manual", None),
        ("generated", "en"),
        ("generated", None),
    ]

    # Convert the TranscriptList to a regular list
    transcripts = list(transcript_list)

    # Sort the transcripts based on the preferred order
    transcripts.sort(
        key=lambda t: (
            preferred_order.index((t.is_generated, t.language_code))
            if (t.is_generated, t.language_code) in preferred_order
            else len(preferred_order)
        )
    )

    # Return the first transcript in the sorted list
    return transcripts[0] if transcripts else None


def yt_extract_video_id(url):
    pattern = (
        r"(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/|youtube\.com/shorts/)([^\"&?/\s]{11})"
    )
    match = re.search(pattern, url)
    return match.group(1) if match else None


def yt_get_transcript(url):
    video_id = yt_extract_video_id(url)
    preferred_transcript = yt_find_preferred_transcript(video_id)

    if preferred_transcript:
        transcript = preferred_transcript.fetch()
        return str(transcript)

    raise Exception("Error getting the YouTube transcript")


def yt_get_video_info(url):
    ydl_opts = {
        "quiet": True,
        # 'no_warnings': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        title = info["title"]
        description = info["description"]
        uploader = info["uploader"]

        return title, description, uploader


def yt_is_valid_url(url):
    # Pattern to match various YouTube URL formats including video IDs
    pattern = (
        r"(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/|youtube\.com/shorts/)([^\"&?/\s]{11})"
    )
    match = re.search(pattern, url)
    return bool(match)


def yt_get_content(link):
    transcript = yt_get_transcript(link)
    title, description, uploader = yt_get_video_info(link)
    return f"""
    <youtube_video_details>
        <title>{title}</title>
        <description>{description}</description>
        <uploader>{uploader}</uploader>
        <transcript>{transcript}</transcript>
    </youtube_video_details>
    """


def request_flaresolverr(link):
    payload = {
        "cmd": "request.get",
        "url": link,
        "maxTimeout": 30000,
    }
    response = httpx.post(flaresolverr_endpoint, json=payload)
    response.raise_for_status()
    data = response.json()

    if data["status"] == "ok":
        content = data["solution"]["response"]
        return content

    raise Exception(f"FlareSolverr request failed: {data}")


def request_httpx(prev_response):
    content_chunks = []
    total_size = 0
    for chunk in prev_response.iter_bytes():
        content_chunks.append(chunk)
        total_size += len(chunk)
        if total_size > max_response_size:
            raise Exception("Website size exceeded the maximum limit for the chatbot")
    return b"".join(content_chunks)


def request_link_text_content(link, prev_response):
    raw_content = None
    try:
        if flaresolverr_endpoint:
            raw_content = request_flaresolverr(link)
        else:
            raise Exception("FlareSolverr endpoint not available")
    except Exception as e:
        logger.debug(f"Falling back to HTTPX. Reason: {str(e)}")

    if not raw_content:
        raw_content = request_httpx(prev_response)

    soup = BeautifulSoup(raw_content, "html.parser")
    website_content = soup.get_text(" | ", strip=True)

    if not website_content:
        raise Exception("No text content found on website")

    return website_content


def request_link_image_content(prev_response, content_type):
    total_size = 0

    # Check for compatible content types
    if content_type not in compatible_image_content_types:
        raise Exception(f"Unsupported image content type: {content_type}")

    # Handle image content from link
    image_data = b""
    for chunk in prev_response.iter_bytes():
        image_data += chunk
        total_size += len(chunk)
        if total_size > max_response_size:
            raise Exception("Image size from the website exceeded the maximum limit for the chatbot")

    image_data_base64 = process_image(image_data)
    return [construct_image_content_message(content_type, image_data_base64)]


@timed_lru_cache(seconds=1800, maxsize=100)
def request_link_content(link):
    if yt_is_valid_url(link):
        return yt_get_content(link), []

    with httpx.Client() as client:
        # By doing the redirect itself, we might already allow a local request?
        with client.stream("GET", link, timeout=4, follow_redirects=True) as response:
            final_url = str(response.url)

            if re.search(REGEX_LOCAL_LINKS, final_url):
                logger.info(f"Skipping local URL after redirection: {final_url}")
                raise Exception("Local URL is disallowed")

            content_type = response.headers.get("content-type", "").lower()
            if "image/" in content_type:
                return "", request_link_image_content(response, content_type)

            if "application/pdf" in content_type:
                return request_link_pdf_content(response)

            return request_link_text_content(link, response), []


def request_link_pdf_content(prev_response):
    total_size = 0

    pdf_data = b""
    for chunk in prev_response.iter_bytes():
        pdf_data += chunk
        total_size += len(chunk)
        if total_size > max_response_size:
            raise Exception("PDF size from the website exceeded the maximum limit for the chatbot")

    return extract_pdf_content(pdf_data)


def process_image(image_data):
    # Open the image using Pillow
    image = Image.open(BytesIO(image_data))

    # Calculate the aspect ratio of the image
    width, height = image.size
    aspect_ratio = width / height

    # Define the supported aspect ratios and their corresponding dimensions
    supported_ratios = [
        (1, 1, 1092, 1092),
        (0.75, 3 / 4, 951, 1268),
        (0.67, 2 / 3, 896, 1344),
        (0.56, 9 / 16, 819, 1456),
        (0.5, 1 / 2, 784, 1568),
    ]

    # Find the closest supported aspect ratio
    # pylint: disable=cell-var-from-loop
    closest_ratio = min(
        supported_ratios,
        key=lambda x: abs(x[0] - aspect_ratio),
    )
    # Determine the target dimensions based on the image orientation
    if width >= height:
        # Landscape orientation
        target_width, target_height = (
            closest_ratio[2],
            closest_ratio[3],
        )
    else:
        # Portrait orientation
        target_width, target_height = (
            closest_ratio[3],
            closest_ratio[2],
        )

    # Resize the image to the target dimensions
    resized_image = image.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    )

    # Save the resized image to a BytesIO object
    buffer = BytesIO()
    resized_image.save(buffer, format=image.format, optimize=True)
    resized_image_data = buffer.getvalue()

    # Check if the resized image size exceeds 3MB
    quality = 90
    while len(resized_image_data) > 3 * 1024 * 1024:
        if quality <= 0:
            raise Exception("Image too large, can't compress any further")

        # Reduce the image quality until the size is within the target
        buffer = BytesIO()
        resized_image.save(
            buffer,
            format=image.format,
            optimize=True,
            quality=quality,
        )
        resized_image_data = buffer.getvalue()
        quality -= 5

    return base64.b64encode(resized_image_data).decode("utf-8")


def main():
    try:
        # Log in to the Mattermost server
        driver.login()
        global CHATBOT_USERNAME, CHATBOT_USERNAME_AT
        CHATBOT_USERNAME = driver.client.username
        CHATBOT_USERNAME_AT = f"@{CHATBOT_USERNAME}"

        logger.debug(f"SYSTEM PROMPT: {get_system_instructions()}")

        # Initialize the WebSocket connection
        while True:
            try:
                # Initialize the WebSocket connection
                driver.init_websocket(message_handler)
            except Exception as e:
                logger.error(f"Error initializing WebSocket: {str(e)} {traceback.format_exc()}")
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, logout and exit")
        driver.logout()
    except Exception as e:
        logger.error(f"Error: {str(e)} {traceback.format_exc()}")


if __name__ == "__main__":
    main()
