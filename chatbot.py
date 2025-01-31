import time
import traceback
import json
import threading
import re
import datetime
import concurrent.futures
import base64
import tempfile
import asyncio
from functools import lru_cache
from defusedxml import ElementTree
import yfinance
import pymupdf
import pymupdf4llm
import httpx
from mattermostdriver_patched import Driver
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from openai import OpenAI, NOT_GIVEN
import tiktoken
import nodriver as uc
from helpers import (
    resize_image_data,
    yt_is_valid_url,
    yt_extract_video_id,
    wrapper_function_call,
    split_message,
    is_valid_url,
    sanitize_username,
    timed_lru_cache,
    remove_background_from_image,
)
from config import *  # pylint: disable=W0401 wildcard-import, unused-wildcard-import

logging.basicConfig(level=log_level_root)

logger = logging.getLogger(__name__)
logger.setLevel(log_level)


tools = [
    {
        "type": "function",
        "function": {
            "name": "raw_html_to_image",
            "description": "Generates an image from raw HTML code. You can also pass a URL which will be screenshotted, but only do that if a screenshot is specifically requested (e.g. the user says screenshot this).",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_html_code": {
                        "type": "string",
                        "description": "Full valid HTML code to be opened on a browser and taken a screenshot of. Only one parameter is allowed",
                    },
                    "url": {
                        "type": "string",
                        "description": "Valid URL (with http/https in front) to be opened on a browser and taken a screenshot of. Only one parameter is allowed",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_custom_emoji_by_url",
            "description": "Creates a custom emoji from an image URL, optionally - at the user's request - removes the background of the image. If no emoji name was given, derive a name from the content of the image or use the context",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "Full valid URL to an image to be uploaded. Don't make up URLs, only use one URL the user has provided you",
                    },
                    "emoji_name": {"type": "string", "description": "The desired emoji name"},
                    "remove_background": {
                        "type": "boolean",
                        "description": "Whether to remove the background from the image.",
                        "default": False,
                    },
                },
                "required": ["image_url", "emoji_name"],
            },
        },
    },
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
            "description": "Fetches cryptocurrency data by ID (ex. ethereum) or symbol (ex. BTC), prices in USD",
            "parameters": {
                "type": "object",
                "properties": {
                    "crypto_id": {"type": "string", "description": "The identifier or symbol of the cryptocurrency"}
                },
                "required": ["crypto_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cryptocurrency_data_by_market_cap",
            "description": "Fetches cryptocurrency data for the top N currencies by market cap, prices in USD",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_currencies": {
                        "type": "integer",
                        "description": "The number of top cryptocurrencies to retrieve. Optional",
                        "default": 15,
                        "max": 20,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_ticker_data",
            "description": "Retrieves information about a specified company from the stock market",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol of the company (ex. AAPL)",
                    }
                },
                "required": ["ticker_symbol"],
            },
        },
    },
]

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
        "verify": MATTERMOST_CERT_VERIFY,
        "timeout": mattermost_timeout,
        # "websocket_kw_args": {"ping_interval": None},
    }
)

# Chatbot account username, automatically fetched
CHATBOT_USERNAME = ""
CHATBOT_USERNAME_AT = ""

# Create an AI client instance
ai_client = OpenAI(api_key=api_key, base_url=ai_api_baseurl)

# Used to count tokens, do not modify unless you know what you are doing
model_encoder = tiktoken.encoding_for_model("gpt-4o")

# Create a thread pool with a fixed number of worker threads
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)


def get_system_instructions(initial_time):
    return system_prompt_unformatted.format(current_time=initial_time, CHATBOT_USERNAME=CHATBOT_USERNAME)


@lru_cache(maxsize=1000)
def get_username_from_user_id(user_id):
    try:
        user = driver.users.get_user(user_id)
        return sanitize_username(user["username"])
    except Exception as e:
        logger.error(f"Error retrieving username for user ID {user_id}: {str(e)} {traceback.format_exc()}")
        return f"Unknown_{user_id}"


def send_typing_indicator_loop(user_id, channel_id, parent_id, stop_event):
    """Send a "typing" indicator to show that work is in progress."""
    while not stop_event.is_set():
        try:
            # If full mode is active and we have a parent_id, also send an indicator to the main channel
            # We send this first because I prefer it and there is a slight lag for the second indicator
            if typing_indicator_mode_is_full and parent_id:
                options = {
                    "channel_id": channel_id,
                }

                driver.client.make_request("post", f"/users/{user_id}/typing", options=options)

            options = {"channel_id": channel_id, "parent_id": parent_id}  # id may be substituted with "me"

            driver.client.make_request("post", f"/users/{user_id}/typing", options=options)

            time.sleep(1)
        except Exception as e:
            logger.error(f"Error sending typing indicator: {str(e)} {traceback.format_exc()}")


def handle_typing_indicator(user_id, channel_id, parent_id):
    logger.debug("Starting typing indicator")
    stop_typing_event = threading.Event()
    typing_indicator_thread = threading.Thread(
        target=send_typing_indicator_loop,
        args=(user_id, channel_id, parent_id, stop_typing_event),
    )
    typing_indicator_thread.start()
    return stop_typing_event, typing_indicator_thread


# aaa maybe use general handle generation function that does typing stuff etc so we can save some code
def handle_html_image_generation(raw_html_code, url, channel_id, root_id):
    stop_typing_event = None
    typing_indicator_thread = None

    try:
        logger.info("Starting HTML Image generation")

        # Start the typing indicator as this is a new thread
        stop_typing_event, typing_indicator_thread = handle_typing_indicator(driver.client.userid, channel_id, root_id)

        image_data = uc.loop().run_until_complete(asyncio.wait_for(raw_html_to_image(raw_html_code, url), 30))
        file_id = driver.files.upload_file(
            channel_id=channel_id,
            files={"files": ("image.png", image_data)},
        )[
            "file_infos"
        ][0]["id"]

        # Send the response back to the Mattermost channel as a reply to the thread or as a new thread
        driver.posts.create_post(
            {
                "channel_id": channel_id,
                "message": "_Web preview:_",
                "root_id": root_id,
                "file_ids": [file_id],
            }
        )
    except Exception as e:
        logger.error(f"HTML Image generation error: {str(e)} {traceback.format_exc()}")
        driver.posts.create_post(
            {"channel_id": channel_id, "message": f"HTML Image generation error occurred: {str(e)}", "root_id": root_id}
        )
    finally:
        logger.debug("Stopping typing indicator")

        if stop_typing_event:
            stop_typing_event.set()
        if typing_indicator_thread:
            typing_indicator_thread.join()


def handle_custom_emoji_generation(image_url, emoji_name, remove_background, channel_id, root_id):
    stop_typing_event = None
    typing_indicator_thread = None

    try:
        logger.info(f"Starting Custom emoji generation for emoji name {emoji_name} and image URL {image_url}")

        # Start the typing indicator as this is a new thread
        stop_typing_event, typing_indicator_thread = handle_typing_indicator(driver.client.userid, channel_id, root_id)

        if not is_valid_url(image_url):
            raise Exception("No local/invalid URL allowed for custom emoji generation")

        emoji_name = re.sub(r"[^a-z0-9\-+_]", "", emoji_name.lower())[:64]

        if not emoji_name:
            raise Exception("Invalid emoji name")

        # Refactor the image grab code into one function with the other code that we have
        with httpx.Client() as client:
            # By doing the redirect itself, we might already allow a local request?
            with client.stream("GET", image_url, timeout=4, follow_redirects=True) as response:
                response.raise_for_status()

                final_url = str(response.url)

                if not is_valid_url(final_url):
                    logger.info(f"Skipping local/invalid URL {final_url} after redirection: {image_url}")
                    raise Exception("No local/invalid URL allowed for custom emoji generation")

                content_type = response.headers.get("content-type", "").lower()
                if content_type not in compatible_emoji_image_content_types:
                    raise Exception(f"Unsupported image content type: {content_type}")

                total_size = 0

                image_data = b""
                for chunk in response.iter_bytes():
                    image_data += chunk
                    total_size += len(chunk)
                    if total_size > max_response_size:
                        raise Exception("Image size from the website exceeded the maximum limit for the chatbot")

                if remove_background:
                    logger.debug(f"Removing background of image from URL {image_url}")
                    image_data = remove_background_from_image(image_data)
                    content_type = "image/png"

                image_data = resize_image_data(image_data, mattermost_max_emoji_image_dimensions,
                                               MATTERMOST_MAX_EMOJI_IMAGE_FILE_SIZE, content_type)

                try:
                    driver.emoji.create_custom_emoji(emoji_name, files={"image": image_data})
                except Exception as e:
                    raise Exception(f"Emoji name: {emoji_name}, {str(e)}") from e

                # Send the response back to the Mattermost channel as a reply to the thread or as a new thread
                driver.posts.create_post(
                    {
                        "channel_id": channel_id,
                        "message": f":{emoji_name}:",
                        "root_id": root_id,
                    }
                )
    except Exception as e:
        logger.error(f"Custom emoji generation error: {str(e)} {traceback.format_exc()}")
        driver.posts.create_post(
            {
                "channel_id": channel_id,
                "message": f"Custom emoji generation error occurred: {str(e)}",
                "root_id": root_id,
            }
        )
    finally:
        logger.debug("Stopping typing indicator")

        if stop_typing_event:
            stop_typing_event.set()
        if typing_indicator_thread:
            typing_indicator_thread.join()


def handle_image_generation(prompt, is_raw, channel_id, root_id):
    stop_typing_event = None
    typing_indicator_thread = None

    try:
        logger.info("Querying Image generation API")

        # Start the typing indicator as this is a new thread
        stop_typing_event, typing_indicator_thread = handle_typing_indicator(driver.client.userid, channel_id, root_id)

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

        # Decode the base64-encoded image data, resize and compress if necessary
        decoded_image_data = resize_image_data(base64.b64decode(image_data), mattermost_max_image_dimensions, 10,
                                               AI_MODEL_IMAGE_GENERATION_MIME_TYPE)

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
    except Exception as e:
        logger.error(f"Image generation error: {str(e)} {traceback.format_exc()}")
        driver.posts.create_post(
            {"channel_id": channel_id, "message": f"Image generation error occurred: {str(e)}", "root_id": root_id}
        )
    finally:
        logger.debug("Stopping typing indicator")

        if stop_typing_event:
            stop_typing_event.set()
        if typing_indicator_thread:
            typing_indicator_thread.join()


def process_tool_calls(tool_calls, current_message, channel_id, root_id):
    if len(tool_calls) > 15:
        raise Exception("Too many function calls in the message, maximum is 15")

    tool_messages = []

    for call in tool_calls:
        if call.function.name == "get_stock_ticker_data":
            data = wrapper_function_call(get_stock_ticker_data, call.function.arguments)
            func_response = {
                "tool_call_id": call.id,
                "role": "tool",
                "name": call.function.name,
                "content": str(data),
            }

            tool_messages.append(func_response)
        elif call.function.name == "get_cryptocurrency_data_by_market_cap":
            data = wrapper_function_call(get_cryptocurrency_data_by_market_cap, call.function.arguments)
            func_response = {
                "tool_call_id": call.id,
                "role": "tool",
                "name": call.function.name,
                "content": str(data),
            }

            tool_messages.append(func_response)
        elif call.function.name == "get_cryptocurrency_data_by_id":
            data = wrapper_function_call(get_cryptocurrency_data_by_id, call.function.arguments)
            func_response = {
                "tool_call_id": call.id,
                "role": "tool",
                "name": call.function.name,
                "content": str(data),
            }

            tool_messages.append(func_response)
        elif call.function.name == "get_exchange_rates":
            data = wrapper_function_call(get_exchange_rates, call.function.arguments)
            func_response = {
                "tool_call_id": call.id,
                "role": "tool",
                "name": call.function.name,
                "content": str(data),
            }

            tool_messages.append(func_response)
        elif call.function.name == "generate_image":
            arguments = json.loads(call.function.arguments)
            prompt_is_raw = current_message.startswith("#")
            image_prompt = arguments["prompt"]
            thread_pool.submit(
                handle_image_generation,
                current_message if prompt_is_raw else image_prompt,
                prompt_is_raw,
                channel_id,
                root_id,
            )
        elif call.function.name == "create_custom_emoji_by_url":
            arguments = json.loads(call.function.arguments)
            image_url = arguments["image_url"]
            emoji_name = arguments["emoji_name"]
            remove_background = arguments.get("remove_background", None)

            thread_pool.submit(
                handle_custom_emoji_generation,
                image_url,
                emoji_name,
                remove_background,
                channel_id,
                root_id,
            )
        elif call.function.name == "raw_html_to_image":
            arguments = json.loads(call.function.arguments)
            raw_html_code = arguments.get("raw_html_code", None)
            url = arguments.get("url", None)

            thread_pool.submit(
                handle_html_image_generation,
                raw_html_code,
                url,
                channel_id,
                root_id,
            )
        else:
            logger.error(f"Hallucinated function call: {call.function.name}")

            func_response = {
                "tool_call_id": call.id,
                "role": "tool",
                "name": call.function.name,
                "content": "You hallucinated this function call, it does not exist",
            }

            tool_messages.append(func_response)

    return tool_messages


def handle_text_generation(current_message, messages, channel_id, root_id, initial_time):
    start_time = time.time()

    system_instructions = get_system_instructions(initial_time)

    # Send the messages to the AI API
    response = ai_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "developer", "content": system_instructions}, *messages],
        timeout=timeout,
        temperature=temperature,
        tools=tools if tool_use_enabled else NOT_GIVEN,
        tool_choice="auto" if tool_use_enabled else NOT_GIVEN,  # Let model decide to call the function or not
    )

    end_time = time.time()
    duration = end_time - start_time

    logger.debug(f"AI API response received after {duration:.2f} seconds")

    initial_message_response = response.choices[0].message
    prompt_tokens = response.usage.prompt_tokens
    cached_prompt_tokens = response.usage.prompt_tokens_details.cached_tokens
    completion_tokens = response.usage.completion_tokens

    # Check if tool calls are present in the response
    if initial_message_response.tool_calls:
        logger.debug("Handling tool calls")

        tool_calls = initial_message_response.tool_calls

        tool_messages = process_tool_calls(tool_calls, current_message, channel_id, root_id)

        # If all tool calls were image generation, we do not need to continue here. Refactor this sometime
        image_gen_calls_only = all(
            call.function.name in ("generate_image", "raw_html_to_image", "create_custom_emoji_by_url")
            for call in tool_calls
        )
        if image_gen_calls_only:
            logger.debug("All tool calls were image generation, skipping text generation")
            return

        # Remove all image generation tool calls from the message for API compliance, as we handle images differently
        initial_message_response.tool_calls = [
            call
            for call in tool_calls
            if call.function.name not in ("generate_image", "raw_html_to_image", "create_custom_emoji_by_url")
        ]

        # Requery in case there are new messages from function calls
        if tool_messages:
            logger.debug("Requerying AI API after tool calls")

            # Add the initial response to the messages array as it contains infos about tool calls
            messages.append(initial_message_response)

            messages.extend(tool_messages)

            response = ai_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system_instructions}, *messages],
                timeout=timeout,
                temperature=temperature,
                tools=tools,
                tool_choice="none",
            )

            prompt_tokens += response.usage.prompt_tokens
            cached_prompt_tokens = response.usage.prompt_tokens_details.cached_tokens
            completion_tokens += response.usage.completion_tokens

    response_text = response.choices[0].message.content

    if response_text is None:
        raise Exception("Empty AI response, likely API error or mishandling")

    if response.choices[0].finish_reason == "content_filter":
        logger.debug("Response censored, finish reason: content_filter")
        response_text += "\n**Response censored, finish reason: content_filter**"

    # Split the response into multiple messages if necessary
    response_parts = split_message(response_text)

    # Send each part of the response as a separate message
    for part in response_parts:
        # Send the API response back to the Mattermost channel as a reply to the thread or as a new thread
        driver.posts.create_post({"channel_id": channel_id, "message": part, "root_id": root_id})

    prompt_tokens_cost = 2.5 / 1_000_000 * prompt_tokens - 1.25 * cached_prompt_tokens
    completion_tokens_cost = 10 / 1_000_000 * completion_tokens
    tokens_cost_total = prompt_tokens_cost + completion_tokens_cost
    logger.debug(
        f"Text Token cost: ${tokens_cost_total:.4f} | Input ${prompt_tokens_cost:.4f} ({prompt_tokens}, cached: {cached_prompt_tokens}) + Output ${completion_tokens_cost:.4f} ({completion_tokens})"
    )


def handle_generation(current_message, messages, channel_id, root_id, initial_time):
    try:
        logger.info("Querying AI API")
        handle_text_generation(current_message, messages, channel_id, root_id, initial_time)
    except Exception as e:
        logger.error(f"Text generation error: {str(e)} {traceback.format_exc()}")
        driver.posts.create_post(
            {"channel_id": channel_id, "message": f"Text generation error occurred: {str(e)}", "root_id": root_id}
        )


@timed_lru_cache(seconds=300, maxsize=100)
def get_stock_ticker_data(arguments):
    arguments = json.loads(arguments)
    ticker_symbol = arguments["ticker_symbol"]

    stock = yfinance.Ticker(ticker_symbol)

    stock_data = {
        "info": str(stock.info),
        "calendar": str(stock.calendar),
        "news": str(stock.news),
        "dividends": str(stock.dividends),
        "splits": str(stock.splits),
        "quarterly_financials": str(stock.quarterly_financials),
        "financials": str(stock.financials),
        "cashflow": str(stock.cashflow),
    }

    return stock_data


async def raw_html_to_image(raw_html, url):
    browser = await uc.start(
        browser_executable_path=browser_executable_path, headless=True, browser_args=["--window-size=1920,1080"]
    )

    try:
        final_url = None

        if raw_html:
            encoded_html = base64.b64encode(raw_html.encode("utf-8")).decode("utf-8")
            final_url = f"data:text/html;base64,{encoded_html}"
        elif url:
            if not is_valid_url(url):
                raise Exception(f"Local/invalid URLs are not allowed for screenshotting {url}")
            final_url = url

        if not final_url:
            raise Exception("No URL or raw HTML provided")

        page = await browser.get(final_url)
        await page  # wait for events to be processed
        await browser.wait(3)  # wait some time for more elements

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_screen_path = temp_file.name

        try:
            await page.save_screenshot(filename=temp_screen_path, format="png", full_page=True)
            await page.close()
            with open(temp_screen_path, "rb") as file:
                file_bytes = file.read()
        finally:
            os.remove(temp_screen_path)
    finally:
        browser.stop()  # uc.util.deconstruct_browser() but may affect other instances running at the same time?

    return resize_image_data(file_bytes, mattermost_max_image_dimensions, 10, "image/png")


@timed_lru_cache(seconds=7200, maxsize=100)
def get_exchange_rates(_arguments):
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
def get_cryptocurrency_data_by_market_cap(arguments):
    arguments = json.loads(arguments)
    num_currencies = arguments.get("num_currencies", 15)
    num_currencies = min(num_currencies, 20)  # Limit to 20

    url = "https://api.coingecko.com/api/v3/coins/markets"  # possible alternatives: coincap.io, mobula.io
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": num_currencies,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h,7d",
    }

    with httpx.Client() as client:
        response = client.get(url, timeout=15, params=params)
        response.raise_for_status()

        data = response.json()
        return data


@timed_lru_cache(seconds=180, maxsize=100)
def get_cryptocurrency_data_by_id(arguments):
    arguments = json.loads(arguments)
    crypto_id = arguments["crypto_id"].lower()

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 500,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h,7d",
    }

    with httpx.Client() as client:
        response = client.get(url, timeout=15, params=params)
        response.raise_for_status()

        data = response.json()
        # Filter data to find the cryptocurrency with the matching id or symbol
        matched_crypto = next((item for item in data if crypto_id in (item["id"], item["symbol"])), None)
        if matched_crypto:
            return matched_crypto

        return "No data found for the specified cryptocurrency ID/symbol"


def process_message(event_data):
    post = json.loads(event_data["data"]["post"])
    if should_ignore_post(post):
        return

    current_message, channel_id, sender_name, root_id, post_id, channel_display_name = extract_post_data(
        post, event_data
    )

    stop_typing_event = None
    typing_indicator_thread = None
    chatbot_invoked = False

    try:
        messages = []

        # Chatbot is invoked if it was mentioned, the chatbot has already been invoked in the thread or its a DM
        chatbot_invoked = is_chatbot_invoked(post, post_id, root_id, channel_display_name)

        if chatbot_invoked:
            # Start the typing indicator
            stop_typing_event, typing_indicator_thread = handle_typing_indicator(
                driver.client.userid, channel_id, root_id
            )

            # Retrieve the thread context if there is any
            thread_messages = []

            if root_id:
                thread_messages = get_thread_posts(root_id, post_id)
                root_post = driver.posts.get_post(root_id)
                posted_at = root_post["create_at"]
            else:
                # If we don't have any thread, add our own message to the array
                thread_messages.append((post, sender_name, "user", current_message))
                posted_at = post["create_at"]

            current_time_utc = datetime.datetime.now(datetime.UTC)
            post_time_utc = datetime.datetime.fromtimestamp(posted_at / 1000.0, tz=datetime.UTC)
            initial_time = min(current_time_utc, post_time_utc).strftime("%Y-%m-%d %H:%M:%S.%f")[
                           :-3]  # Gets the UTC time of the root post

            for index, thread_message in enumerate(thread_messages):
                content = {}

                thread_post, thread_sender_name, thread_role, thread_message_text = thread_message

                image_messages = []

                links = re.findall(r"(https?://\S+)", thread_message_text, re.IGNORECASE)  # Allow http and https links
                content["website_data"] = []

                # We don't want to grab URL content from links the assistant sent
                # If keep URL content is disabled, we will skip the URL content code unless its the last message
                is_last_message = index == len(thread_messages) - 1
                if thread_role == "user" and keep_all_url_content or is_last_message:
                    for link in links:
                        website_data = {
                            "url": link,
                        }

                        try:
                            if not is_valid_url(link):
                                raise Exception("Local or invalid link")

                            website_data["url_content"], link_image_messages = request_link_content(link)
                            image_messages.extend(link_image_messages)
                        except Exception as e:
                            logger.error(
                                f"Error extracting content from link {link}: {str(e)} {traceback.format_exc()}"
                            )
                            website_data["error"] = (
                                f"fetching website caused an exception, warn the chatbot user: {str(e)}"
                            )
                        finally:
                            content["website_data"].append(website_data)

                files_text_content, files_image_messages = get_files_content(thread_post)
                image_messages.extend(files_image_messages)

                if files_text_content:
                    content["file_data"] = files_text_content
                if not content["website_data"]:
                    del content["website_data"]

                # We use str() and not JSON.dumps() to avoid the AI replying in (partially) escaped JSON format
                content = f"{str(content)}{thread_message_text}" if content else thread_message_text

                if image_messages:
                    image_messages.append({"type": "text", "text": content})
                    # We force a user role here, as this is an API requirement for images for GPT-4o
                    messages.append({"role": "user", "content": image_messages})  # "name": thread_sender_name in front
                else:
                    messages.append(construct_text_message(thread_sender_name, thread_role, content))

            # If the message is not part of a thread, reply to it to create a new thread
            handle_generation(current_message, messages, channel_id, post_id if not root_id else root_id, initial_time)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)} {traceback.format_exc()}")
        if chatbot_invoked:
            driver.posts.create_post(
                {"channel_id": channel_id, "message": f"Process message error occurred: {str(e)}", "root_id": root_id}
            )
    finally:
        logger.debug("Clearing cache and stopping typing indicator")

        get_raw_thread_posts.cache_clear()  # We clear this cache as it won't be useful for the next message with the current implementation
        if stop_typing_event:
            stop_typing_event.set()
        if typing_indicator_thread:
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
    return {
        # "name": name,
        "role": role,
        "content": [
            {
                "type": "text",
                "text": str(message),
            }
        ],
    }


def construct_image_content_message(content_type, image_data_base64):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{content_type};base64,{image_data_base64}"},
    }


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
    # We directly access the raw message here as we filter the mention earlier
    last_message = post["message"]
    if CHATBOT_USERNAME_AT in last_message:
        return True

    # It is a direct message
    if channel_display_name.startswith("@"):
        return True

    if root_id:
        thread = get_raw_thread_posts(root_id, post_id)

        # Check if the last post in the thread starts with a mention of ANY other bot than the chatbot
        # If so, ignore it, as it is likely a mention for another bot
        if thread:
            match = re.match(r"@(\w+)", last_message)

            if match:
                mentioned_username = match.group(1)

                try:
                    mentioned_user = driver.users.get_user_by_username(mentioned_username)
                    mentioned_user_id = mentioned_user["id"]

                    if mentioned_user_id != driver.client.userid and mentioned_user.get("is_bot", False):
                        logger.debug(
                            "Ignoring post and not checking further if we have been invoked as it is a mention for another bot"
                        )
                        return False
                except Exception as e:
                    logger.debug(f"Could not get user {mentioned_username}: {str(e)}")

        # Check if we have been mentioned in the past or if the chatbot had already replied
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
        image_data_base64 = base64.b64encode(
            resize_image_data(file.content, ai_model_max_vision_image_dimensions, 3, content_type)
        ).decode("utf-8")

        image_messages.append(construct_image_content_message(content_type, image_data_base64))
        return "", image_messages

    if "application/pdf" in content_type:
        return extract_pdf_content(file.content)

    # Return other files simply as string
    return str(file.content), image_messages


def extract_pdf_content(stream):
    pdf_text_content = ""
    image_messages = []

    with pymupdf.open(None, stream, "pdf") as pdf:
        pdf_text_content += pymupdf4llm.to_markdown(pdf, margins=0)

        for page in pdf:
            # Extract images
            for img in page.get_images():
                xref = img[0]
                pdf_base_image = pdf.extract_image(xref)
                pdf_image_extension = pdf_base_image["ext"]
                pdf_image_content_type = f"image/{pdf_image_extension}"
                if pdf_image_content_type not in compatible_image_content_types:
                    continue
                pdf_image_data_base64 = base64.b64encode(
                    resize_image_data(pdf_base_image["image"], ai_model_max_vision_image_dimensions, 3,
                                      pdf_image_content_type)
                ).decode("utf-8")

                image_messages.append(construct_image_content_message(pdf_image_content_type, pdf_image_data_base64))

    return pdf_text_content, image_messages


def get_files_content(post):
    files_text_content_all = {}
    image_messages = []

    try:
        if post.get("metadata"):
            metadata = post["metadata"]
            if metadata.get("files"):
                metadata_files = metadata["files"]

                for file_details in metadata_files:
                    file_name = file_details["name"]
                    files_text_content_all[file_name] = {}

                    try:
                        files_text_content_all[file_name]["file_content"], file_image_messages = get_file_content(
                            json.dumps(file_details)
                        )  # JSON to make it cachable
                        image_messages.extend(file_image_messages)
                    except Exception as e:
                        logger.error(
                            f"Error extracting content from file {file_name}: {str(e)} {traceback.format_exc()}"
                        )
                        files_text_content_all[file_name][
                            "error"
                        ] = f"fetching file content caused an exception, warn the chatbot user: {str(e)}"
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


def yt_get_content(link):
    transcript = yt_get_transcript(link)
    title, description, uploader = yt_get_video_info(link)
    return {
        "youtube_video_details": {
            "title": title,
            "description": description,
            "uploader": uploader,
            "transcript": transcript,
        }
    }


def request_flaresolverr(link):
    payload = {
        "cmd": "request.get",
        "url": link,
        "maxTimeout": 30000,
    }
    response = httpx.post(flaresolverr_endpoint, json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    if data["status"] == "ok":
        # FlareSolverr always returns empty headers/200 status code, as per https://github.com/FlareSolverr/FlareSolverr/issues/1162
        content = data["solution"]["response"]
        return content

    raise Exception(f"FlareSolverr request failed: {data}")


def request_httpx(prev_response, content_type):
    content_chunks = []
    total_size = 0
    for chunk in prev_response.iter_bytes():
        content_chunks.append(chunk)
        total_size += len(chunk)
        if total_size > max_response_size:
            raise Exception("Website size exceeded the maximum limit for the chatbot")
    content = b"".join(content_chunks)
    if content_type.startswith("text/"):
        content = content.decode("utf-8", errors="surrogateescape")
    return content


def request_link_text_content(link, prev_response, content_type):
    raw_content = None
    try:
        # Note: FlareSolverr does not support returning content_type, so after redirections it could possibly be a different type
        if flaresolverr_endpoint:
            raw_content = request_flaresolverr(link)
        else:
            raise Exception("FlareSolverr endpoint not available")
    except Exception as e:
        logger.debug(f"Falling back to HTTPX. Reason: {str(e)}")

    if raw_content and "<title>New Tab</title>" in raw_content:
        logger.debug(
            "Website content is 'New Tab', retrying with HTTPX."
        )  # FlareSolverr issue I haven't figured out yet, happens with direct .CSV files for example
        raw_content = None

    if not raw_content:
        raw_content = request_httpx(prev_response, content_type)

    if content_type.startswith(("text/html", "application/xhtml+xml")):
        soup = BeautifulSoup(raw_content, "html.parser")
        website_content = soup.get_text(" | ", strip=True)

        tokens = len(model_encoder.encode(website_content))

        if tokens > 120000:
            logger.debug("Website text content too large, trying to extract article content only")
            article_texts = [article.get_text(" | ", strip=True) for article in soup.find_all("article")]
            website_content = " | ".join(article_texts)
    else:
        website_content = raw_content.strip()

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

    image_data_base64 = base64.b64encode(
        resize_image_data(image_data, ai_model_max_vision_image_dimensions, 3, content_type)).decode(
        "utf-8"
    )
    return [construct_image_content_message(content_type, image_data_base64)]


@timed_lru_cache(seconds=1800, maxsize=100)
def request_link_content(link):
    if yt_is_valid_url(link):
        return yt_get_content(link), []

    with httpx.Client() as client:
        # By doing the redirect itself, we might already allow a local request?
        with client.stream("GET", link, timeout=4, follow_redirects=True) as response:
            # Raise for bad status codes if we don't have a FlareSolverr endpoint, this can cause issues though if the requested content is not text
            if not flaresolverr_endpoint:
                response.raise_for_status()

            final_url = str(response.url)

            if not is_valid_url(final_url):
                logger.info(f"Skipping local/invalid URL {final_url} after redirection: {link}")
                raise Exception("Local/invalid URL is disallowed")

            content_type = response.headers.get("content-type", "").lower()
            if "image/" in content_type:
                return "", request_link_image_content(response, content_type)

            if "application/pdf" in content_type:
                return request_link_pdf_content(response)

            return request_link_text_content(link, response, content_type), []


def request_link_pdf_content(prev_response):
    total_size = 0

    pdf_data = b""
    for chunk in prev_response.iter_bytes():
        pdf_data += chunk
        total_size += len(chunk)
        if total_size > max_response_size:
            raise Exception("PDF size from the website exceeded the maximum limit for the chatbot")

    return extract_pdf_content(pdf_data)


def main():
    try:
        global CHATBOT_USERNAME, CHATBOT_USERNAME_AT, tools

        # Log in to the Mattermost server
        driver.login()

        CHATBOT_USERNAME = driver.client.username
        CHATBOT_USERNAME_AT = f"@{CHATBOT_USERNAME}"

        try:
            driver.emoji.get_emoji_list(params={"page": 1, "per_page": 1})
        except Exception as e:
            logger.info("Custom emoji permissions not available, removing custom emoji function from tools.")
            logger.debug(str(e))
            tools = [tool for tool in tools if tool["function"]["name"] != "create_custom_emoji_by_url"]

        if not os.path.exists(browser_executable_path) or not os.access(browser_executable_path, os.X_OK):
            logger.error(
                "Chromium binary not found or not executable, removing raw_html_to_image function from tools. This is nothing to worry about if you don't use it."
            )
            tools = [tool for tool in tools if tool["function"]["name"] != "raw_html_to_image"]

        if disable_specific_tool_calls[0]:
            logger.info(f"Disabling tools: {disable_specific_tool_calls}")
            for disable_tool in disable_specific_tool_calls:
                tools = [tool for tool in tools if tool["function"]["name"] != disable_tool]

        system_instructions = get_system_instructions(
            datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        logger.debug(f"SYSTEM PROMPT: {system_instructions}")

        while True:
            try:
                # Initialize the WebSocket connection
                driver.init_websocket(message_handler)
            except Exception as e:
                logger.error(f"Error with WebSocket: {str(e)} {traceback.format_exc()}")
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, logout and exit")
        driver.logout()
    except Exception as e:
        logger.error(f"Error: {str(e)} {traceback.format_exc()}")


if __name__ == "__main__":
    main()
