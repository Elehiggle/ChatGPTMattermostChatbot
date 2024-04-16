import time
from mattermostdriver.driver import Driver
import ssl
import certifi
import traceback
import json
import os
import string
import random
import threading
import re
import datetime
import logging
from bs4 import BeautifulSoup
import concurrent.futures
import base64
import httpx
from io import BytesIO
from PIL import Image
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# save the original create_default_context function so we can call it later
create_default_context_orig = ssl.create_default_context


# define a new create_default_context function that sets purpose to ssl.Purpose.SERVER_AUTH
def cdc(*args, **kwargs):
    kwargs["cafile"] = certifi.where()  # Use certifi's CA bundle
    kwargs["purpose"] = ssl.Purpose.SERVER_AUTH
    return create_default_context_orig(*args, **kwargs)


# monkey patching ssl.create_default_context to fix SSL error
ssl.create_default_context = cdc

# AI parameters
api_key = os.environ["AI_API_KEY"]
model = os.getenv("AI_MODEL", "gpt-4-turbo")
ai_api_baseurl = os.getenv("AI_API_BASEURL", None)
timeout = int(os.getenv("AI_TIMEOUT", "120"))
max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
temperature = float(os.getenv("TEMPERATURE", "1"))
system_prompt_unformatted = os.getenv(
    "AI_SYSTEM_PROMPT",
    "You are a helpful assistant. The current UTC time is {current_time}. Whenever users asks you for help you will provide them with succinct answers formatted using Markdown; do not unnecessarily greet people with their name. Do not be apologetic. You know the user's name as it is provided within [CONTEXT, from:username] bracket at the beginning of a user-role message. Never add any CONTEXT bracket to your replies (eg. [CONTEXT, from:{chatbot_username}]). The CONTEXT bracket may also include grabbed text from a website if a user adds a link to his question. Users may post YouTube links for which you will get the transcript, in your answer DO NOT don't contain the link to the video the user just provided to you as he already knows it.",
)

image_size = os.getenv("IMAGE_SIZE", "1024x1024")
image_quality = os.getenv("IMAGE_QUALITY", "standard")
image_style = os.getenv("IMAGE_STYLE", "vivid")

# Mattermost server details
mattermost_url = os.environ["MATTERMOST_URL"]
mattermost_scheme = os.getenv("MATTERMOST_SCHEME", "https")
mattermost_port = int(os.getenv("MATTERMOST_PORT", "443"))
mattermost_basepath = os.getenv("MATTERMOST_BASEPATH", "/api/v4")
mattermost_cert_verify = os.getenv("MATTERMOST_CERT_VERIFY", True)
mattermost_token = os.getenv("MATTERMOST_TOKEN", "")
mattermost_ignore_sender_id = os.getenv("MATTERMOST_IGNORE_SENDER_ID", "")
mattermost_username = os.getenv("MATTERMOST_USERNAME", "")
mattermost_password = os.getenv("MATTERMOST_PASSWORD", "")
mattermost_mfa_token = os.getenv("MATTERMOST_MFA_TOKEN", "")

flaresolverr_endpoint = os.getenv("FLARESOLVERR_ENDPOINT", "")

# Maximum website size
max_response_size = 1024 * 1024 * int(os.getenv("MAX_RESPONSE_SIZE_MB", "100"))

# For filtering local links
regex_local_links = r"(?:127\.|192\.168\.|10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.|::1|[fF][cCdD]|localhost)"

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
chatbot_username = ""
chatbot_username_at = ""

# Create an AI client instance
ai_client = OpenAI(api_key=api_key, base_url=ai_api_baseurl)

# Create a thread pool with a fixed number of worker threads
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)


def get_system_instructions():
    current_time = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[
        :-3
    ]
    return system_prompt_unformatted.format(
        current_time=current_time, chatbot_username=chatbot_username
    )


def sanitize_username(username):
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", username):
        username = re.sub(r"[.@!?]", "", username)[:64]
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", username):
        username = "".join(re.findall(r"[a-zA-Z0-9_-]", username))[:64]
    return username


def get_username_from_user_id(user_id):
    try:
        user = driver.users.get_user(user_id)
        return sanitize_username(user["username"])
    except Exception as e:
        logger.error(
            f"Error retrieving username for user ID {user_id}: {str(e)} {traceback.format_exc()}"
        )
        return f"Unknown_{user_id}"


def ensure_alternating_roles(messages):
    updated_messages = []
    for i in range(len(messages)):
        if i > 0 and messages[i]["role"] == messages[i - 1]["role"]:
            updated_messages.append(
                {
                    "role": "assistant" if messages[i]["role"] == "user" else "user",
                    "content": [
                        {"type": "text", "text": "[CONTEXT, acknowledged post]"}
                    ],
                }
            )
        updated_messages.append(messages[i])
    return updated_messages


def send_typing_indicator(user_id, channel_id, parent_id):
    """Send a "typing" indicator to show that work is in progress."""
    options = {
        "channel_id": channel_id,
        # "parent_id": parent_id  # somehow bugged/doesnt work
    }
    driver.client.make_request("post", f"/users/{user_id}/typing", options=options)


def send_typing_indicator_loop(user_id, channel_id, parent_id, stop_event):
    while not stop_event.is_set():
        try:
            send_typing_indicator(user_id, channel_id, parent_id)
            time.sleep(2)
        except Exception as e:
            logging.error(
                f"Error sending busy indicator: {str(e)} {traceback.format_exc()}"
            )


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
        raise Exception("Message too long.")

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
                code_block_lang = line[
                    3:
                ].strip()  # Remove the backticks and get the language
                current_chunk += line + "\n"
        else:
            # If adding this line exceeds the max length, we need to split here
            if len(current_chunk) + len(line) + 1 > max_length:
                # Split here, preserve the code block state and language if necessary
                current_chunk = add_chunk(current_chunk, in_code_block, code_block_lang)
                current_chunk += line
                if (
                    i < len(lines) - 1
                ):  # Avoid adding a newline at the end of the last line
                    current_chunk += "\n"
            else:
                current_chunk += line
                if (
                    i < len(lines) - 1
                ):  # Avoid adding a newline at the end of the last line
                    current_chunk += "\n"

    # Don't forget to add the last chunk
    if current_chunk:
        add_chunk(current_chunk, in_code_block, code_block_lang)

    return chunks


def handle_image_generation(
    last_message, messages, channel_id, root_id, sender_name, links
):
    random_file_name = None
    try:
        # Check if "#draw " is present in any case and replace the first occurrence
        if re.search(r"#draw ", last_message, re.IGNORECASE):
            last_message = re.sub(
                r"#draw ", "", last_message, count=1, flags=re.IGNORECASE
            )
            last_message = f"I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: {last_message}"
        else:
            # If "#draw " is not found, replace the first occurrence of "draw " in any case
            last_message = re.sub(
                r"\bdraw ", "", last_message, count=1, flags=re.IGNORECASE
            )

        response = ai_client.images.generate(
            model="dall-e-3",
            prompt=last_message,
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

        # Generate a random file name
        random_file_name = (
            "".join(random.choices(string.ascii_letters + string.digits, k=20)) + ".png"
        )
        # Save the image data to a local file
        with open(random_file_name, "wb") as file:
            file.write(decoded_image_data)

        file_id = driver.files.upload_file(
            channel_id=channel_id,
            files={"files": (random_file_name, open(random_file_name, "rb"))},
        )["file_infos"][0]["id"]

        # Send the API response back to the Mattermost channel as a reply to the thread or as a new thread
        driver.posts.create_post(
            {
                "channel_id": channel_id,
                "message": f"_{revised_prompt}_",
                "root_id": root_id,
                "file_ids": [file_id],
            }
        )

    finally:
        # Delete the image file if it was created
        if random_file_name is not None and os.path.exists(random_file_name):
            os.remove(random_file_name)


def handle_text_generation(
    last_message, messages, channel_id, root_id, sender_name, links
):
    # Send the messages to the AI API
    response = ai_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": get_system_instructions()}, *messages],
        timeout=timeout,
        temperature=temperature,
    )

    # Extract the text content
    response_text = response.choices[0].message.content

    # Failsafe: Remove all blocks containing [CONTEXT
    response_text = re.sub(r"(?s)\[CONTEXT.*?]", "", response_text).strip()

    # Failsafe: Remove all input links from the response
    for link in links:
        response_text = response_text.replace(link, "")
    response_text = response_text.strip()

    # Failsafe: Remove all empty Markdown links
    response_text = re.sub(r"\[.*?]\(\)", "", response_text).strip()

    # Split the response into multiple messages if necessary
    response_parts = split_message(response_text)

    # Send each part of the response as a separate message
    for part in response_parts:
        # Send the API response back to the Mattermost channel as a reply to the thread or as a new thread
        driver.posts.create_post(
            {"channel_id": channel_id, "message": part, "root_id": root_id}
        )


def process_message(last_message, messages, channel_id, root_id, sender_name, links):
    stop_typing_event = None
    typing_indicator_thread = None
    try:
        logger.info("Querying AI API")

        # Start the typing indicator
        stop_typing_event, typing_indicator_thread = handle_typing_indicator(
            driver.client.userid, channel_id, root_id
        )

        # Check if "draw " is present in any case
        if re.search(r"\bdraw ", last_message, re.IGNORECASE):
            handle_image_generation(
                last_message, messages, channel_id, root_id, sender_name, links
            )
        else:
            handle_text_generation(
                last_message, messages, channel_id, root_id, sender_name, links
            )

    except Exception as e:
        logging.error(f"Error: {str(e)} {traceback.format_exc()}")
        driver.posts.create_post(
            {"channel_id": channel_id, "message": str(e), "root_id": root_id}
        )

    finally:
        if stop_typing_event is not None:
            stop_typing_event.set()
        if typing_indicator_thread is not None:
            typing_indicator_thread.join()


def should_ignore_post(post):
    sender_id = post["user_id"]

    # Ignore own posts
    if sender_id == driver.client.userid or sender_id == mattermost_ignore_sender_id:
        return True

    if sender_id == mattermost_ignore_sender_id:
        logging.info("Ignoring post from an ignored sender ID")
        return True

    if post.get("props", {}).get("from_bot") == "true":
        logging.info("Ignoring post from a bot")
        return True

    return False


def extract_post_data(post, event_data):
    # Remove the "@chatbot" mention from the message
    message = post["message"].replace(chatbot_username_at, "").strip()
    channel_id = post["channel_id"]
    sender_name = sanitize_username(event_data["data"]["sender_name"])
    root_id = post["root_id"]
    post_id = post["id"]
    channel_display_name = event_data["data"]["channel_display_name"]
    return message, channel_id, sender_name, root_id, post_id, channel_display_name


async def message_handler(event):
    try:
        event_data = json.loads(event)
        logging.info(f"Received event: {event_data}")
        if event_data.get("event") == "hello":
            logging.info("Received 'hello' event. WebSocket connection established.")
        elif event_data.get("event") == "posted":
            post = json.loads(event_data["data"]["post"])
            if should_ignore_post(post):
                return

            # Check if the post is from a bot
            if post.get("props", {}).get("from_bot") == "true":
                logging.info("Ignoring post from a bot")
                return

            message, channel_id, sender_name, root_id, post_id, channel_display_name = (
                extract_post_data(post, event_data)
            )

            try:
                # Retrieve the thread context
                messages = []
                chatbot_invoked = False
                if root_id:
                    thread = driver.posts.get_thread(root_id)
                    # Sort the thread posts based on their create_at timestamp
                    sorted_posts = sorted(
                        thread["posts"].values(), key=lambda x: x["create_at"]
                    )
                    for thread_post in sorted_posts:
                        if thread_post["id"] != post_id:
                            thread_sender_name = get_username_from_user_id(
                                thread_post["user_id"]
                            )
                            thread_message = thread_post["message"]
                            role = (
                                "assistant"
                                if thread_post["user_id"] == driver.client.userid
                                else "user"
                            )
                            messages.append(
                                {
                                    "role": role,
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"[CONTEXT, from:{thread_sender_name}] {thread_message}",
                                        }
                                    ],
                                }
                            )

                            if role == "assistant":
                                chatbot_invoked = True
                else:
                    # If the message is not part of a thread, reply to it to create a new thread
                    root_id = post["id"]

                # Add the current message to the messages array if "@chatbot" is mentioned, the chatbot has already been invoked in the thread or its a DM
                if (
                    chatbot_username_at in post["message"]
                    or chatbot_invoked
                    or channel_display_name.startswith("@")
                ):
                    links = re.findall(
                        r"(https?://\S+)", message
                    )  # Allow both http and https links
                    extracted_text = ""
                    total_size = 0
                    image_messages = []

                    with httpx.Client() as client:
                        for link in links:
                            if re.search(regex_local_links, link):
                                logging.info(f"Skipping local URL: {link}")
                                continue
                            try:
                                if yt_is_valid_url(link):
                                    transcript_text = yt_get_transcript(link)
                                    extracted_text += transcript_text
                                    continue

                                with client.stream(
                                    "GET", link, timeout=4, follow_redirects=True
                                ) as response:
                                    final_url = str(response.url)

                                    if re.search(regex_local_links, final_url):
                                        logging.info(
                                            f"Skipping local URL after redirection: {final_url}"
                                        )
                                        continue

                                    content_type = response.headers.get(
                                        "content-type", ""
                                    ).lower()
                                    if "image" in content_type:
                                        # Check for compatible content types
                                        compatible_content_types = [
                                            "image/jpeg",
                                            "image/png",
                                            "image/gif",
                                            "image/webp",
                                        ]
                                        if content_type not in compatible_content_types:
                                            raise Exception(
                                                f"Unsupported image content type: {content_type}"
                                            )

                                        # Handle image content
                                        image_data = b""
                                        for chunk in response.iter_bytes():
                                            image_data += chunk
                                            total_size += len(chunk)
                                            if total_size > max_response_size:
                                                extracted_text += "*WEBSITE SIZE EXCEEDED THE MAXIMUM LIMIT FOR THE CHATBOT, WARN THE CHATBOT USER*"
                                                raise Exception(
                                                    "Response size exceeds the maximum limit at image processing"
                                                )

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
                                        closest_ratio = min(
                                            supported_ratios,
                                            key=lambda x: abs(x[0] - aspect_ratio),
                                        )
                                        target_width, target_height = (
                                            closest_ratio[2],
                                            closest_ratio[3],
                                        )

                                        # Resize the image to the target dimensions
                                        resized_image = image.resize(
                                            (target_width, target_height),
                                            Image.Resampling.LANCZOS,
                                        )

                                        # Save the resized image to a BytesIO object
                                        buffer = BytesIO()
                                        resized_image.save(
                                            buffer, format=image.format, optimize=True
                                        )
                                        resized_image_data = buffer.getvalue()

                                        # Check if the resized image size exceeds 3MB
                                        if len(resized_image_data) > 3 * 1024 * 1024:
                                            # Compress the resized image to a target size of 3MB
                                            target_size = 3 * 1024 * 1024
                                            quality = 90
                                            while len(resized_image_data) > target_size:
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

                                                if quality <= 0:
                                                    raise Exception(
                                                        "Image too large, can't compress"
                                                    )

                                        image_data_base64 = base64.b64encode(
                                            resized_image_data
                                        ).decode("utf-8")
                                        image_messages.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{content_type};base64,{image_data_base64}"
                                                },
                                            }
                                        )
                                    else:
                                        # Handle text content
                                        try:
                                            if flaresolverr_endpoint:
                                                extracted_text += extract_content_with_flaresolverr(link, flaresolverr_endpoint)
                                            else:
                                                raise Exception("FlareSolverr endpoint not available")
                                        except Exception as e:
                                            logging.info(f"Falling back to HTTPX. Reason: {str(e)}")

                                        content_chunks = []
                                        for chunk in response.iter_bytes():
                                            content_chunks.append(chunk)
                                            total_size += len(chunk)
                                            if total_size > max_response_size:
                                                extracted_text += "*WEBSITE SIZE EXCEEDED THE MAXIMUM LIMIT FOR THE CHATBOT, WARN THE CHATBOT USER*"
                                                raise Exception(
                                                    "Response size exceeds the maximum limit"
                                                )
                                        content = b"".join(content_chunks)
                                        soup = BeautifulSoup(content, "html.parser")
                                        extracted_text += soup.get_text(
                                            " | ", strip=True
                                        )
                            except Exception as e:
                                logging.error(
                                    f"Error extracting content from link {link}: {str(e)} {traceback.format_exc()}"
                                )

                    content = f"[CONTEXT, from:{sender_name}"
                    if extracted_text != "":
                        content += f", extracted_website_text:{extracted_text}"
                    content += f"] {message}"

                    if image_messages:
                        image_messages.append({"type": "text", "text": content})
                        messages.append({"role": "user", "content": image_messages})
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": content}],
                            }
                        )

                    # Ensure alternating roles in the messages array
                    messages = ensure_alternating_roles(messages)

                    # Submit the task to the thread pool. We do this because Mattermostdriver-async is outdated
                    thread_pool.submit(
                        process_message,
                        message,
                        messages,
                        channel_id,
                        root_id,
                        sender_name,
                        links,
                    )

            except Exception as e:
                logging.error(
                    f"Error inner message handler: {str(e)} {traceback.format_exc()}"
                )
        else:
            # Handle other events
            pass
    except json.JSONDecodeError:
        logging.error(
            f"Failed to parse event as JSON: {event} {traceback.format_exc()}"
        )
    except Exception as e:
        logging.error(f"Error message_handler: {str(e)} {traceback.format_exc()}")


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
    pattern = r"(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/|youtube\.com/shorts/)([^\"&?/\s]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def yt_get_transcript(url):
    try:
        video_id = yt_extract_video_id(url)
        preferred_transcript = yt_find_preferred_transcript(video_id)

        if preferred_transcript:
            transcript = preferred_transcript.fetch()
            return str(transcript)
    except Exception as e:
        logging.info(f"YouTube Transcript Exception: {str(e)}")

    return (
        "*COULD NOT FETCH THE VIDEO TRANSCRIPT FOR THE CHATBOT, WARN THE CHATBOT USER*"
    )


def yt_is_valid_url(url):
    # Pattern to match various YouTube URL formats including video IDs
    pattern = r"(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/|youtube\.com/shorts/)([^\"&?/\s]{11})"
    match = re.search(pattern, url)
    return bool(match)  # True if match found, False otherwise


def extract_content_with_flaresolverr(link, flaresolverr_endpoint):
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
        soup = BeautifulSoup(content, "html.parser")
        extracted_text = soup.get_text(" | ", strip=True)
        return extracted_text
    else:
        raise Exception(f"FlareSolverr request failed: {data}")


def main():
    try:
        # Log in to the Mattermost server
        driver.login()
        global chatbot_username, chatbot_username_at
        chatbot_username = driver.client.username
        chatbot_username_at = f"@{chatbot_username}"

        logging.info(f"SYSTEM PROMPT: {get_system_instructions()}")

        # Initialize the WebSocket connection
        while True:
            try:
                # Initialize the WebSocket connection
                driver.init_websocket(message_handler)
            except Exception as e:
                logging.error(
                    f"Error initializing WebSocket: {str(e)} {traceback.format_exc()}"
                )
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, logout and exit")
        driver.logout()
    except Exception as e:
        logging.error(f"Error: {str(e)} {traceback.format_exc()}")


if __name__ == "__main__":
    main()
