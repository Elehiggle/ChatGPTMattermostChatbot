import logging
import os
import certifi

log_level_root = os.getenv("LOG_LEVEL_ROOT", "INFO").upper()

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# fix ssl certificates for compiled binaries
# https://github.com/pyinstaller/pyinstaller/issues/7229
# https://stackoverflow.com/questions/55736855/how-to-change-the-cafile-argument-in-the-ssl-module-in-python3
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# AI parameters
api_key = os.environ["AI_API_KEY"]
model = os.getenv("AI_MODEL", "gpt-4o")
ai_api_baseurl = os.getenv("AI_API_BASEURL", None)
timeout = int(os.getenv("AI_TIMEOUT", "120"))
max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
temperature = float(os.getenv("TEMPERATURE", "1"))
system_prompt_unformatted = os.getenv(
    "AI_SYSTEM_PROMPT",
    """
You are a helpful assistant used in a Mattermost chat. The current UTC time is {current_time}. 
Whenever users asks you for help you will provide them with succinct answers formatted using Markdown. Do not be apologetic. 
For tasks requiring reasoning or math, use the Chain-of-Thought methodology to explain your step-by-step calculations or logic before presenting your answer. 
Extra data is sent to you in a structured way, which might include file data, website data, and more, which is sent alongside the user message. 
If a user sends a link, use the extracted URL content provided, do not assume or make up stories based on the URL alone. 
If a user sends a YouTube link, primarily focus on the transcript and do not unnecessarily repeat the title, description or uploader of the video. 
In your answer DO NOT contain the link to the video/website the user just provided to you as the user already knows it, unless the task requires it. 
If your response contains any URLs, make sure to properly escape them using Markdown syntax for display purposes.
For creating custom emojis: Only create a custom emoji if the user explicitly requests it. 
Do not proceed with creating a custom emoji if no valid URL to an image is provided by the user. 
Do not make assumptions based on emoji usage alone; look for clear instructions from the user. 
For the raw_html_to_image function: Only use it if the user explicitly requests a screenshot of a website."""
)

image_size = os.getenv("IMAGE_SIZE", "1024x1024")
image_quality = os.getenv("IMAGE_QUALITY", "standard")
image_style = os.getenv("IMAGE_STYLE", "vivid")

# Mattermost server details
mattermost_url = os.environ["MATTERMOST_URL"]
mattermost_scheme = os.getenv("MATTERMOST_SCHEME", "https")
mattermost_port = int(os.getenv("MATTERMOST_PORT", "443"))
mattermost_basepath = os.getenv("MATTERMOST_BASEPATH", "/api/v4")

MATTERMOST_CERT_VERIFY = os.getenv("MATTERMOST_CERT_VERIFY", "TRUE")

# Handle the situation where a string path to a cert file might be handed over
if MATTERMOST_CERT_VERIFY == "TRUE":
    MATTERMOST_CERT_VERIFY = True
if MATTERMOST_CERT_VERIFY == "FALSE":
    MATTERMOST_CERT_VERIFY = False

mattermost_token = os.getenv("MATTERMOST_TOKEN", "")
mattermost_ignore_sender_id = os.getenv("MATTERMOST_IGNORE_SENDER_ID", "").split(",")
mattermost_username = os.getenv("MATTERMOST_USERNAME", "")
mattermost_password = os.getenv("MATTERMOST_PASSWORD", "")
mattermost_mfa_token = os.getenv("MATTERMOST_MFA_TOKEN", "")

typing_indicator_mode_is_full = os.getenv("TYPING_INDICATOR_MODE", "FULL") == "FULL"

flaresolverr_endpoint = os.getenv("FLARESOLVERR_ENDPOINT", "")

browser_executable_path = os.getenv("BROWSER_EXECUTABLE_PATH", "/usr/bin/chromium")

# Maximum website/file size
max_response_size = 1024 * 1024 * int(os.getenv("MAX_RESPONSE_SIZE_MB", "100"))

keep_all_url_content = os.getenv("KEEP_ALL_URL_CONTENT", "TRUE").upper() == "TRUE"

tool_use_enabled = os.getenv("TOOL_USE_ENABLED", "TRUE").upper() == "TRUE"

disable_specific_tool_calls = os.getenv("DISABLE_SPECIFIC_TOOL_CALLS", "").lower().split(",")


compatible_emoji_image_content_types = [
    "image/jpeg",
    "image/png",
    "image/gif",
]

compatible_image_content_types = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
]

mattermost_max_image_dimensions = (
    7680,
    4320,
)  # https://docs.mattermost.com/collaborate/share-files-in-messages.html#attachment-limits-and-sizes

mattermost_max_emoji_image_dimensions = (128, 128)
MATTERMOST_MAX_EMOJI_IMAGE_FILE_SIZE = 0.524287  # technically 0.524288 / 512 KiB

ai_model_max_vision_image_dimensions = (2000, 768)  # https://platform.openai.com/docs/guides/vision/managing-images

AI_MODEL_IMAGE_GENERATION_MIME_TYPE = "image/png"  # in some cases WEBP, maybe only web based ChatGPT DALL-E-3?
