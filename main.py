import asyncio
from enum import Enum, auto
from flask import Flask, request, abort, send_file

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    ImageMessage,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import aiohttp
import dotenv
import os
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import aiofiles


dotenv.load_dotenv()
app = Flask(__name__)


def get_required_env(key: str) -> str:
    ret = os.getenv(key)
    if ret is None:
        raise EnvironmentError(f"{key} is not set")

    return ret


configuration = Configuration(access_token=os.getenv("CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("CHANNEL_SECRET"))
config = {}
BASE_DIR = Path(__file__).resolve().parent
STORAGE_PATH = Path(get_required_env("STORAGE_PATH"))
OPEN_WEBUI_URL = get_required_env("OPEN_WEBUI_URL").rstrip("/")
HTTPS_URL = get_required_env("HTTPS_URL").rstrip("/")

if not STORAGE_PATH.is_absolute():
    STORAGE_PATH = BASE_DIR / STORAGE_PATH

with open("config.json", "r") as f:
    config = json.load(f)
    app.logger.debug(config)


@app.route("/files/<id>")
def files(id: str):
    path = Path(STORAGE_PATH) / f"{id}.png"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/callback", methods=["POST"])
def callback():
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info(
            "Invalid signature. Please check your channel access token/channel secret."
        )
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        replies = asyncio.run(retreive_reply_from_open_webui(event.message.text))
        messages = []

        for reply in replies:
            if reply.type == ReplyType.ImageUrl:
                local_url = asyncio.run(download_image_from_open_webui(reply.content))
                messages.append(
                    ImageMessage(
                        originalContentUrl=local_url, previewImageUrl=local_url
                    )
                )
            elif reply.type == ReplyType.Text:
                messages.append(TextMessage(text=reply.content))

        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=messages,
            )
        )


class ExtractType(Enum):
    Features = "features"
    TaskIds = "task_ids"


def extract(text: str, type: ExtractType):
    ret = {}
    type_dict = config.get(type.value, {})
    for k, v in type_dict.get("type", {}).items():
        if v.get("default", False):
            ret[k] = True

        for y in v.get("triggers", []):
            target = f"{v.get('prefix', '/')}{y} "
            if text.find(target) != -1:
                text = text.replace(target, "")
                ret[k] = True

        if not v.get("enable", True) and k in ret:
            del ret[k]

    return (text.strip(), ret)


def extract_features(text: str) -> tuple[str, dict]:
    return extract(text, ExtractType.Features)


def extract_tool_ids(text: str) -> tuple[str, dict]:
    return extract(text, ExtractType.TaskIds)


class ReplyType(Enum):
    ImageUrl = auto()
    Text = auto()


@dataclass
class Reply:
    type: ReplyType
    content: str


async def download_image_from_open_webui(url: str) -> str:
    id = uuid4()
    file_path = Path(STORAGE_PATH) / f"{id}.png"
    https_url = HTTPS_URL + "/files/" + f"{id}"
    url = OPEN_WEBUI_URL + "/" + url.lstrip("/")

    os.makedirs(STORAGE_PATH, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        api_key = get_required_env("OPEN_WEBUI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        async with session.get(url, headers=headers) as response:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await response.read())

    return https_url


async def retreive_reply_from_open_webui(text: str) -> list[Reply]:
    async with aiohttp.ClientSession() as session:
        (text, features) = extract_features(text)
        (text, tool_ids) = extract_tool_ids(text)

        is_image_generation = features.get("image_generation", False)

        api_key = get_required_env("OPEN_WEBUI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": get_required_env("OPEN_WEBUI_MODEL"),
            "messages": [
                {"role": "system", "content": get_required_env("OPEN_WEBUI_PROMPT")},
                {"role": "user", "content": text},
            ],
            "tool_ids": tool_ids,
            "features": features,
            "stream": False,
        }

        url = OPEN_WEBUI_URL + "/api/chat/completions"

        if is_image_generation:
            url = OPEN_WEBUI_URL + "/api/v1/images/generations"
            data["prompt"] = text

        async with session.post(url, headers=headers, json=data) as response:
            data = await response.json()
            if is_image_generation:
                return [Reply(type=ReplyType.ImageUrl, content=data[0]["url"])]
            else:
                return [
                    Reply(
                        type=ReplyType.Text,
                        content=data["choices"][0]["message"]["content"],
                    )
                ]


if __name__ == "__main__":
    app.run(debug=True)
