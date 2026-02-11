import asyncio
from enum import Enum, auto
from io import BytesIO
from deprecated import params
from flask import Flask, request, abort, send_file

from linebot import LineBotApi
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    ImageMessage,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    MessagingApiBlob,
)
from linebot.v3.messaging.models import audio_message
from linebot.v3.webhooks import (
    FileMessageContent,
    ImageMessageContent,
    MessageEvent,
    TextMessageContent,
)
import aiohttp
import dotenv
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
import aiofiles
from linebot.v3.webhooks.models import message_content
from werkzeug.datastructures import headers


dotenv.load_dotenv()
app = Flask(__name__)


def get_required_env(key: str) -> str:
    ret = os.getenv(key)
    if ret is None:
        raise EnvironmentError(f"{key} is not set")

    return ret


@dataclass
class SelectionElement:
    id: str
    name: str

    @staticmethod
    def format_list(items: list["SelectionElement"]) -> str:
        return "\n".join([f"<{i}>-{x.name}" for i, x in enumerate(items)])


class SelectionType(Enum):
    Knowledges = auto()
    Files = auto()


@dataclass
class Cache:
    file_id: str | None = None
    collection_id: str | None = None
    selection_list: list[SelectionElement] = field(default_factory=list)
    selection_list_type: SelectionType | None = None

    def __str__(self) -> str:
        return f"""
            using knowledge id: {self.collection_id}
            using file id: {self.file_id}
            ========
            selection type: {self.selection_list_type}
            selection: {SelectionElement.format_list(self.selection_list)}
        """


class UserCache:
    user_cache: dict[str, Cache] = {}

    def __init__(self) -> None:
        self.user_cache: dict[str, Cache] = {}

    def __getitem__(self, user_id: str, auto_init: bool = True) -> Cache:
        if auto_init and user_id not in self.user_cache:
            self.user_cache[user_id] = Cache()
        return self.user_cache[user_id]

    def __contains__(self, user_id: str) -> bool:
        return user_id in self.user_cache


class ReplyException(Exception):
    pass


class ParamsNotSufficant(ReplyException):
    pass


class CacheException(Exception):
    pass


configuration = Configuration(access_token=os.getenv("CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("CHANNEL_SECRET"))
config = {}
user_cache = UserCache()
BASE_DIR = Path(__file__).resolve().parent
STORAGE_PATH = Path(get_required_env("STORAGE_PATH"))
OPEN_WEBUI_URL = get_required_env("OPEN_WEBUI_URL").rstrip("/")
OPEN_WEBUI_KNOWLEDGE_API = OPEN_WEBUI_URL + "/api/v1/knowledge"
OPEN_WEBUI_FILE_API = OPEN_WEBUI_URL + "/api/v1/files"
HTTPS_URL = get_required_env("HTTPS_URL").rstrip("/")
OPEN_WEBUI_API_KEY = get_required_env("OPEN_WEBUI_API_KEY")

if not STORAGE_PATH.is_absolute():
    STORAGE_PATH = BASE_DIR / STORAGE_PATH

with open("config.json", "r") as f:
    config = json.load(f)
    app.logger.debug(config)


@app.route("/files/<id>", methods=["GET"])
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


def handle_all_file(event):
    with ApiClient(configuration) as api_client:
        messaging_api = MessagingApiBlob(api_client)
        line_bot_api = MessagingApi(api_client)
        file_id = event.message.id
        message_content = messaging_api.get_message_content(file_id)
        file_id = asyncio.run(upload_file_data_to_open_webui(message_content))
        user_id = event.source.user_id
        messages = [
            TextMessage(
                text=f"{file_id}: is uploading.\nUse /file_status to check progress of the file.\nOnly use that file after it complete.",
                quickReply=None,
                quoteToken=None,
            )
        ]
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=messages,
                notificationDisabled=None,
            )
        )
        user_cache[user_id].file_id = file_id


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    handle_all_file(event)


@handler.add(MessageEvent, message=FileMessageContent)
def handle_file_message(event):
    handle_all_file(event)


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        user_id = event.source.user_id
        replies = asyncio.run(
            retreive_reply_from_open_webui(user_id, event.message.text)
        )
        messages = []

        for reply in replies:
            if reply.type == ReplyType.ImageUrl:
                local_url = asyncio.run(download_image_from_open_webui(reply.content))
                messages.append(
                    ImageMessage(
                        originalContentUrl=local_url,
                        previewImageUrl=local_url,
                        quickReply=None,
                    )
                )
            elif reply.type == ReplyType.Text:
                messages.append(
                    TextMessage(text=reply.content, quickReply=None, quoteToken=None)
                )

        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=messages,
                notificationDisabled=False,
            )
        )


class ExtractType(Enum):
    Features = "features"
    TaskIds = "task_ids"
    Helpers = "helpers"


def extract(text: str, type: ExtractType) -> tuple[str, dict]:
    ret = {}
    type_dict = config.get(type.value, {})
    for k, v in type_dict.get("type", {}).items():
        if v.get("default", False) and not v.get("params", []):
            ret[k] = {}

        for y in v.get("triggers", []):
            target = f"{type_dict.get('prefix', '/')}{y}{v.get('postfix', ' ')}"
            index = text.find(target)
            if index != -1:
                text = text.replace(target, "")
                maybe_params = text[index:].split()
                target_params = v.get("params", [])
                if len(target_params) > len(maybe_params):
                    raise ParamsNotSufficant

                params = {}
                for i, z in enumerate(target_params):
                    params[z] = maybe_params[i]
                    text = text[:index] + text[index:].lstrip()[len(maybe_params[i]) :]

                ret[k] = params

        if not v.get("enable", True) and k in ret:
            del ret[k]

    return (text.strip(), ret)


def extract_features(text: str) -> tuple[str, dict]:
    return extract(text, ExtractType.Features)


def extract_tool_ids(text: str) -> tuple[str, dict]:
    return extract(text, ExtractType.TaskIds)


def extract_helpers(text: str) -> tuple[str, dict]:
    return extract(text, ExtractType.Helpers)


class ReplyType(Enum):
    ImageUrl = auto()
    Text = auto()


@dataclass
class Reply:
    type: ReplyType
    content: str


async def upload_file_data_to_open_webui(data: bytearray) -> str:
    url = OPEN_WEBUI_URL + "/api/v1/files/"
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}",
            "Accept": "application/json",
        }
        form = aiohttp.FormData()
        form.add_field(
            name="file",
            value=data,
            content_type="application/octet-stream",
        )
        async with session.post(url, headers=headers, data=form) as response:
            res = await response.json()
            response.raise_for_status()
            return res["id"]


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
                response.raise_for_status()

    return https_url


async def retreive_reply_from_open_webui(user_id: str, text: str) -> list[Reply]:
    async with aiohttp.ClientSession() as session:
        (text, features) = extract_features(text)
        (text, tool_ids) = extract_tool_ids(text)
        (text, helpers) = extract_helpers(text)
        print(helpers)

        is_image_generation = features.get("image_generation") is not None

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

        # if this evnet do not need to chat, just handle reply hardcode reply from webui api
        res = await handle_non_main_event(user_id=user_id, helpers=helpers)
        if res is not None:
            return res

        # below is the process for real chat to llm
        if is_image_generation:
            url = OPEN_WEBUI_URL + "/api/v1/images/generations"
            data["prompt"] = text
        else:
            if helpers.get("chat_with_file"):
                if user_id in user_cache:
                    file_id = user_cache[user_id].file_id
                    if file_id is None:
                        return [
                            Reply(
                                type=ReplyType.Text,
                                content="there is no file set. use /list_files, /use_file or upload a file first",
                            )
                        ]
                    status = await get_file_status_in_open_webui(file_id)
                    if "complete" == status:
                        print(f"chat with file_id: {file_id}")
                        data["files"] = [{"type": "file", "id": file_id}]
                    else:
                        return [
                            Reply(
                                type=ReplyType.Text,
                                content=f"{file_id} status is {status}. please try again after it become complete. get status using /file_status",
                            ),
                        ]
                else:
                    return [
                        Reply(
                            type=ReplyType.Text,
                            content="no file_id cache, please update file or use file first",
                        )
                    ]

            if helpers.get("chat_with_collection") is not None:
                if user_id in user_cache:
                    collection_id = user_cache[user_id].collection_id
                    print(f"chat with collection_id {collection_id}")
                    data["collections"] = [{"tpye": "collection", "id": collection_id}]
                else:
                    return [
                        Reply(
                            type=ReplyType.Text,
                            content="no collection_id cache, please create knowledge or use knowledge first",
                        )
                    ]

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


async def handle_non_main_event(user_id: str, helpers: dict) -> list[Reply] | None:
    if helpers.get("info") is not None:
        tmp = [
            f"using knowledge id: {user_cache[user_id].collection_id}",
            f"using file id: {user_cache[user_id].file_id}",
        ]
        return [Reply(type=ReplyType.Text, content=str(user_cache[user_id]))]
    elif helpers.get("list_knowledges") is not None:
        knowledges = await list_knowledges()
        user_cache[user_id].selection_list_type = SelectionType.Knowledges
        user_cache[user_id].selection_list = knowledges
        return [
            Reply(type=ReplyType.Text, content=SelectionElement.format_list(knowledges))
        ]
    elif helpers.get("list_files") is not None:
        files = await list_files()
        user_cache[user_id].selection_list_type = SelectionType.Files
        user_cache[user_id].selection_list = files
        return [Reply(type=ReplyType.Text, content=SelectionElement.format_list(files))]
    elif helpers.get("list_knowledge_files") is not None:
        knowledge_id = user_cache[user_id].collection_id
        if knowledge_id is None:
            return [
                Reply(
                    type=ReplyType.Text,
                    content="knowledge not set, please /create_knowledge or /use_knowledge first",
                )
            ]
        files = await list_knowledge_files(knowledge_id=knowledge_id)
        user_cache[user_id].selection_list_type = SelectionType.Files
        user_cache[user_id].selection_list = files
        return [Reply(type=ReplyType.Text, content=SelectionElement.format_list(files))]
    elif helpers.get("use_knowledge") is not None:
        number = int(helpers.get("use_knowledge", {}).get("number"))
        elem = await use_knowledge(user_id=user_id, no=number)
        content = f"not found knowledge on: {number}"
        if elem is not None:
            content = f"use knowledge: <{number}>{elem.name}"
        return [Reply(type=ReplyType.Text, content=content)]
    elif helpers.get("use_file") is not None:
        number = int(helpers.get("use_file", {}).get("number"))
        elem = await use_file(user_id=user_id, no=number)
        content = f"not found file on: {number}"
        if elem is not None:
            content = f"use file: <{number}>{elem.name}"
        return [Reply(type=ReplyType.Text, content=content)]
    elif helpers.get("create_knowledge") is not None:
        id = await create_knowledge_in_open_webui(user_id)
        user_cache[user_id].collection_id = id
        return [Reply(type=ReplyType.Text, content=f"knowledge created: {id}")]
    elif helpers.get("add_file_to_knowledge") is not None:
        file_id = user_cache[user_id].file_id
        if not file_id:
            return [Reply(type=ReplyType.Text, content="file_id cache not set")]
        collection_id = user_cache[user_id].collection_id
        if not collection_id:
            return [Reply(type=ReplyType.Text, content="collection_id cache not set")]
        ok = await add_file_to_knowledge(file_id, collection_id)
        if ok:
            return [
                Reply(
                    type=ReplyType.Text,
                    content=f"added file {file_id} into {collection_id}",
                )
            ]
        else:
            return [Reply(type=ReplyType.Text, content="failed")]
    elif helpers.get("file_status") is not None:
        file_id = user_cache[user_id].file_id
        if file_id:
            return [
                Reply(
                    type=ReplyType.Text,
                    content=await get_file_status_in_open_webui(file_id),
                )
            ]
        else:
            return [
                Reply(
                    type=ReplyType.Text,
                    content="do not have file cache, please upload a file first",
                )
            ]


async def create_knowledge_in_open_webui(user_id):
    async with aiohttp.ClientSession() as session:
        url = OPEN_WEBUI_KNOWLEDGE_API + "/create"
        headers = {
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {"name": str(uuid4()), "description": f"created by line: {user_id}"}
        async with session.post(url=url, headers=headers, json=data) as response:
            res = await response.json()
            print(res)
            response.raise_for_status()
            return res["id"]


async def get_file_status_in_open_webui(file_id: str) -> str:
    async with aiohttp.ClientSession() as session:
        url = OPEN_WEBUI_URL + f"/api/v1/files/{file_id}/process/status"
        headers = {
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}",
            "Content-Type": "application/json",
        }
        params = {"stream": "false"}
        async with session.get(url, headers=headers, params=params) as response:
            res = await response.json()
            response.raise_for_status()
            return res["status"]


async def add_file_to_knowledge(file_id: str, knowledge_id: str) -> bool:
    async with aiohttp.ClientSession() as session:
        url = OPEN_WEBUI_KNOWLEDGE_API + f"/{knowledge_id}/file/add"
        headers = {
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {"file_id": file_id}
        async with session.post(url=url, headers=headers, json=data) as response:
            res = await response.json()
            print(res)
            response.raise_for_status()
            return response.ok


async def list_knowledges() -> list[SelectionElement]:
    async with aiohttp.ClientSession() as session:
        url = OPEN_WEBUI_KNOWLEDGE_API + "/"
        headers = {
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}",
            "Content-Type": "application/json",
        }
        async with session.get(url, headers=headers) as response:
            res = await response.json()
            response.raise_for_status()
            return [SelectionElement(id=x["id"], name=x["name"]) for x in res["items"]]


async def list_files() -> list[SelectionElement]:
    async with aiohttp.ClientSession() as session:
        url = OPEN_WEBUI_FILE_API + "/"
        headers = {
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}",
            "Content-Type": "application/json",
        }
        async with session.get(url, headers=headers) as response:
            res = await response.json()
            print(res)
            response.raise_for_status()
            return [SelectionElement(id=x["id"], name=x["name"]) for x in res]


async def list_knowledge_files(knowledge_id) -> list[SelectionElement]:
    async with aiohttp.ClientSession() as session:
        url = OPEN_WEBUI_KNOWLEDGE_API + f"/{knowledge_id}/files"
        print(url)
        headers = {
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}",
            "Content-Type": "application/json",
        }
        async with session.get(url=url, headers=headers) as response:
            res = await response.json()
            print(res)
            response.raise_for_status()
            return [
                SelectionElement(id=x["id"], name=x["filename"]) for x in res["items"]
            ]


async def if_knowledge_exist(knowledge_id) -> bool:
    async with aiohttp.ClientSession() as session:
        url = OPEN_WEBUI_KNOWLEDGE_API + f"/{knowledge_id}"
        headers = {
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}",
            "Content-Type": "application/json",
        }
        async with session.get(url, headers=headers) as response:
            await response.json()
            if response.status == 404:
                return False
            response.raise_for_status()
            return True


async def if_file_exist(file_id) -> bool:
    async with aiohttp.ClientSession() as session:
        url = OPEN_WEBUI_FILE_API + f"/{file_id}"
        headers = {
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}",
            "Content-Type": "application/json",
        }
        async with session.get(url, headers=headers) as response:
            await response.json()
            if response.status == 404:
                return False
            response.raise_for_status()
            return True


async def use_file(user_id: str, no: int) -> SelectionElement | None:
    cache = user_cache[user_id]
    if cache.selection_list_type != SelectionType.Files:
        raise CacheException(
            f"expect cache {SelectionType.Files}, find {cache.selection_list_type}, send /list_files or /list_knowledge_files first"
        )

    file_id = cache.selection_list[no].id
    if await if_file_exist(file_id=file_id):
        user_cache[user_id].file_id = file_id
        return cache.selection_list[no]
    return None


async def use_knowledge(user_id, no) -> SelectionElement | None:
    cache = user_cache[user_id]
    if cache.selection_list_type != SelectionType.Knowledges:
        raise CacheException(
            f"expect cache {SelectionType.Knowledges}, find {cache.selection_list_type}, send /list_knowledges or /list_knowledge_knowledges first"
        )

    knowledge_id = cache.selection_list[no].id
    if await if_knowledge_exist(knowledge_id=knowledge_id):
        user_cache[user_id].collection_id = knowledge_id
        return cache.selection_list[no]
    return None


if __name__ == "__main__":
    app.run(debug=True)
