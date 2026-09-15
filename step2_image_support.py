import os
import mimetypes
import gradio as gr
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# 設定 API Key
# ------------------------------------------------------------------
# 執行前先在終端機下： export GEMINI_API_KEY="你的key"
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.6-flash"

# ------------------------------------------------------------------
# Exercise 3: system prompt，讓機器人像蘑菇專家，並把話題導回蘑菇
# ------------------------------------------------------------------
SYSTEM_PROMPT = """你是一個蘑菇專家聊天機器人，名字叫 MycoBot。
你的任務：
1. 用專業但易懂的方式回答關於蘑菇的問題（辨識、食用性、生態、料理等）。
2. 如果使用者傳送蘑菇圖片，仔細觀察圖片中的特徵（顏色、形狀、菌褶、生長環境等）並給出你的判斷。
3. 如果使用者想聊蘑菇以外的話題，禮貌地把話題導回蘑菇相關內容，
   例如可以說「這是個有趣的話題，不過我更擅長聊蘑菇！你知道...(蘑菇冷知識)...嗎？」
4. 保持對話自然、友善、有趣。
"""


def extract_text(content):
    """gradio 的 content 有時是字串，有時是 [{'text': ..., 'type': 'text'}] 這種結構"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(texts)
    return str(content)


def image_to_part(file_path):
    """把圖片檔案路徑轉成 Gemini 要的 Part 物件"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "image/jpeg"  # 預設值，以防猜不出來
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


# ------------------------------------------------------------------
# Exercise 2: 多模態聊天邏輯（可以讀取圖片）
# ------------------------------------------------------------------
def chat_fn(message, history):
    """
    message: dict，格式為 {"text": "...", "files": ["/path/to/img.jpg", ...]}
             （因為 ChatInterface 開了 multimodal=True）
    history: list of {"role": ..., "content": ...}
    """
    # --- 組合歷史紀錄（先簡化：只帶入過去的文字，不重複帶入過去的圖片） ---
    contents = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        text = extract_text(turn["content"])
        if not text:
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

    # --- 組合這次使用者輸入的內容（文字 + 圖片） ---
    current_parts = []

    # 先放圖片（Exercise 2 的重點）
    for file_path in message.get("files", []):
        current_parts.append(image_to_part(file_path))

    # 再放文字
    user_text = message.get("text", "")
    if user_text:
        current_parts.append(types.Part(text=user_text))

    # 如果使用者只傳圖片、沒打字，補一句預設提示，避免 parts 只有圖片沒有指令
    if not current_parts:
        current_parts.append(types.Part(text="請看看這個。"))

    contents.append(types.Content(role="user", parts=current_parts))

    # --- 呼叫 Gemini（先用一次性回覆，下一步才加串流） ---
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return response.text


# ------------------------------------------------------------------
# 介面（multimodal=True 才能上傳圖片）
# ------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    multimodal=True,
    title="🍄 MycoBot - 你的蘑菇專家助手",
    description="問我蘑菇相關的問題，或直接上傳蘑菇照片讓我幫你看看！",
)

if __name__ == "__main__":
    demo.launch()