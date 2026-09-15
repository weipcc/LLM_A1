import os
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
2. 如果使用者想聊蘑菇以外的話題，禮貌地把話題導回蘑菇相關內容，
   例如可以說「這是個有趣的話題，不過我更擅長聊蘑菇！你知道...(蘑菇冷知識)...嗎？」
3. 保持對話自然、友善、有趣。
"""

# ------------------------------------------------------------------
# Exercise 1: 最簡單版本的聊天邏輯（純文字，先不處理圖片、不串流）
# ------------------------------------------------------------------
def chat_fn(message, history):
    """
    message: 使用者這次輸入的文字（字串）
    history: list of {"role": ..., "content": ...}，Gradio 會自動維護
    """
    # 把歷史紀錄轉成 Gemini 要的格式
    def extract_text(content):
        """gradio 的 content 有時是字串，有時是 [{'text': ..., 'type': 'text'}] 這種結構"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "".join(texts)
        return str(content)

    contents = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        text = extract_text(turn["content"])
        if not text:
            continue
        contents.append(
            types.Content(role=role, parts=[types.Part(text=text)])
        )
    # 加入這次的使用者輸入
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    # 呼叫 Gemini（先用一次性回覆，不串流）
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return response.text


# ------------------------------------------------------------------
# 介面
# ------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="🍄 MycoBot - 你的蘑菇專家助手",
    description="先問我任何蘑菇相關的問題吧！（這版還不能傳圖片）",
)

if __name__ == "__main__":
    demo.launch()