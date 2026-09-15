import os
import time
import json
import mimetypes
import gradio as gr
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# API Key setup
# ------------------------------------------------------------------
# Before running: export GEMINI_API_KEY="your_key_here"
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.6-flash"

# ------------------------------------------------------------------
# Exercise 3: system prompt - mushroom expert persona + topic steering
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are MycoBot, an expert mushroom chatbot.
Your responsibilities:
1. Answer questions about mushrooms (identification, edibility, ecology, cooking, etc.) in a way that is professional yet easy to understand.
2. When the user sends a mushroom image, carefully examine its features (color, shape, gills, growing environment, etc.) and give your assessment.
3. If the user tries to talk about topics unrelated to mushrooms, politely steer the conversation back to mushrooms,
   for example: "That's an interesting topic, but I'm much more of a mushroom expert! Did you know that... (a mushroom fun fact)...?"
4. Keep the conversation natural, friendly, and engaging.
"""

# ------------------------------------------------------------------
# Exercise 5: JSON output schema (forces Gemini to return this exact structure)
# ------------------------------------------------------------------
CLASSIFICATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "common_name": types.Schema(type=types.Type.STRING, description="Common name of the mushroom"),
        "genus": types.Schema(type=types.Type.STRING, description="Genus (scientific name) of the mushroom"),
        "confidence": types.Schema(
            type=types.Type.NUMBER, description="Confidence of the identification, between 0 and 1"
        ),
        "visible": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.STRING, enum=["cap", "hymenium", "stipe"]
            ),
            description="Which parts of the mushroom are visible in the image",
        ),
        "color": types.Schema(type=types.Type.STRING, description="Color of the mushroom in the image"),
        "edible": types.Schema(type=types.Type.BOOLEAN, description="Whether the mushroom is edible"),
    },
    required=["common_name", "genus", "confidence", "visible", "color", "edible"],
)

# ------------------------------------------------------------------
# Exercise 6: global variable that remembers previously identified mushrooms
# ------------------------------------------------------------------
# Note: this is a simplification suitable for a single-user local demo.
# A production multi-user system would need to key this by session id
# instead of a single global list.
remembered_mushroom_info = []


def extract_text(content):
    """Gradio's message content is sometimes a plain string, sometimes a
    list like [{'text': ..., 'type': 'text'}] - normalize both to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(texts)
    return str(content)


def image_to_part(file_path):
    """Convert an image file path into a Gemini Part object."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def classify_mushroom_image(image_part, file_name="image"):
    """
    Exercise 5: call Gemini and force a JSON response following the schema.
    Prints the result to the console and returns it as a dict for later use.

    Note: the instruction text below was tuned after an experiment (see
    the "Classification Accuracy" section of the report) - the original
    version made the model confidently report edible=true even when key
    parts (e.g. the stipe) were missing from the image. The instruction
    now explicitly asks the model to fold identification reliability
    into the edible judgement, instead of only answering whether the
    species is theoretically edible.
    """
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[
                    image_part,
                    types.Part(
                        text=(
                            "Analyze this mushroom image and output your identification "
                            "following the given JSON schema.\n"
                            "Important: the 'edible' field should NOT simply answer "
                            "\"is this species theoretically edible\". It should reflect "
                            "how reliable your identification is from THIS image. "
                            "If a key part (e.g. the stipe) is not visible, and this "
                            "prevents you from ruling out a similar-looking toxic species, "
                            "set edible to false and lower the confidence score accordingly."
                        )
                    ),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=CLASSIFICATION_SCHEMA,
        ),
    )
    classification = json.loads(response.text)

    # Printed to the console only - never shown in the chat interface
    print(f"\n=== [{file_name}] Mushroom classification JSON ===")
    print(json.dumps(classification, ensure_ascii=False, indent=2))
    print("=" * 40 + "\n")

    return classification


# ------------------------------------------------------------------
# Exercise 5+6: main chat logic
# ------------------------------------------------------------------
def chat_fn(message, history):
    # --- Rebuild conversation history for Gemini ---
    contents = []

    # Inject remembered mushroom classifications as hidden context, so the
    # bot can answer follow-up questions without the image being re-sent.
    if remembered_mushroom_info:
        memory_text = (
            "Here is the classification info (JSON) from mushroom images "
            "analyzed earlier in this conversation. Use it when relevant:\n"
        )
        memory_text += "\n".join(
            json.dumps(info, ensure_ascii=False) for info in remembered_mushroom_info
        )
        contents.append(types.Content(role="user", parts=[types.Part(text=memory_text)]))
        contents.append(
            types.Content(role="model", parts=[types.Part(text="Understood, I'll keep that in mind.")])
        )

    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        text = extract_text(turn["content"])
        if not text:
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

    # --- Handle the current turn's input ---
    files = message.get("files", [])
    user_text = message.get("text", "")

    current_parts = []

    if files:
        # Exercise 5: run structured JSON classification for each image
        for file_path in files:
            image_part = image_to_part(file_path)
            file_name = os.path.basename(file_path)
            classification = classify_mushroom_image(image_part, file_name)
            remembered_mushroom_info.append(classification)
            current_parts.append(image_part)

        # Exercise 6: answer the question if one was asked, otherwise summarize
        if user_text:
            current_parts.append(types.Part(text=user_text))
        else:
            current_parts.append(
                types.Part(
                    text="The user did not ask a specific question. Based on your "
                    "analysis above, give a short natural-language summary "
                    "(do not output JSON)."
                )
            )
    else:
        current_parts.append(types.Part(text=user_text))

    contents.append(types.Content(role="user", parts=current_parts))

    # --- Call Gemini (streaming) ---
    response_stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    partial_text = ""
    for chunk in response_stream:
        if chunk.text:
            for char in chunk.text:
                partial_text += char
                yield partial_text
                time.sleep(0.01)


# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    multimodal=True,
    title="🍄 MycoBot - Your Mushroom Expert Assistant",
    description="Ask me anything about mushrooms, or upload a photo and I'll take a look!",
)

if __name__ == "__main__":
    demo.launch()