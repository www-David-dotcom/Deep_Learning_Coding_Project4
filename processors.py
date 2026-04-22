import re
from typing import Any, TypedDict

from PIL.Image import Image

type Conversation = list[dict[str, Any]]


class ConversationalLanguageModeling(TypedDict):
    messages: Conversation


class ConversationalPromptCompletion(TypedDict):
    prompt: Conversation
    completion: Conversation


class IconQASample(TypedDict):
    question: str
    choices: str
    answer: str | None
    query_image: Image
    choice_image_0: Image
    choice_image_1: Image

ANSWER_TO_OPTION = {"choice_0.png":"A", "choice_1.png": "B"}
OPTION_TO_ANSWER = {v: k for k, v in ANSWER_TO_OPTION.items()} 
SYSTEM_PROMPT = (
    "You are solving IconQA-style visual multiple-choise questions. "
    "Ths user will provide 1 main diagram and 2 candidate answer images labeled A and B. "
    "Use the diagram and question to choose the better candidate. "
    "Please think silently and respond with only <answer>A</answer> or <answer>B</answer>"
)

def convert_custom_train_to_conversation(
    sample: dict[str, Any],
) -> ConversationalPromptCompletion:
    """Builds one SFT conversation from a custom training sample.

    Args:
        sample: A sample in the custom training dataset. The schema of this
            dataset is student-defined.

    Returns:
        A conversation for training. You are responsible for converting your
        custom sample format into this prompt-completion structure.
    """

    # YOUR CODE BEGIN.
    formatted_sample = IconQASample(
        question=sample["question"].strip(),
        choices=sample.get("choices", "choice_0.png,choice_1.png"),
        answer=sample["answer"],
        query_image=sample["query_image"],
        choice_image_0=sample["choice_image_0"],
        choice_image_1=sample["choice_image_1"],
    )
    return convert_icon_qa_train_to_conversation(formatted_sample)

    # YOUR CODE END.


def convert_icon_qa_test_to_conversation(
    sample: IconQASample,
) -> ConversationalLanguageModeling:
    """Builds one eval conversation from an IconQA sample.

    Args:
        sample: A IconQA sample, whose ``answer`` field is always ``None``.

    Returns:
        A conversation for testing.
    """

    # YOUR CODE BEGIN.
    question = sample["question"].strip()

    return ConversationalLanguageModeling(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Main diagram:"},
                    {
                        "type": "image",
                        "image": sample["query_image"],
                    },
                    {
                        "type": "text",
                        "text": f"Question: {question}",
                    },
                    {"type": "text", "text": "Option A:"},
                    {
                        "type": "image",
                        "image": sample["choice_image_0"],
                    },
                    {"type": "text", "text": "Option B:"},
                    {
                        "type": "image",
                        "image": sample["choice_image_1"],
                    },
                    {
                        "type": "text",
                        "text": (
                            "Choose the better answer image. "
                            "Return only <answer>A</answer> or <answer>B</answer>."
                        ),
                    },
                ],
            }
        ]
    )

    # YOUR CODE END.


def convert_icon_qa_train_to_conversation(
    sample: IconQASample,
) -> ConversationalPromptCompletion:
    """Builds one SFT conversation from an IconQA training sample.

    Args:
        sample: A IconQA sample.

    Returns:
        A conversation for training, where the prompt is the same as the test conversation
    """

    # YOUR CODE BEGIN.
    option = ANSWER_TO_OPTION[sample["answer"]]

    return ConversationalPromptCompletion(
        prompt=convert_icon_qa_test_to_conversation(sample)["messages"],
        completion=[
            {
                "role": "assistant",
                "content": f"<answer>{option}</answer>",
            }
        ],
    )

    # YOUR CODE END.


def extract_answer(generated_text: str) -> str:
    """Extracts the final answer token from model output.

    Args:
        generated_text: Raw generated text.

    Returns:
        The parsed answer.
    """

    # YOUR CODE BEGIN.
    text = generated_text.strip()
    match = re.search(r"<answer>\s*([AB])</answer", text, flags=re.IGNORECASE)
    if not match: match = re.search(r"\b([AB])\b", text, flags=re.IGNORECASE) # if no strict matches, search for A or B in text

    if match: return OPTION_TO_ANSWER[match.group(1).upper()]

    # fallback to "choice_0.png" or "choice_1.png" in answer text if there indeed is not any choices found
    if "choice_0.png" in text: return "choice_0.png"
    if "choice_1.png" in text: return "choice_1.png"
    return ""
    # YOUR CODE END.
