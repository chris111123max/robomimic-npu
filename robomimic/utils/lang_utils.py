import os
from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection


os.environ["TOKENIZERS_PARALLELISM"] = "true"


# CLIP model name
TOKENIZER_NAME = "openai/clip-vit-large-patch14"
# TOKENIZER_NAME = "openai/clip-vit-base-patch32"


# Lazy loading
lang_emb_model = None
tz = None


LANG_EMB_OBS_KEY = "lang_emb"


def _load_clip_model():
    """
    Lazy load CLIP model.
    The model is only loaded when language observations are required.
    """

    global lang_emb_model
    global tz

    if lang_emb_model is None:

        print("[robomimic] Loading CLIP language encoder...")

        cache_dir = os.path.expanduser(
            os.path.join(
                os.environ.get("HF_HOME", "~/tmp"),
                "clip"
            )
        )

        lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
            TOKENIZER_NAME,
            cache_dir=cache_dir
        ).eval()


        tz = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            cache_dir=cache_dir
        )


        print("[robomimic] CLIP language encoder loaded.")


def get_lang_emb(lang):
    """
    Convert language instruction into CLIP embedding.

    Only used for language-conditioned policies.
    """

    if lang is None:
        return None


    _load_clip_model()


    tokens = tz(
        text=lang,
        add_special_tokens=True,
        max_length=25,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )


    lang_emb = lang_emb_model(**tokens)[
        "text_embeds"
    ].detach()[0]


    return lang_emb



def get_lang_emb_shape():
    """
    Return language embedding dimension.
    """

    return list(get_lang_emb("dummy").shape)