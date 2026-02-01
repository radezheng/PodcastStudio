from __future__ import annotations

import os
import json
from dataclasses import dataclass
from urllib.parse import urlsplit

from openai import BadRequestError, OpenAI

try:
    from .script_logs import ScriptLogWriter
except Exception:  # pragma: no cover
    ScriptLogWriter = object  # type: ignore


@dataclass(frozen=True)
class AzureOpenAIConfig:
    api_key: str
    base_url: str
    deployment: str


def _normalize_azure_base_url(value: str) -> str:
    """Normalize Azure OpenAI endpoint/base URL to the OpenAI v1 compatibility path.

    This backend uses the OpenAI Responses API against Azure's `/openai/v1/` endpoint.
    Users sometimes set `AZURE_OPENAI_BASE_URL` to a legacy deployments URL that includes
    an `api-version=...` query param; that breaks with 400 "API version not supported".
    """

    raw = (value or "").strip().strip('"')
    if not raw:
        return ""

    parts = urlsplit(raw)

    scheme = parts.scheme or "https"
    netloc = parts.netloc or parts.path  # handles inputs like "my-resource.openai.azure.com"

    # If the user passed a full URL without netloc (rare), try to keep it as-is.
    if not netloc:
        raise RuntimeError("Invalid AZURE_OPENAI_BASE_URL/AZURE_OPENAI_ENDPOINT")

    # Always target the OpenAI v1 compatibility root (no api-version query param).
    return f"{scheme}://{netloc.rstrip('/')}/openai/v1/"


def load_azure_openai_config() -> AzureOpenAIConfig:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip().strip('"')
    base_url = _normalize_azure_base_url(os.environ.get("AZURE_OPENAI_BASE_URL", ""))
    endpoint = _normalize_azure_base_url(os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip().strip('"')

    if not base_url:
        if not endpoint:
            raise RuntimeError("Missing AZURE_OPENAI_BASE_URL or AZURE_OPENAI_ENDPOINT")
        base_url = endpoint

    if not api_key:
        raise RuntimeError("Missing AZURE_OPENAI_API_KEY")
    if not deployment:
        raise RuntimeError("Missing AZURE_OPENAI_DEPLOYMENT_NAME")

    return AzureOpenAIConfig(
        api_key=api_key,
        base_url=base_url,
        deployment=deployment,
    )


def create_client(cfg: AzureOpenAIConfig) -> OpenAI:
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)


def _extract_response_text(resp: object) -> str:
    # Newer SDKs expose a convenient `output_text` property.
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    # Fallback: attempt to flatten structured outputs.
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    return ""


DEFAULT_PODCAST_SYSTEM_PROMPT = (
    "You are a professional podcast scriptwriter and producer.\n\n"
    "Your task is to generate a high-quality podcast script based on the input provided by the user. "
    "The input may include a topic, a link, or a file (such as an article or document). "
    "You must extract and synthesize the key ideas from the provided material.\n\n"
    "Requirements:\n\n"
    "The podcast should be interview-style, with clear dialogue between a Host and a Guest.\n"
    "The Guest is an expert on the topic and should speak with authority, insight, and confidence.\n"
    "The script should match the specified duration (estimate time based on natural speaking pace).\n"
    "The tone should be engaging, witty, and humorous, while still being informative and professional.\n"
    "Use clear structure: introduction, main discussion segments, and a conclusion.\n"
    "Include natural conversational elements such as follow-up questions, light jokes, analogies, and smooth transitions.\n"
    "Avoid reading directly from the source; instead, reinterpret and explain concepts in an accessible way.\n"
    "If the source material is complex, simplify it for a general audience without losing accuracy.\n\n"
    "Return only the script content. No analysis, no markdown."
)


def _truncate(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "\n...\n(截断)"


def _plan_search_queries(
    *,
    client: OpenAI,
    model: str,
    theme: str,
    source_text: str | None,
    log: "ScriptLogWriter | None" = None,
) -> list[str]:
    material = _truncate(source_text or "", 4000)
    prompt = (
        "你是研究助理。请为播客选题生成用于网络检索的关键词/搜索语句。\n"
        "要求：\n"
        "- 输出必须是严格的 JSON 数组（不要 markdown，不要解释）\n"
        "- 数组里每个元素是字符串\n"
        "- 1 到 4 条查询语句\n"
        "- 查询语句要具体，包含关键实体/时间范围/地点(如适用)\n\n"
        f"主题：{theme}\n"
    )
    if material:
        prompt += f"\n补充材料（可能是 txt/markdown，供你参考）：\n{material}\n"

    if log is not None:
        log.stage_start("web_plan", "Plan search queries")
        log.step("web_plan", "Planning queries")

    resp = client.responses.create(model=model, input=prompt)
    raw = _extract_response_text(resp).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            queries = [str(x).strip() for x in data if isinstance(x, str) and str(x).strip()]
            queries = queries[:4]
            if log is not None:
                log.step("web_plan", "Planned queries", data={"queries": queries})
                log.stage_end("web_plan", "ok")
            return queries
    except Exception:
        pass

    # Fallback: treat as newline-separated.
    lines = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()]
    queries = lines[:4]
    if log is not None:
        log.step("web_plan", "Planned queries (fallback)", data={"queries": queries})
        log.stage_end("web_plan", "ok")
    return queries


def _web_search_notes(
    *,
    client: OpenAI,
    model: str,
    query: str,
    log: "ScriptLogWriter | None" = None,
) -> str:
    prompt = (
        "请使用网络搜索获取最新/可靠信息，并输出可用于写播客脚本的要点。\n"
        "输出要求：\n"
        "- 8 条以内要点\n"
        "- 每条要点尽量包含关键数据/事实（如有）\n"
        "- 只输出要点，不要标题，不要解释\n\n"
        f"搜索查询：{query}"
    )
    resp = client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=prompt,
    )
    text = _extract_response_text(resp).strip()
    if log is not None:
        log.detail("web_search", "Search result", data={"query": query, "chars": len(text)})
    return text


def _synthesize_notes(
    *,
    client: OpenAI,
    model: str,
    theme: str,
    notes: list[str],
    log: "ScriptLogWriter | None" = None,
) -> str:
    joined = "\n\n".join([f"[Search #{i+1}]\n{n}" for i, n in enumerate(notes) if n.strip()])
    prompt = (
        "你是编辑。请把多次网络搜索的要点综合成一份写作素材备忘录。\n"
        "要求：\n"
        "- 10 条以内\n"
        "- 去重、合并同义信息\n"
        "- 保留关键事实/数据/时间线\n"
        "- 只输出要点，不要标题，不要解释\n\n"
        f"主题：{theme}\n\n"
        f"搜索要点：\n{joined}"
    )
    if log is not None:
        log.stage_start("web_summarize", "Synthesize web notes")
        log.step("web_summarize", "Synthesizing")

    resp = client.responses.create(model=model, input=prompt)
    out = _extract_response_text(resp).strip()
    if log is not None:
        log.step("web_summarize", "Synthesis done", data={"chars": len(out)})
        log.stage_end("web_summarize", "ok")
    return out


def generate_podcast_script(
    *,
    theme: str,
    minutes: int = 4,
    speaker_count: int = 2,
    system_prompt: str | None = None,
    source_text: str | None = None,
    use_web_search: bool = False,
    log: "ScriptLogWriter | None" = None,
) -> str:
    cfg = load_azure_openai_config()
    client = create_client(cfg)

    if not (1 <= speaker_count <= 4):
        raise ValueError("speaker_count must be between 1 and 4")

    system = (system_prompt or DEFAULT_PODCAST_SYSTEM_PROMPT).strip()
    if not system:
        system = DEFAULT_PODCAST_SYSTEM_PROMPT

    speaker_format = "\n".join([f"Speaker {i}: ..." for i in range(1, speaker_count + 1)])

    materials_block = ""
    if source_text:
        if log is not None:
            log.step("generation", "Using uploaded/source text", data={"chars": len(source_text)})
        materials_block = (
            "\n补充材料（来自用户上传的 txt/markdown，请提取关键信息用于写作）：\n"
            f"{_truncate(source_text, 12000)}\n"
        )

    web_notes_block = ""
    if use_web_search:
        if log is not None:
            log.stage_start("web_search", "Web search")
            log.step("web_search", "Web search enabled")

        queries = _plan_search_queries(
            client=client,
            model=cfg.deployment,
            theme=theme,
            source_text=source_text,
            log=log,
        )
        queries = [q for q in queries if q][:4]
        notes: list[str] = []
        for q in queries:
            if log is not None:
                log.step("web_search", "Searching", data={"query": q})
            n = _web_search_notes(client=client, model=cfg.deployment, query=q, log=log)
            if n:
                notes.append(n)
        if notes:
            synthesized = _synthesize_notes(
                client=client,
                model=cfg.deployment,
                theme=theme,
                notes=notes,
                log=log,
            )
            if synthesized:
                web_notes_block = "\n网络搜索素材（已综合）：\n" + synthesized + "\n"

        if log is not None:
            log.stage_end("web_search", "ok")
    else:
        if log is not None:
            log.step("generation", "Web search disabled")
    user = (
        f"主题：{theme}\n\n"
        f"请生成一段约 {minutes} 分钟的 {speaker_count} 人中文播客对话脚本。\n"
        "严格使用以下格式逐行输出（只能使用 Speaker + 数字，不要写人名）：\n"
        f"{speaker_format}\n"
        "要求：\n"
        "- 主持人轮流对话，语气自然，有少量幽默\n"
        f"- 只允许出现 Speaker 1 到 Speaker {speaker_count}，不要输出其他名字或编号\n"
        "- 每行一句或两句，避免超长段落\n"
        "- 不要输出标题、要点列表或任何解释\n"
        f"{materials_block}"
        f"{web_notes_block}"
    )

    prompt = (
        f"{system}\n\n"
        "You may use web search if it improves accuracy or timeliness. "
        "If you use search, incorporate only the relevant facts and still output only the script.\n\n"
        f"{user}"
    )

    try:
        resp = client.responses.create(
            model=cfg.deployment,
            tools=[{"type": "web_search_preview"}],
            input=prompt,
        )
    except BadRequestError as e:
        # Some Azure deployments reject `temperature` (or other sampling params) for
        # certain models. If that ever gets added back, retry cleanly.
        message = ""
        try:
            message = (e.body or {}).get("error", {}).get("message", "")  # type: ignore[union-attr]
        except Exception:
            message = ""
        if "Unsupported parameter" in message and "temperature" in message:
            resp = client.responses.create(
                model=cfg.deployment,
                tools=[{"type": "web_search_preview"}],
                input=prompt,
            )
        else:
            raise

    text = _extract_response_text(resp).strip()
    if not text:
        raise RuntimeError("Azure OpenAI returned empty script")
    return text
