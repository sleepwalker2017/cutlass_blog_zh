#!/usr/bin/env python3
import hashlib
import html
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
import torch
from lxml import etree, html as lhtml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
ARTICLES_DIR = ROOT / "articles"
ARTICLES_EN_DIR = ROOT / "articles-en"
IMAGES_DIR = ROOT / "images"
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/nllb-200-distilled-600M"
GOOGLE_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
GLOSSARY_TERMS = [
    "PyTorch",
    "CUTLASS",
    "CUDA",
    "NVIDIA",
    "torch.mm",
    "ATen API",
    "ATen",
    "Tensor Memory Accelerator",
    "FlashAttention",
    "Stream-K",
    "Blackwell",
    "Hopper",
    "CuTe",
    "WGMMA",
    "TMA",
    "GEMM",
    "Tensor Core",
    "Tensor Cores",
    "SM",
    "FP8",
    "FP16",
    "BF16",
    "INT8",
    "FP32",
    "C++",
    "Python",
]


@dataclass(frozen=True)
class ArticleSpec:
    slug: str
    url: str
    external_url: str | None = None


ARTICLE_SPECS = [
    ArticleSpec(
        slug="cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus",
        url="https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/",
    ),
    ArticleSpec(
        slug="cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external",
        url="https://research.colfax-intl.com/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external/",
        external_url="https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/",
    ),
    ArticleSpec(
        slug="cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus",
        url="https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/",
    ),
    ArticleSpec(
        slug="cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus",
        url="https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/",
    ),
    ArticleSpec(
        slug="cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus",
        url="https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/",
    ),
    ArticleSpec(
        slug="cutlass-tutorial-persistent-kernels-and-stream-k",
        url="https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/",
    ),
    ArticleSpec(
        slug="epilogue_visitor_tree",
        url="https://research.colfax-intl.com/epilogue_visitor_tree/",
    ),
    ArticleSpec(
        slug="cutlass-tutorial-design-of-a-gemm-kernel",
        url="https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/",
    ),
    ArticleSpec(
        slug="cutlass-tutorial-wgmma-hopper",
        url="https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/",
    ),
    ArticleSpec(
        slug="tutorial-hopper-tma",
        url="https://research.colfax-intl.com/tutorial-hopper-tma/",
    ),
    ArticleSpec(
        slug="tutorial-matrix-transpose-in-cutlass",
        url="https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/",
    ),
    ArticleSpec(
        slug="tutorial-python-binding-for-cuda-libraries-in-pytorch",
        url="https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/",
    ),
]


TRANSLATION_CACHE_PATH = CACHE_DIR / "translation_cache.json"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class Translator:
    def __init__(self) -> None:
        if TRANSLATION_CACHE_PATH.exists():
            self.cache = json.loads(TRANSLATION_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            self.cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.tokenizer = None
        self.model = None
        self.target_lang_id = None

    def save(self) -> None:
        TRANSLATION_CACHE_PATH.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def translate_text(self, text: str) -> str:
        if not text:
            return text
        if not re.search(r"[A-Za-z]", text):
            return text

        key = text
        if key in self.cache:
            return self.cache[key]

        protected_text, placeholders = self.protect_terms(text)
        chunks = self._chunk_text(protected_text)
        translated = [self._translate_chunk(chunk) for chunk in chunks]
        result = self._restore_spacing(text, "".join(translated))
        for placeholder, original in placeholders.items():
            result = result.replace(placeholder, original)
        self.cache[key] = result
        return result

    def _translate_chunk(self, text: str) -> str:
        if not re.search(r"[A-Za-z]", text):
            return text
        try:
            return self._translate_google(text)
        except Exception:
            pass
        self.ensure_local_model()
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=self.target_lang_id,
                max_new_tokens=512,
                num_beams=4,
            )
        out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return out

    def _translate_google(self, text: str) -> str:
        response = self.session.get(
            GOOGLE_TRANSLATE_URL,
            params={
                "client": "gtx",
                "sl": "en",
                "tl": "zh-CN",
                "dt": "t",
                "q": text,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        translated = "".join(part[0] for part in payload[0] if part and part[0])
        return translated

    def ensure_local_model(self) -> None:
        if self.model is not None and self.tokenizer is not None and self.target_lang_id is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self.target_lang_id = self.tokenizer.convert_tokens_to_ids("zho_Hans")

    def _chunk_text(self, text: str) -> list[str]:
        text = text.strip()
        if len(text) <= 420:
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
        if len(sentences) == 1:
            sentences = textwrap.wrap(text, width=380, break_long_words=False)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not current:
                current = sentence
                continue
            if len(current) + 1 + len(sentence) <= 420:
                current = f"{current} {sentence}"
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
        return chunks

    def _restore_spacing(self, original: str, translated: str) -> str:
        translated = translated.strip()
        if original.endswith(":") and not translated.endswith("："):
            translated += "："
        if original.endswith("?") and not translated.endswith("？"):
            translated += "？"
        if original.endswith(".") and not translated.endswith(("。", ".", "：", "？", "！")):
            translated += "。"
        return translated

    def protect_terms(self, text: str) -> tuple[str, dict[str, str]]:
        placeholders: dict[str, str] = {}

        def replace_exact(value: str, original: str) -> str:
            placeholder = f"ZXQPH{len(placeholders)}ZXQ"
            placeholders[placeholder] = original
            return value.replace(original, placeholder)

        protected = text
        for term in sorted(GLOSSARY_TERMS, key=len, reverse=True):
            if term in protected:
                protected = replace_exact(protected, term)

        code_like_pattern = re.compile(
            r"\b[A-Za-z_][A-Za-z0-9_]*(?:[./:][A-Za-z0-9_:+-]+)+\b|\b[A-Z]{2,}\d*\b"
        )

        def replace_match(match: re.Match[str]) -> str:
            original = match.group(0)
            if original in placeholders.values():
                return original
            placeholder = f"ZXQPH{len(placeholders)}ZXQ"
            placeholders[placeholder] = original
            return placeholder

        protected = code_like_pattern.sub(replace_match, protected)
        return protected, placeholders


class SiteConverter:
    def __init__(self, translator: Translator) -> None:
        self.translator = translator
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.image_cache: dict[str, str] = {}

    def build(self, specs: Iterable[ArticleSpec]) -> None:
        outputs = []
        for spec in specs:
            print(f"Building {spec.slug}...", flush=True)
            record = self.build_article(spec)
            outputs.append(record)
            self.translator.save()
        outputs.sort(key=lambda item: item["published_at"], reverse=True)
        self.write_index(outputs)

    def build_article(self, spec: ArticleSpec) -> dict[str, str]:
        source_url = spec.external_url or spec.url
        doc = self.fetch_html(source_url)
        title_en = self.extract_title(doc)
        title_zh = self.translator.translate_text(title_en)
        published_at = self.extract_published_at(doc)
        source_site = self.extract_source_site(source_url)
        content = self.extract_content_root(doc, source_url)
        blocks_zh = self.render_blocks(content, spec, source_url, translate=True)
        body_zh = "\n\n".join(block for block in blocks_zh if block.strip())
        blocks_en = self.render_blocks(content, spec, source_url, translate=False)
        body_en = "\n\n".join(block for block in blocks_en if block.strip())
        front_matter = "\n".join(
            [
                "---",
                f'title_zh: "{self.escape_yaml(title_zh)}"',
                f'title_en: "{self.escape_yaml(title_en)}"',
                f'source_url: "{self.escape_yaml(source_url)}"',
                f'published_at: "{published_at}"',
                f'source_site: "{self.escape_yaml(source_site)}"',
                f"external: {'true' if spec.external_url else 'false'}",
                f'english_markdown: "articles-en/{spec.slug}.en.md"',
                "---",
                "",
            ]
        )
        markdown = f"{front_matter}# {title_zh}\n\n原文标题：{title_en}\n\n英文对照：[articles-en/{spec.slug}.en.md](../articles-en/{spec.slug}.en.md)\n\n{body_zh}\n"
        output_path = ARTICLES_DIR / f"{spec.slug}.md"
        output_path.write_text(markdown, encoding="utf-8")
        front_matter_en = "\n".join(
            [
                "---",
                f'title_en: "{self.escape_yaml(title_en)}"',
                f'title_zh: "{self.escape_yaml(title_zh)}"',
                f'source_url: "{self.escape_yaml(source_url)}"',
                f'published_at: "{published_at}"',
                f'source_site: "{self.escape_yaml(source_site)}"',
                f"external: {'true' if spec.external_url else 'false'}",
                f'chinese_markdown: "articles/{spec.slug}.md"',
                "---",
                "",
            ]
        )
        markdown_en = (
            f"{front_matter_en}# {title_en}\n\n中文翻译：[articles/{spec.slug}.md](../articles/{spec.slug}.md)\n\n{body_en}\n"
        )
        output_path_en = ARTICLES_EN_DIR / f"{spec.slug}.en.md"
        output_path_en.write_text(markdown_en, encoding="utf-8")
        return {
            "slug": spec.slug,
            "title_zh": title_zh,
            "title_en": title_en,
            "source_url": source_url,
            "published_at": published_at,
            "path": f"articles/{spec.slug}.md",
            "path_en": f"articles-en/{spec.slug}.en.md",
        }

    def fetch_html(self, url: str) -> etree._Element:
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        return lhtml.fromstring(response.content)

    def extract_title(self, doc: etree._Element) -> str:
        title = normalize_space(doc.xpath("string(//h1)"))
        if title:
            return html.unescape(title)
        title = normalize_space(doc.xpath("string(//title)"))
        if title:
            return html.unescape(title)
        raise RuntimeError("Unable to extract title")

    def extract_published_at(self, doc: etree._Element) -> str:
        candidates = (
            doc.xpath("//meta[@property='article:published_time']/@content")
            or doc.xpath("//time/@datetime")
            or doc.xpath("//meta[@name='date']/@content")
        )
        if not candidates:
            return ""
        return candidates[0][:10]

    def extract_source_site(self, url: str) -> str:
        host = urlparse(url).netloc.lower()
        if "nvidia.com" in host:
            return "NVIDIA Technical Blog"
        if "pytorch.org" in host:
            return "PyTorch Blog"
        return "Colfax Research"

    def extract_content_root(self, doc: etree._Element, url: str) -> etree._Element:
        host = urlparse(url).netloc.lower()
        if "research.colfax-intl.com" in host:
            nodes = doc.xpath("//*[contains(@class,'wp-block-post-content')]")
        elif "nvidia.com" in host:
            nodes = doc.xpath("//*[contains(@class,'entry-content')]")
        else:
            nodes = doc.xpath("//article") or doc.xpath("//main")
        if not nodes:
            raise RuntimeError(f"Unable to find article body for {url}")
        content = nodes[0]
        self.strip_noise(content)
        return content

    def strip_noise(self, content: etree._Element) -> None:
        xpath = (
            ".//*[contains(@class,'sharedaddy') or contains(@class,'jetpack') "
            "or contains(@class,'wp-block-buttons') or contains(@class,'newsletter') "
            "or contains(@class,'subscribe') or contains(@class,'author') "
            "or contains(@class,'related') or contains(@class,'post-rate')]"
        )
        for node in content.xpath(xpath):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)

    def render_blocks(
        self,
        content: etree._Element,
        spec: ArticleSpec,
        base_url: str,
        translate: bool,
    ) -> list[str]:
        blocks: list[str] = []
        for child in content:
            if not isinstance(child.tag, str):
                continue
            blocks.extend(self.render_block(child, spec, base_url, translate))
        return self.clean_blocks(blocks)

    def render_block(
        self,
        node: etree._Element,
        spec: ArticleSpec,
        base_url: str,
        translate: bool,
    ) -> list[str]:
        tag = node.tag.lower()
        classes = " ".join(node.get("class", "").split())
        if "sharedaddy" in classes or "jetpack" in classes:
            return []

        if tag in {"h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            text = self.render_inline(node, base_url, translate).strip()
            return [f"{'#' * level} {text}"] if text else []

        if tag == "p":
            text = self.render_inline(node, base_url, translate).strip()
            return [text] if text else []

        if tag in {"ul", "ol"}:
            return [self.render_list(node, base_url, ordered=(tag == "ol"), translate=translate)]

        if tag == "figure" or "wp-block-image" in classes:
            block = self.render_figure(node, spec, base_url, translate)
            return [block] if block else []

        if tag == "div" and "wp-block-syntaxhighlighter-code" in classes:
            return [self.render_code_block(node)]

        if tag == "pre":
            return [self.render_pre(node)]

        if tag == "blockquote":
            text = self.render_nested_blocks(node, spec, base_url, translate)
            lines = [f"> {line}" if line else ">" for line in text.splitlines()]
            return ["\n".join(lines).strip()]

        if tag == "table":
            return [self.render_table(node, base_url, translate)]

        if tag == "hr":
            return ["---"]

        if tag == "div":
            if node.xpath(".//pre"):
                return [self.render_code_block(node)]
            if node.xpath(".//img"):
                block = self.render_figure(node, spec, base_url, translate)
                return [block] if block else []
            return self.render_generic_container(node, spec, base_url, translate)

        if tag in {"section", "article"}:
            return self.render_generic_container(node, spec, base_url, translate)

        return []

    def render_generic_container(
        self,
        node: etree._Element,
        spec: ArticleSpec,
        base_url: str,
        translate: bool,
    ) -> list[str]:
        blocks: list[str] = []
        for child in node:
            if isinstance(child.tag, str):
                blocks.extend(self.render_block(child, spec, base_url, translate))
        if blocks:
            return blocks
        text = self.render_inline(node, base_url, translate).strip()
        return [text] if text else []

    def render_nested_blocks(
        self,
        node: etree._Element,
        spec: ArticleSpec,
        base_url: str,
        translate: bool,
    ) -> str:
        nested = self.render_generic_container(node, spec, base_url, translate)
        return "\n\n".join(nested)

    def render_list(self, node: etree._Element, base_url: str, ordered: bool, translate: bool) -> str:
        lines = []
        for idx, li in enumerate(node.xpath("./li"), start=1):
            text = self.render_inline(li, base_url, translate).strip()
            if not text:
                continue
            prefix = f"{idx}. " if ordered else "- "
            lines.append(prefix + text.replace("\n", " "))
        return "\n".join(lines)

    def render_figure(
        self,
        node: etree._Element,
        spec: ArticleSpec,
        base_url: str,
        translate: bool,
    ) -> str:
        img = node.xpath(".//img[1]")
        if not img:
            return ""
        img = img[0]
        src = img.get("src")
        if not src:
            return ""
        image_url = self.resolve_asset_url(base_url, src)
        image_path = self.download_image(image_url, spec.slug)
        alt_text = normalize_space(img.get("alt", ""))
        caption = normalize_space(" ".join(node.xpath(".//figcaption//text()")))
        label = caption or alt_text
        if label and translate:
            label = self.translator.translate_text(label)
        return f"![{label}]({image_path})"

    def render_pre(self, node: etree._Element) -> str:
        code = "".join(node.xpath(".//text()"))
        code = code.rstrip("\n")
        lang = self.detect_code_language(node)
        return f"```{lang}\n{code}\n```"

    def render_code_block(self, node: etree._Element) -> str:
        pre = node.xpath(".//pre")
        if pre:
            return self.render_pre(pre[0])
        code = "".join(node.xpath(".//text()")).rstrip("\n")
        lang = self.detect_code_language(node)
        return f"```{lang}\n{code}\n```"

    def detect_code_language(self, node: etree._Element) -> str:
        classes = " ".join(node.get("class", "").split())
        mapping = {
            "language-cpp": "cpp",
            "language-python": "python",
            "language-bash": "bash",
            "language-cuda": "cuda",
            "language-c": "c",
        }
        for source, lang in mapping.items():
            if source in classes:
                return lang
        if "cuda" in classes.lower():
            return "cuda"
        return ""

    def render_table(self, node: etree._Element, base_url: str, translate: bool) -> str:
        rows = []
        for tr in node.xpath(".//tr"):
            cells = tr.xpath("./th|./td")
            values = [
                self.escape_table(self.render_inline(cell, base_url, translate).strip()) for cell in cells
            ]
            if values:
                rows.append(values)
        if not rows:
            return ""
        if len(rows) == 1:
            rows.append(["---"] * len(rows[0]))
        header = rows[0]
        separator = ["---"] * len(header)
        body = rows[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in body:
            padded = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(padded[: len(header)]) + " |")
        return "\n".join(lines)

    def render_inline(self, node: etree._Element, base_url: str, translate: bool) -> str:
        parts: list[str] = []
        if node.text:
            parts.append(self.translate_text_if_needed(node.text, translate))
        for child in node:
            if not isinstance(child.tag, str):
                continue
            parts.append(self.render_inline_child(child, base_url, translate))
            if child.tail:
                parts.append(self.translate_text_if_needed(child.tail, translate))
        joined = "".join(parts)
        joined = re.sub(r"[ \t]+\n", "\n", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return normalize_space(joined) if "\n" not in joined else joined.strip()

    def render_inline_child(self, node: etree._Element, base_url: str, translate: bool) -> str:
        tag = node.tag.lower()
        if tag == "a":
            label = self.render_inline(node, base_url, translate).strip() or urljoin(base_url, node.get("href", ""))
            href = urljoin(base_url, node.get("href", ""))
            return f"[{label}]({href})"
        if tag == "code":
            code = "".join(node.xpath(".//text()")).strip()
            return f"`{code}`"
        if tag in {"strong", "b"}:
            inner = self.render_inline(node, base_url, translate).strip()
            return f"**{inner}**" if inner else ""
        if tag in {"em", "i"}:
            inner = self.render_inline(node, base_url, translate).strip()
            return f"*{inner}*" if inner else ""
        if tag == "br":
            return "\n"
        if tag == "img":
            src = node.get("src")
            if not src:
                return ""
            alt = self.translate_text_if_needed(normalize_space(node.get("alt", "")), translate)
            return f"![{alt}]({self.resolve_asset_url(base_url, src)})"
        if tag in {"span", "sup", "sub"}:
            return self.render_inline(node, base_url, translate)
        return self.render_inline(node, base_url, translate)

    def download_image(self, url: str, slug: str) -> str:
        if url in self.image_cache:
            return self.image_cache[url]
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix or ".png"
        base_name = Path(parsed.path).stem or "image"
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        file_name = f"{base_name}-{digest}{ext}"
        target_dir = IMAGES_DIR / slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_name
        response = self.session.get(url, timeout=60)
        if response.status_code >= 400:
            fallback_url = self.normalize_image_url(url)
            if fallback_url != url:
                response = self.session.get(fallback_url, timeout=60)
                response.raise_for_status()
                url = fallback_url
            else:
                response.raise_for_status()
        target_path.write_bytes(response.content)
        rel_path = f"../images/{slug}/{file_name}"
        self.image_cache[url] = rel_path
        return rel_path

    def normalize_image_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc.endswith("wp.com"):
            path = parsed.path
            embedded = re.match(r"^/((?:github\.com|raw\.githubusercontent\.com|gist\.githubusercontent\.com)/.+)$", path)
            if embedded:
                return "https://" + embedded.group(1)
            path = re.sub(r"^/[^/]+\.wp\.com/", "/", path)
            path = re.sub(r"^/research\.colfax-intl\.com/", "/", path)
            return f"https://research.colfax-intl.com{path}"
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def resolve_asset_url(self, base_url: str, src: str) -> str:
        src = (src or "").strip()
        if not src:
            return src
        src = re.sub(r"^/(github\.com/)", r"\1", src)
        src = re.sub(r"^/(raw\.githubusercontent\.com/)", r"\1", src)
        src = re.sub(r"^/(gist\.githubusercontent\.com/)", r"\1", src)
        if src.startswith("//"):
            return "https:" + src
        if re.match(r"^https?://", src):
            return src
        if re.match(r"^(github\.com|raw\.githubusercontent\.com|raw\.githubusercontent\.com/|gist\.githubusercontent\.com)/", src):
            return "https://" + src
        if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}/", src):
            return "https://" + src
        return urljoin(base_url, src)

    def write_index(self, outputs: list[dict[str, str]]) -> None:
        lines = [
            "# Colfax CUTLASS/CUDA 博客中文翻译",
            "",
            f"共生成 {len(outputs)} 篇文章。",
            "",
        ]
        for item in outputs:
            lines.append(
                f"- {item['published_at']} - [{item['title_zh']}]({item['path']}) "
                f"| 英文稿：[English]({item['path_en']}) | 原文：{item['title_en']} | 来源：{item['source_url']}"
            )
        (ROOT / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def clean_blocks(self, blocks: list[str]) -> list[str]:
        cleaned: list[str] = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if block in {"Share this:", "Like this: Like Loading…"}:
                continue
            cleaned.append(block)
        return cleaned

    @staticmethod
    def escape_yaml(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def escape_table(value: str) -> str:
        return value.replace("|", "\\|")

    def translate_text_if_needed(self, text: str, translate: bool) -> str:
        return self.translator.translate_text(text) if translate else text


def parse_args() -> list[ArticleSpec]:
    requested = set(sys.argv[1:])
    if not requested:
        return ARTICLE_SPECS
    specs = [spec for spec in ARTICLE_SPECS if spec.slug in requested]
    missing = requested - {spec.slug for spec in specs}
    if missing:
        raise SystemExit(f"Unknown slug(s): {', '.join(sorted(missing))}")
    return specs


def main() -> None:
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    ARTICLES_EN_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    translator = Translator()
    converter = SiteConverter(translator)
    converter.build(parse_args())
    translator.save()


if __name__ == "__main__":
    main()
