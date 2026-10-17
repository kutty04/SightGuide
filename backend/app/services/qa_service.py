"""SightGuide Visual Question Answering (Q&A) Service.

Provides conversational visual assistance for blind and low-vision users
by answering questions grounded in:
1. Direct Image-Conditioned Vision (Targeted BLIP vision-language encoder-decoder / Multimodal VLM).
2. Verified Visual Scene Descriptions (BLIP anchor + MMR fusion).
3. Verified Visible OCR Text (RapidOCR).

Scientific Architecture & Grounding Boundary:
- When an image is provided (via camera capture or upload), the service executes
  genuine image-conditioned visual question answering, directly conditioning on
  pixel features using question-targeted attribute prompts.
  It returns capability status 'VISION_CONDITIONED'.
- When only textual scene descriptions and OCR text are available (without image pixels),
  it performs transparent context-grounded fallback and returns 'CONTEXT_GROUNDED_FALLBACK'.
- When models or credentials cannot answer, or visual evidence is insufficient or occluded,
  it returns honest uncertainty statements without speculative hallucinations.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image
import sys

# Ensure backend directory is in sys.path
_backend_dir = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

try:
    from app.schemas import QARequest, QAResponse, QAHistoryItem
    from app.services.caption_pipeline import CaptionPipeline
except ImportError:
    from ..schemas import QARequest, QAResponse, QAHistoryItem
    from .caption_pipeline import CaptionPipeline


QA_SYSTEM_PROMPT_VISION = """You are SightGuide's accurate, descriptive accessibility visual assistant for blind and low-vision users.
You answer user questions using the DIRECT VISUAL OBSERVATIONS extracted from the image pixels by the computer vision model, supplemented by verified scene context and visible OCR text.

FIRST-RESPONSE DESCRIPTIVENESS & GROUNDING POLICY:
1. INFORMATIVE FIRST RESPONSE (NO ELABORATION FOLLOW-UP REQUIRED):
   - Answer the exact question directly first in the opening sentence.
   - When supported by the direct visual observations or scene context, accompany the direct answer with 1 or 2 relevant, clearly visible companion details strictly about the SAME queried attribute in 1 to 2 natural, spoken-friendly sentences:
     * Clothing queries: State the color and integrate visible garment type, style, pattern, or accompanying accessories (e.g., "She is wearing a blue shirt, paired with visible earrings." or "She is wearing a white dress with a subtle floral pattern.").
     * Hair queries: State the hair color along with visible length, style, or texture (e.g., "Her hair is black and worn long.").
     * Facial expression queries: State the visible expression directly with appropriate caution along with visible facial or gaze cues (e.g., "Yes, she appears to be smiling while looking at the camera." or "She has a neutral, serious expression with no visible smile."). Never speculate on internal psychological emotions, personality, or mood.
     * Object queries: State what the object is along with its visible color, distinguishing feature, material, or spatial placement (e.g., "There is a blue water bottle resting on the table.").
     * Surroundings/spatial queries: Describe the relevant spatial relationship or visible background setting directly.
2. STRICT SUBJECT-ATTRIBUTE FOCUS (NO UNRELATED ATTRIBUTE BLEED):
   - For facial expression queries: ONLY describe facial expression, smile, or gaze. DO NOT mention clothing, hair, or accessories.
   - For clothing queries: ONLY describe clothing, color, pattern, or accompanying accessories. DO NOT mention hair or face.
   - For hair queries: ONLY describe hair color, length, or style. DO NOT mention clothing or face.
   - For object queries: ONLY describe the object and its immediate position.
   - Synthesize the answer fluidly: avoid repeating the same statement twice. Keep answers focused (1 to 2 sentences).
3. STRICT ANTI-HALLUCINATION & HONESTY:
   - ONLY include details directly corroborated by the visual observations or verified scene context.
   - NEVER invent fabrics, textures, patterns, facial expressions, emotions, or invisible background items that are not in the evidence.
   - If only part of an item is visible, state what is clearly observable and note what is obscured.
4. OUT OF FRAME, OCCLUDED, OR ABSENT:
   - If the requested detail (such as shoes or lower body in a portrait/headshot, or an absent object) is outside the camera frame or absent from the evidence, state honestly:
     "Based on the visual evidence, [detail] cannot be seen / is out of frame in this image."
   - In headshots, portraits, or bust crops, never extrapolate or assume lower-body clothing.
5. EXPERIMENTAL DETECTOR EVIDENCE:
   - If EXPERIMENTAL OBJECT DETECTOR EVIDENCE is present, treat it as auxiliary, tentative support for object presence or general location.
   - Never treat lack of a detection as proof of absence."""


QA_SYSTEM_PROMPT_CONTEXT = """You are SightGuide's accurate, descriptive accessibility visual assistant for blind and low-vision users.
You answer user questions strictly based on two sources of verified visual evidence provided below:
1. Verified Visual Scene Description (generated by computer vision pipeline)
2. Verified Visible OCR Text (extracted from signage, labels, and documents)

FIRST-RESPONSE DESCRIPTIVENESS & STRICT GROUNDING POLICY:
1. ONLY state facts directly corroborated by the verified scene description or visible OCR text.
2. Answer the exact question directly first, followed by relevant visible companion details (such as color, pattern, style, or setting) if corroborated by the verified evidence, in 1 to 2 clear, spoken-friendly sentences without requiring follow-up prompts.
3. If the user asks about an attribute, object, person, identity, color, clothing item, brand, or location that is NOT mentioned in the evidence, YOU MUST STATE:
   "Based on the verified scene description and visible text, this cannot be determined from the available visual evidence."
4. In portraits, headshots, or cropped upper-body views, NEVER assume or state lower-body attire (e.g. jeans, pants, shoes).
5. Do NOT speculate, hallucinate, or pretend to inspect raw camera pixels directly.
6. If the question asks about text, numbers, signs, or words, cite the visible OCR text accurately."""

QA_MODEL = "openai/gpt-oss-20b"
QA_TEMPERATURE = 0.1
QA_MAX_TOKENS = 600
MAX_IMAGE_PAYLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


class QAService:
    _instance: Optional[QAService] = None

    @classmethod
    def get_instance(cls) -> QAService:
        if cls._instance is None:
            cls._instance = QAService()
        return cls._instance

    def _decode_image(self, image_data: Optional[str]) -> Optional[Image.Image]:
        """Safely decode base64 data URL or raw base64 string into a PIL Image."""
        if not image_data or not isinstance(image_data, str):
            return None

        clean_data = image_data.strip()
        if not clean_data:
            return None

        # Check payload size
        if len(clean_data) > MAX_IMAGE_PAYLOAD_BYTES:
            print("[QAService] Warning: Image payload exceeded maximum allowed size.")
            return None

        # Strip Data URL header if present (e.g. 'data:image/jpeg;base64,')
        if "," in clean_data and clean_data.startswith("data:image"):
            clean_data = clean_data.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(clean_data)
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            return image
        except Exception as exc:
            print(f"[QAService] Warning: Could not decode image payload: {exc}")
            return None

    def _build_targeted_blip_prefixes(self, question: str) -> Tuple[List[str], bool, Optional[str]]:
        """Dynamically build question-conditioned targeted prefixes for BLIP visual reasoning."""
        q = question.strip().lower()
        prefixes = []

        # 1. Garments and clothing terms
        clothing_keywords = [
            'dress', 'shirt', 't-shirt', 'top', 'blouse', 'skirt', 'jacket', 'coat',
            'sweater', 'hoodie', 'pants', 'jeans', 'trousers', 'shorts',
            'shoes', 'boots', 'sneakers', 'hat', 'cap', 'scarf', 'tie',
            'suit', 'uniform', 'costume', 'attire', 'outfit', 'necklace', 'earrings'
        ]
        detected_garments = [g for g in clothing_keywords if re.search(r'\b' + g + r'\b', q)]
        primary_garment = detected_garments[0] if detected_garments else None

        # 2. Person terms
        person_match = re.search(r'\b(girl|woman|lady|man|boy|guy|child|baby|person|people)\b', q)
        person_noun = person_match.group(1) if person_match else 'person'
        person_phrase = f'the {person_noun} are' if person_noun in ['people', 'men', 'women'] else f'the {person_noun} is'

        # 3. Intent flags
        is_color = any(w in q for w in ['color', 'colour', 'shade', 'what color', 'what colour'])
        is_wearing = any(w in q for w in ['wear', 'wearing', 'dressed', 'clothing', 'clothes', 'outfit', 'attire'])
        is_pattern = any(w in q for w in ['pattern', 'print', 'design', 'floral', 'stripe', 'striped', 'checkered', 'texture', 'material', 'fabric', 'style', 'sleeves'])
        is_lower_body = any(w in q for w in ['shoe', 'shoes', 'footwear', 'feet', 'boot', 'boots', 'sneaker', 'sneakers', 'pants', 'jeans', 'trousers', 'legs'])
        is_bottle = any(w in q for w in ['bottle', 'water bottle', 'cup', 'glass', 'can', 'mug'])
        is_table = any(w in q for w in ['table', 'desk', 'counter', 'shelf'])
        is_holding = any(w in q for w in ['holding', 'in hand', 'carrying', 'holds'])
        is_hair = any(re.search(r'\b' + w + r'\b', q) for w in [
            'hair', 'hairstyle', 'haircut', 'curly', 'straight', 'blonde', 'brunette',
            'ponytail', 'braid', 'bald', 'bangs', 'locks'
        ])
        is_expression = self._is_facial_expression_query(q)

        # Build question-conditioned prefixes:
        # A. Targeted facial expression / smile queries
        if is_expression:
            is_smile_specific = any(w in q for w in ['smile', 'smiling', 'grin', 'grinning', 'laugh', 'laughing'])
            if is_smile_specific:
                prefixes.append(f'{person_phrase} looking at the camera with a')
                prefixes.append(f'the expression on the {person_noun}\'s face is')
                prefixes.append(f'the facial expression of {person_phrase}')
            else:
                prefixes.append(f'the expression on the {person_noun}\'s face is')
                prefixes.append(f'{person_phrase} looking at the camera with a')
                prefixes.append(f'the facial expression of {person_phrase}')

        # B. Hair & Hairstyle queries
        if is_hair:
            if is_color:
                prefixes.append(f"the {person_noun}'s hair color is")
                prefixes.append(f'the {person_noun} has')
                prefixes.append(f'the {person_noun} with')
                prefixes.append(f"the {person_noun}'s hair is")
            else:
                prefixes.append(f"the {person_noun}'s hairstyle is")
                prefixes.append(f'the {person_noun} has')
                prefixes.append(f'the {person_noun} with')
                prefixes.append(f"the {person_noun}'s hair is")

        # C. Targeted garment & pattern queries
        if primary_garment:
            verb = 'are' if primary_garment in ['shoes', 'boots', 'sneakers', 'pants', 'jeans', 'trousers', 'shorts', 'earrings'] else 'is'
            if is_color or is_pattern:
                prefixes.append(f'the color of the {primary_garment} is')
                prefixes.append(f'{person_phrase} wearing a {primary_garment}')
                prefixes.append(f'the {primary_garment} {verb}')
                prefixes.append(f'the pattern on the {primary_garment} is')
            else:
                prefixes.append(f'{person_phrase} wearing a')
                prefixes.append(f'the {primary_garment} {verb}')
                prefixes.append(f'the color of the {primary_garment} is')
        elif is_wearing or is_pattern:
            prefixes.append(f'{person_phrase} wearing a')
            prefixes.append('the clothing in the image is')
            prefixes.append('the person is wearing')

        # D. Specific object queries (e.g. water bottle, cup)
        if is_bottle:
            b_term = 'water bottle' if 'water' in q else 'bottle'
            if is_color:
                prefixes.append(f'the color of the {b_term} is')
                prefixes.append(f'the {b_term} is')
            else:
                prefixes.append(f'the {b_term} is')
                prefixes.append(f'the color of the {b_term} is')

        # E. Lower body / footwear
        if is_lower_body:
            prefixes.append('the shoes are')
            prefixes.append('the footwear is')
            prefixes.append('the lower body shows')

        # F. Table / surface objects
        if is_table:
            prefixes.append('on the table there is a')
            prefixes.append('the items on the table include')

        # G. Holding / carrying
        if is_holding:
            prefixes.append(f'{person_phrase} holding a')

        # H. Specific object color queries (e.g., 'what color is the car?', 'color of the flower')
        if is_color and not primary_garment and not is_bottle and not is_hair:
            m = re.search(r'(?:color of (?:the |a |an )?|what color is (?:the |a |an )?)([\w\s]{2,20}?)(?:\?|$|\s+in|\s+on)', q)
            if m:
                raw_target = m.group(1).strip()
                # Clean possessive pronouns to prevent ungrammatical 'the her hair'
                target = re.sub(r'^(?:her|his|their|its|the|a|an)\s+', '', raw_target).strip()
                if target not in ['this', 'that', 'it', 'there', 'image', 'picture', 'photo']:
                    prefixes.append(f'the color of the {target} is')
                    prefixes.append(f'the {target} is')
            prefixes.append('the color of the object is')
            prefixes.append('the main colors in the scene are')

        # I. Background / setting
        if any(w in q for w in ['background', 'surrounding', 'surroundings', 'setting', 'environment']):
            prefixes.append('the background and surroundings show')

        # J. Scene-level anchors
        if person_match and not is_wearing and not is_expression and not is_hair:
            prefixes.append(f'{person_phrase} in the photo is')
        prefixes.append('in this image,')
        prefixes.append('a photo of')

        # Deduplicate while preserving priority order
        seen = set()
        deduped = []
        for p in prefixes:
            p_clean = p.strip()
            if p_clean and p_clean not in seen:
                seen.add(p_clean)
                deduped.append(p_clean)

        return deduped[:4], is_lower_body, primary_garment

    @staticmethod
    def _is_facial_expression_query(question: str) -> bool:
        """Detect whether a question asks about facial expressions, smiling, or face features."""
        q = question.lower()
        return any(re.search(r'\b' + w + r'\b', q) for w in [
            'smile', 'smiling', 'expression', 'expressions', 'facial', 'face', 'frown', 'frowning',
            'grin', 'grinning', 'laugh', 'laughing', 'happy', 'sad', 'neutral', 'serious'
        ])

    def _query_blip_vision(self, image: Image.Image, question: str) -> dict:
        """Run question-conditioned BLIP inference against image pixels for targeted attributes."""
        is_expression = self._is_facial_expression_query(question)
        pipeline = CaptionPipeline.get_instance()
        if not pipeline.blip_loaded or pipeline.blip_model is None or pipeline.blip_processor is None:
            return {"available": False, "findings": [], "is_lower_body_query": False, "target_garment": None, "is_expression_query": is_expression}

        prefixes, is_lower_body, target_garment = self._build_targeted_blip_prefixes(question)
        findings = []

        try:
            device = pipeline.device
            for prefix in prefixes:
                inputs = pipeline.blip_processor(image, prefix, return_tensors="pt").to(device)
                out = pipeline.blip_model.generate(**inputs, max_new_tokens=35, num_beams=3)
                decoded = pipeline.blip_processor.decode(out[0], skip_special_tokens=True).strip()
                if decoded and decoded not in findings:
                    findings.append(decoded)

            return {
                "available": True,
                "findings": findings,
                "is_lower_body_query": is_lower_body,
                "target_garment": target_garment,
                "is_expression_query": is_expression
            }
        except Exception as exc:
            print(f"[QAService] BLIP visual query error: {exc}")
            return {
                "available": False,
                "findings": [],
                "is_lower_body_query": is_lower_body,
                "target_garment": target_garment,
                "is_expression_query": is_expression
            }

    def answer_question(
        self,
        request: QARequest,
        api_key: Optional[str] = None
    ) -> QAResponse:
        """Answer user questions about the current image using grounded visual evidence."""
        start_time = time.time()
        question = request.question.strip()
        scene_cap = (request.scene_caption or "").strip()
        ocr_text = (request.ocr_text or "").strip()
        history = request.history or []

        if not question:
            return QAResponse(
                question="",
                answer="No question was provided. Please ask a question about the image.",
                capability="CONTEXT_GROUNDED_FALLBACK",
                is_uncertain=True,
                grounded_sources=[],
                evidence_used=[],
                limitation_notice="Empty question string received.",
                model_name=None,
                execution_time_seconds=round(time.time() - start_time, 3)
            )

        # Resolve Groq API key
        if api_key is not None:
            key = api_key.strip()
        else:
            key = os.environ.get("GROQ_API_KEY", "").strip()

        # Step 1: Check for Image Data for Direct Vision Conditioning
        pil_image = self._decode_image(request.image_data)

        if pil_image is not None:
            # Execute genuine image-conditioned VLM pipeline
            return self._answer_with_vision(
                pil_image=pil_image,
                question=question,
                scene_caption=scene_cap,
                ocr_text=ocr_text,
                history=history,
                start_time=start_time,
                groq_key=key
            )

        # Step 2: Fallback to Context-Grounded Pipeline when no image pixels are provided
        return self._answer_with_context_fallback(
            question=question,
            scene_caption=scene_cap,
            ocr_text=ocr_text,
            history=history,
            start_time=start_time,
            groq_key=key
        )

    def _answer_with_vision(
        self,
        pil_image: Image.Image,
        question: str,
        scene_caption: str,
        ocr_text: str,
        history: List[QAHistoryItem],
        start_time: float,
        groq_key: str
    ) -> QAResponse:
        """Answer question conditioned directly on image pixels using question-targeted BLIP + Groq synthesis."""
        blip_res = self._query_blip_vision(pil_image, question)
        if isinstance(blip_res, dict):
            visual_findings = blip_res.get("findings", [])
            is_lower_body = blip_res.get("is_lower_body_query", False)
            is_expression = blip_res.get("is_expression_query", False)
        elif isinstance(blip_res, list):
            visual_findings = blip_res
            is_lower_body = False
            is_expression = False
        else:
            visual_findings = []
            is_lower_body = False
            is_expression = False

        # Evidence tracking
        evidence_used = []
        for finding in visual_findings:
            evidence_used.append(f"Visual Observation: {finding[:120]}")
        if scene_caption:
            evidence_used.append(f"Scene Context: {scene_caption[:100]}")
        if ocr_text:
            evidence_used.append(f"OCR Text: {ocr_text[:100]}")

        # Step 1.5: Optional Experimental Object Detection (YOLOv8n ONNX)
        detector_sources = []
        detector_evidence_str = ""
        detector_dets = []
        try:
            from app.services.detector_service import DetectorService
            detector_svc = DetectorService.get_instance()
            if detector_svc.is_feature_enabled() and detector_svc.is_available:
                should_route, target_classes, _ = detector_svc.route_question(question)
                if should_route:
                    detector_dets = detector_svc.detect_objects(pil_image, conf_threshold=0.55, filter_classes=target_classes)
                    detector_evidence_str = detector_svc.format_evidence_block(detector_dets, target_classes)
                    for d in detector_dets:
                        evidence_used.append(f"Experimental Detector: {d['label']} (conf: {d['confidence']:.2f}, {d['region']})")
                    detector_sources.append("experimental_detector")
        except Exception as det_exc:
            print(f"[QAService] Experimental detector step error: {det_exc}")

        # Check for unobservable queries on cropped portraits / headshots
        # Only declare out-of-frame if lower body was specifically queried AND evidence indicates crop
        all_obs_text = " ".join(visual_findings).lower()
        shoes_not_visible = any(
            phrase in all_obs_text
            for phrase in ["not available", "not visible", "out of frame", "no shoes", "not shown"]
        )
        is_headshot_or_crop = any(
            w in (scene_caption + " " + all_obs_text).lower()
            for w in ["portrait", "headshot", "close-up", "close up", "face", "avatar", "upper-body", "bust"]
        )

        if is_lower_body and (shoes_not_visible or is_headshot_or_crop):
            elapsed = round(time.time() - start_time, 3)
            return QAResponse(
                question=question,
                answer="Based on the visual evidence, the lower body and shoes are out of frame and cannot be seen in this image.",
                capability="VISION_CONDITIONED",
                is_uncertain=True,
                grounded_sources=["visual_image", "scene_caption"],
                evidence_used=evidence_used,
                limitation_notice="Visual inspection detected that the requested lower-body attribute is outside the camera crop.",
                model_name="Salesforce/blip-image-captioning-base",
                execution_time_seconds=elapsed
            )

        # Facial expression grounding and face-presence checks
        if is_expression:
            # Check verified scene context and detector evidence for person/face presence
            has_person_in_scene = any(
                re.search(r'\b' + w + r'\b', scene_caption.lower())
                for w in ["person", "people", "man", "woman", "girl", "boy", "someone", "human", "bride", "child", "avatar", "lady", "guy", "face", "portrait", "headshot"]
            ) if scene_caption else False
            has_person_detector = any(d.get("label") == "person" for d in detector_dets)

            if scene_caption:
                has_person_evidence = has_person_in_scene or has_person_detector
            elif detector_dets:
                has_person_evidence = has_person_detector
            else:
                has_person_evidence = any(
                    any(re.search(r'\b' + w + r'\b', f.lower()) for w in ["person", "people", "man", "woman", "girl", "boy", "someone", "human", "bride", "child", "avatar", "lady", "guy"])
                    for f in visual_findings
                    if not any(f.lower().startswith(prefix) for prefix in ["the person is", "the girl is", "the woman is", "the facial expression", "the face of", "a portrait of"])
                )

            if not has_person_evidence:
                elapsed = round(time.time() - start_time, 3)
                return QAResponse(
                    question=question,
                    answer="Based on the visual evidence, no person or visible face is present in this image.",
                    capability="VISION_CONDITIONED",
                    is_uncertain=True,
                    grounded_sources=["visual_image", "scene_caption"],
                    evidence_used=evidence_used,
                    limitation_notice="Visual inspection detected no person or face in the image.",
                    model_name="Salesforce/blip-image-captioning-base",
                    execution_time_seconds=elapsed
                )

            base_scene_text = (scene_caption + " " + " ".join([f for f in visual_findings if not any(f.lower().startswith(prefix) for prefix in ["the person is smiling", "the girl is smiling", "the woman is smiling", "the facial expression"])])).lower()

            has_face_in_context = bool(
                re.search(r'\b(face|facial|portrait|headshot|eyes|mouth|lips|smile|smiling)\b', scene_caption.lower())
            )
            is_face_out_of_frame = bool(
                re.search(r'\b(torso|lower[\s-]body|shoes|boots|feet|legs|pants|jeans|from behind|facing away|back turned|silhouette|shadow|backlit|obscured)\b', scene_caption.lower())
            ) and not has_face_in_context

            if is_face_out_of_frame:
                elapsed = round(time.time() - start_time, 3)
                return QAResponse(
                    question=question,
                    answer="Based on the visual evidence, the person's face is out of frame, in silhouette, or obscured, so their facial expression cannot be determined.",
                    capability="VISION_CONDITIONED",
                    is_uncertain=True,
                    grounded_sources=["visual_image", "scene_caption"],
                    evidence_used=evidence_used,
                    limitation_notice="Visual inspection detected that the face is obscured or out of frame.",
                    model_name="Salesforce/blip-image-captioning-base",
                    execution_time_seconds=elapsed
                )

        # If Groq is available, synthesize a polished, spoken-friendly answer
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)

                obs_str = "\n".join(f"- {f}" for f in visual_findings) if visual_findings else "[No direct visual observation available]"
                evidence_block = f"""DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
{obs_str}

VERIFIED SCENE CONTEXT:
{scene_caption or '[No verified scene context]'}

VERIFIED VISIBLE OCR TEXT:
{ocr_text or '[No visible text detected]'}"""

                if detector_evidence_str:
                    evidence_block += f"""

EXPERIMENTAL OBJECT DETECTOR EVIDENCE (YOLOv8n ONNX, AUXILIARY ONLY):
{detector_evidence_str}"""

                messages: List[dict[str, str]] = [
                    {"role": "system", "content": QA_SYSTEM_PROMPT_VISION}
                ]

                # Include recent history turns (last 2 turns to prevent stale attribute bleed)
                for turn in history[-2:]:
                    messages.append({"role": "user", "content": turn.question})
                    messages.append({"role": "assistant", "content": turn.answer})

                user_content = f"""EVIDENCE:
{evidence_block}

USER QUESTION: {question}"""

                messages.append({"role": "user", "content": user_content})

                response = client.chat.completions.create(
                    model=QA_MODEL,
                    messages=messages,
                    max_tokens=QA_MAX_TOKENS,
                    temperature=QA_TEMPERATURE
                )
                raw_answer = response.choices[0].message.content.strip().strip('"')
                elapsed = round(time.time() - start_time, 3)

                # Fallback to direct observation if LLM generated empty string
                if not raw_answer and visual_findings:
                    raw_answer = self._best_direct_finding_sentence(visual_findings, question)

                is_uncertain = any(
                    phrase in raw_answer.lower()
                    for phrase in ["cannot be determined", "out of frame", "not visible", "not mentioned", "insufficient evidence", "cannot be seen"]
                )

                sources = ["visual_image"]
                if detector_sources:
                    sources.extend(detector_sources)
                if scene_caption:
                    sources.append("scene_caption")
                if ocr_text and any(w.lower() in raw_answer.lower() for w in ocr_text.split() if len(w) > 2):
                    sources.append("ocr_text")

                limitation = (
                    "Targeted visual observations extracted from image pixels via BLIP vision encoder with auxiliary experimental YOLOv8n detector evidence."
                    if detector_sources
                    else "Targeted visual observations extracted from image pixels via BLIP vision encoder with grounded synthesis."
                )

                return QAResponse(
                    question=question,
                    answer=raw_answer,
                    capability="VISION_CONDITIONED",
                    is_uncertain=is_uncertain,
                    grounded_sources=sources,
                    evidence_used=evidence_used,
                    limitation_notice=limitation,
                    model_name=f"BLIP-vlm + {QA_MODEL}",
                    execution_time_seconds=elapsed
                )
            except Exception as exc:
                print(f"[QAService] Groq synthesis failed, using direct BLIP observation: {exc}")

        # Fallback: direct BLIP observation formatting without Groq
        elapsed = round(time.time() - start_time, 3)
        if visual_findings:
            answer = self._best_direct_finding_sentence(visual_findings, question)
            if detector_dets:
                d0 = detector_dets[0]
                answer += f" Additionally, an experimental detector identified a {d0['label']} ({int(d0['confidence']*100)}% confidence)."
            uncertain = False
        elif detector_dets:
            d0 = detector_dets[0]
            answer = f"Based on experimental detector evidence, a {d0['label']} was detected with {int(d0['confidence']*100)}% confidence in the {d0['region']} area of the image."
            uncertain = False
        else:
            answer = "Based on the visual evidence, this detail could not be reliably determined from the image."
            uncertain = True

        fallback_sources = ["visual_image"]
        if detector_sources:
            fallback_sources.extend(detector_sources)

        return QAResponse(
            question=question,
            answer=answer,
            capability="VISION_CONDITIONED",
            is_uncertain=uncertain,
            grounded_sources=fallback_sources,
            evidence_used=evidence_used,
            limitation_notice="Targeted visual observations extracted from image pixels via BLIP vision encoder.",
            model_name="Salesforce/blip-image-captioning-base",
            execution_time_seconds=elapsed
        )

    def _best_direct_finding_sentence(self, findings: List[str], question: str) -> str:
        """Format the most relevant targeted visual observation into a clean spoken sentence."""
        if not findings:
            return "Based on the visual evidence, this detail could not be reliably determined."

        # Clean stuttering/repeating phrases from beam search in findings
        cleaned_findings = []
        for f in findings:
            text = f.strip()
            # Collapse immediate stutter repetitions
            text = re.sub(r"(\b.+?\b)(?:,\s*\1)+", r"\1", text)
            cleaned_findings.append(text)

        # If question is asking specifically about smiling
        is_smile_query = any(w in question.lower() for w in ['smile', 'smiling', 'grin', 'grinning', 'laugh', 'laughing'])
        if is_smile_query:
            smiling_findings = [f for f in cleaned_findings if any(s in f.lower() for s in ['smil', 'grin', 'laugh'])]
            if smiling_findings:
                return "The person appears to be smiling while looking at the camera."
            else:
                return "The person does not appear to be smiling and has a neutral or serious facial expression."

        # Score findings by relevance to the question
        q_words = set(re.findall(r'\b\w+\b', question.lower())) - {'what', 'is', 'the', 'of', 'in', 'her', 'his', 'a', 'an', 'are'}
        best_finding = cleaned_findings[0]
        best_score = -1
        for f in cleaned_findings:
            f_words = set(re.findall(r'\b\w+\b', f.lower()))
            overlap = len(q_words & f_words)
            if overlap > best_score:
                best_score = overlap
                best_finding = f

        text = best_finding.strip()
        # Clean up punctuation and spacing around punctuation
        text = re.sub(r"\s+([.,!?;'])", r"\1", text)
        if text:
            text = text[0].upper() + text[1:]
            if not text.endswith("."):
                text += "."
        return text

    def _answer_with_context_fallback(
        self,
        question: str,
        scene_caption: str,
        ocr_text: str,
        history: List[QAHistoryItem],
        start_time: float,
        groq_key: str
    ) -> QAResponse:
        """Answer question using verified scene description and OCR text (text-only context fallback)."""
        available_sources = []
        if scene_caption:
            available_sources.append("scene_caption")
        if ocr_text:
            available_sources.append("ocr_text")

        if not groq_key:
            return self._heuristic_grounded_fallback(
                question=question,
                scene_caption=scene_caption,
                ocr_text=ocr_text,
                start_time=start_time,
                reason="No image pixels provided and GROQ_API_KEY is not configured."
            )

        evidence_blocks = []
        if scene_caption:
            evidence_blocks.append(f"VERIFIED SCENE DESCRIPTION:\n{scene_caption}")
        else:
            evidence_blocks.append("VERIFIED SCENE DESCRIPTION:\n[No scene description available]")

        if ocr_text:
            evidence_blocks.append(f"VERIFIED VISIBLE OCR TEXT:\n{ocr_text}")
        else:
            evidence_blocks.append("VERIFIED VISIBLE OCR TEXT:\n[No visible text detected in image]")

        evidence_str = "\n\n".join(evidence_blocks)

        messages: List[dict[str, str]] = [
            {"role": "system", "content": QA_SYSTEM_PROMPT_CONTEXT}
        ]

        for turn in history[-2:]:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})

        user_content = f"""IMAGE EVIDENCE:
{evidence_str}

USER QUESTION: {question}"""

        messages.append({"role": "user", "content": user_content})

        try:
            from groq import Groq
            client = Groq(api_key=groq_key)

            response = client.chat.completions.create(
                model=QA_MODEL,
                messages=messages,
                max_tokens=QA_MAX_TOKENS,
                temperature=QA_TEMPERATURE
            )
            raw_answer = response.choices[0].message.content.strip().strip('"')
            elapsed = round(time.time() - start_time, 3)

            is_uncertain = any(
                term in raw_answer.lower()
                for term in ["cannot be determined", "not mentioned", "not available in", "insufficient evidence", "out of frame"]
            )

            grounded_sources = []
            if scene_caption and any(w.lower() in raw_answer.lower() for w in scene_caption.split() if len(w) > 3):
                grounded_sources.append("scene_caption")
            if ocr_text and any(w.lower() in raw_answer.lower() for w in ocr_text.split() if len(w) > 2):
                grounded_sources.append("ocr_text")
            if not grounded_sources and not is_uncertain:
                grounded_sources = available_sources

            evidence_used = []
            if "scene_caption" in grounded_sources and scene_caption:
                evidence_used.append(f"Scene: {scene_caption[:120]}")
            if "ocr_text" in grounded_sources and ocr_text:
                evidence_used.append(f"Text: {ocr_text[:120]}")

            return QAResponse(
                question=question,
                answer=raw_answer,
                capability="CONTEXT_GROUNDED_FALLBACK",
                is_uncertain=is_uncertain,
                grounded_sources=grounded_sources,
                evidence_used=evidence_used,
                limitation_notice=(
                    "Answers are strictly grounded in verified scene descriptions and RapidOCR text. "
                    "Submit image bytes for direct image-conditioned vision reasoning."
                ),
                model_name=QA_MODEL,
                execution_time_seconds=elapsed
            )
        except Exception as exc:
            return self._heuristic_grounded_fallback(
                question=question,
                scene_caption=scene_caption,
                ocr_text=ocr_text,
                start_time=start_time,
                reason=f"Live LLM call failed: {exc}"
            )

    def _heuristic_grounded_fallback(
        self,
        question: str,
        scene_caption: str,
        ocr_text: str,
        start_time: float,
        reason: str
    ) -> QAResponse:
        """Deterministic heuristic fallback when external LLM is unreachable or unconfigured."""
        elapsed = round(time.time() - start_time, 3)
        q_lower = question.lower()

        # Questions about text / signs
        if any(term in q_lower for term in ["text", "say", "words", "sign", "written", "read", "label", "number", "price"]):
            if ocr_text:
                answer = f"The visible text detected in this image is: \"{ocr_text}\"."
                sources = ["ocr_text"]
                uncertain = False
            else:
                answer = "Based on the verified scene description and visible text, no readable text or signage was detected in this image."
                sources = ["ocr_text"]
                uncertain = True
        # Questions asking for scene description or what's in image
        elif any(term in q_lower for term in ["what is", "describe", "show", "in the image", "see", "picture", "photo"]):
            if scene_caption:
                answer = f"According to the verified visual analysis, the image shows: {scene_caption}."
                sources = ["scene_caption"]
                uncertain = False
            else:
                answer = "Based on the available evidence, no verified visual scene description has been generated yet for this image."
                sources = []
                uncertain = True
        # Questions asking about unobservable items (clothing, brand, specific unmentioned object)
        else:
            answer = "Based on the verified scene description and visible text, this cannot be determined from the available visual evidence."
            sources = []
            uncertain = True

        evidence = []
        if "scene_caption" in sources and scene_caption:
            evidence.append(f"Scene: {scene_caption[:100]}")
        if "ocr_text" in sources and ocr_text:
            evidence.append(f"Text: {ocr_text[:100]}")

        return QAResponse(
            question=question,
            answer=answer,
            capability="CONTEXT_GROUNDED_FALLBACK",
            is_uncertain=uncertain,
            grounded_sources=sources,
            evidence_used=evidence,
            limitation_notice=f"Deterministic grounded fallback ({reason}). Answers are strictly restricted to verified evidence.",
            model_name="heuristic-grounded-fallback",
            execution_time_seconds=elapsed
        )
