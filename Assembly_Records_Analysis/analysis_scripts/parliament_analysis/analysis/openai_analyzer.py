"""
[역할] OpenAI API를 통한 LLM 분석 실행
- analyze_session_summary(): 세션 요약 분석
- analyze_party_positions(): 정당별 입장 분석
- analyze_qa_effectiveness(): QA 효과성 분석
- 프롬프트 작성 및 JSON 응답 파싱 담당
- OpenAI Chat Completions API 호출하여 분석 결과 반환
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable, List, Mapping, Optional, Sequence

from .models import (
    AgendaPartyAnalysis,
    IssueTrend,
    PartyPosition,
    QAAnalysisMetrics,
    SessionSummary,
)


class OpenAISessionAnalyzer:
    """Run LLM analyses over pre-processed session transcripts."""

    def __init__(self, llm_client, *, model: str, temperature: float = 0.3) -> None:
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature

    def analyze_session_summary(
        self,
        *,
        session_name: str,
        session_payload: Mapping[str, object],
        meeting_date: Optional[str] = None,
    ) -> SessionSummary:
        """Generate a high-level session summary using the LLM.

        Parameters
        ----------
        session_payload:
            Pre-aggregated data (e.g., concatenated transcript, metadata) for the
            target session.
        """
        prompt = self._build_session_summary_prompt(session_name, session_payload)

        response_json = self._invoke_llm(prompt)
        summary = SessionSummary(
            session_name=response_json.get("session_name", session_name),
            meeting_date=None,
            key_issues=response_json.get("key_issues", []),
            overall_sentiment=response_json.get("overall_sentiment"),
            raw_summary=response_json.get("session_characteristics"),
            metadata={
                "party_positions_overview": response_json.get("party_positions"),
                "major_conflicts": response_json.get("major_conflicts"),
                "key_events": response_json.get("key_events"),
            },
        )
        if meeting_date:
            summary.metadata["meeting_date"] = meeting_date
        summary.metadata["raw_llm_response"] = response_json
        return summary

    def analyze_party_positions(
        self, *, session_name: str, agenda_payloads: Iterable[Mapping[str, object]]
    ) -> Sequence[AgendaPartyAnalysis]:
        """Derive party stances for each agenda item."""
        analyses: List[AgendaPartyAnalysis] = []
        for payload in agenda_payloads:
            agenda_title = str(payload["agenda_title"])
            prompt = self._build_party_position_prompt(session_name, payload)
            response = self._invoke_llm(prompt)

            party_positions_payload = response.get("party_positions", {})
            positions: List[PartyPosition] = []
            for party_name, details in party_positions_payload.items():
                positions.append(
                    PartyPosition(
                        session_name=session_name,
                        agenda_title=agenda_title,
                        party_name=party_name,
                        stance_label=details.get("stance", ""),
                        key_points=list(details.get("key_points", [])),
                        concerns=list(details.get("concerns", [])),
                        suggestions=list(details.get("suggestions", [])),
                        summary_text=details.get("key_statements"),
                        metadata={
                            "raw_llm_response": details,
                        },
                    )
                )

            analyses.append(
                AgendaPartyAnalysis(
                    session_name=session_name,
                    agenda_title=agenda_title,
                    party_positions=positions,
                    consensus_points=list(response.get("consensus_points", [])),
                    conflict_points=list(response.get("conflict_points", [])),
                    cooperation_level=response.get("cooperation_level"),
                    summary_text=response.get("summary"),
                    metadata={"raw_llm_response": response},
                )
            )
        return analyses

    def analyze_qa_effectiveness(
        self, *, qa_pairs: Iterable[dict]
    ) -> QAAnalysisMetrics:
        """Score question-answer effectiveness."""
        qa_pairs = list(qa_pairs)
        if not qa_pairs:
            raise ValueError("No QA pairs provided for analysis.")

        prompt = self._build_qa_effectiveness_prompt(qa_pairs)
        response = self._invoke_llm(prompt)
        metrics = QAAnalysisMetrics(
            session_name=response.get("session_name") or "",
            total_qa_pairs=response.get("total_qa_pairs", len(qa_pairs)),
            quality_distribution=response.get("quality_distribution", {}),
            question_types=response.get("question_types", {}),
            answer_quality=response.get("answer_quality", {}),
            key_issues=response.get("key_issues", []),
            improvement_suggestions=list(response.get("improvement_suggestions", [])),
            metadata={
                "raw_llm_response": response,
                "sample_qa_pairs": qa_pairs[: len(qa_pairs)],
            },
        )
        if not metrics.session_name and qa_pairs:
            metrics.session_name = qa_pairs[0].get("session_name", "")
        return metrics

    def extract_issue_trends(self, *, session_payload: dict) -> List[IssueTrend]:
        """Optional helper to surface issue trends / sentiment shifts."""
        raise NotImplementedError("LLM issue trend prompt logic to be implemented.")

    @staticmethod
    def as_serializable_dicts(items: Sequence) -> List[dict]:
        """Utility for piping dataclasses to downstream storage layers."""
        return [asdict(item) for item in items]

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_session_summary_prompt(
        self, session_name: str, session_payload: Mapping[str, object]
    ) -> str:
        agenda_stats = session_payload.get("agenda_stats", {})
        agenda_text = "\n".join(
            f"- {agenda}: {count}개 발언"
            for agenda, count in sorted(
                agenda_stats.items(), key=lambda x: x[1], reverse=True
            )[:10]
        )

        party_stats = session_payload.get("party_stats", {})
        party_text = "\n".join(
            f"- {party}: {count}개 발언"
            for party, count in sorted(
                party_stats.items(), key=lambda x: x[1], reverse=True
            )
        )

        speeches = session_payload.get("speeches_sample", [])
        speech_texts = []
        for idx, speech in enumerate(speeches, start=1):
            speech_texts.append(
                f"\n[{idx}] 정당: {speech.get('party')}, 발언자: {speech.get('speaker')}"
            )
            speech_texts.append(f"발언: {speech.get('text')}")
        speeches_text = "\n".join(speech_texts)

        prompt = f"""당신은 국회 회의록 분석 전문가입니다. 다음은 {session_name}의 회의록 데이터입니다.

=== 회차 정보 ===
회차: {session_name}
총 발언 수: {session_payload.get('total_speeches')}개 (전체의 대표 샘플: {len(speeches)}개)

=== 안건 통계 ===
{agenda_text if agenda_text else "안건 정보 없음"}

=== 정당별 발언 통계 ===
{party_text}

=== 대표 발언 샘플 ===
{speeches_text}

=== 분석 요청 ===
다음 발언들을 분석하되, **의사진행 발언은 제외**하고 **정책 관련 실질적인 발언만** 포함하여 분석하세요.
- 의사진행 발언: "의석을 정돈해", "회의를 개최", "감사의 말씀", "다음은", "상정합니다" 등
- 정책 관련 발언: 정책 제안, 질의, 건의, 평가, 문제 제기 등

다음 형식의 JSON으로 분석 결과를 제공해주세요:

{{
  "session_name": "{session_name}",
  "key_issues": [
    {{
      "issue": "이슈명",
      "importance": "높음/중간/낮음",
      "description": "이슈에 대한 설명",
      "mentioned_parties": ["정당1", "정당2"]
    }}
  ],
  "party_positions": {{
    "정당명": {{
      "main_concerns": ["관심사1", "관심사2"],
      "key_statements": "주요 발언 요약",
      "stance": "비판적/지지적/중립적/건의적"
    }}
  }},
  "major_conflicts": [
    {{
      "topic": "쟁점명",
      "parties_involved": ["정당1", "정당2"],
      "nature": "비판/협력/토론/질의"
    }}
  ],
  "key_events": [
    {{
      "event": "사건/참사명",
      "description": "설명",
      "response": "국회 대응"
    }}
  ],
  "session_characteristics": "회차의 전반적인 특징 요약"
}}

중요한 점:
1. 핵심 이슈는 3-7개 정도로 추출 (의사진행 발언은 제외)
2. 정당별 관점은 구체적으로 명시 (비판적/지지적/중립적/건의적)
3. 이 회의는 질의-응답 형태이므로 "반대/지지"가 아니라 문제 제기, 건의, 평가 중심으로 분석
4. 주요 쟁점과 협력/대립 관계를 명확히 구분
5. 문맥을 고려하여 실질적인 정책 발언만 포함
6. 한국어로 응답하되, JSON 형식은 정확히 유지
7. 실제 데이터에 기반한 분석만 제공"""
        return prompt

    def _build_party_position_prompt(
        self, session_name: str, agenda_payload: Mapping[str, object]
    ) -> str:
        agenda_title = str(agenda_payload["agenda_title"])
        party_speeches = agenda_payload.get("party_speeches", {})

        prompt_lines = [
            f"""당신은 국회 회의록 분석 전문가입니다. 다음은 {session_name}의 안건 "{agenda_title}"에 대한 정당별 발언입니다.

=== 안건 정보 ===
안건명: {agenda_title}
총 발언 수: {agenda_payload.get('total_speeches', 0)}개

=== 정당별 발언 샘플 ===""",
        ]

        for party, speeches in party_speeches.items():
            prompt_lines.append(f"\n[정당: {party}]")
            for idx, speech in enumerate(speeches, start=1):
                prompt_lines.append(f"{idx}. {speech}")

        prompt_lines.append(
            """
=== 분석 요청 ===
이 회의는 입법 표결이 아니라 질의-응답 형태의 위원회 회의입니다. 
다음 JSON 형식으로 정당별 관점을 분석해주세요:

{
  "agenda": "%s",
  "party_positions": {
    "정당명": {
      "stance": "비판적/지지적/중립적/건의적",
      "key_points": ["주요 포인트1", "주요 포인트2"],
      "concerns": ["우려사항1", "우려사항2"],
      "suggestions": ["제안사항1", "제안사항2"],
      "key_statements": "주요 발언 요약"
    }
  },
  "consensus_points": ["합의점1", "합의점2"],
  "conflict_points": ["대립점1", "대립점2"],
  "cooperation_level": "높음/중간/낮음",
  "summary": "안건에 대한 종합 분석"
}

중요한 점:
1. "반대/지지"가 아니라 "비판적/지지적/중립적/건의적" 관점으로 표현
2. 질의-응답 형태의 회의 특성을 반영하여 문제 제기, 건의, 평가 중심으로 분석
3. 합의점과 대립점을 명확히 구분
4. 협력 수준을 객관적으로 평가
5. 한국어로 응답하되 JSON 형식 유지"""
            % agenda_title
        )

        return "\n".join(prompt_lines)

    def _build_qa_effectiveness_prompt(self, qa_pairs: Sequence[Mapping[str, object]]) -> str:
        qa_sections = []
        for idx, pair in enumerate(qa_pairs[:10], start=1):
            qa_sections.append(
                f"""
[질의-응답 {idx}]
질문자 ({pair.get('question_party')}): {pair.get('questioner')}
질문: {pair.get('question')}
답변자 ({pair.get('answer_party')}): {pair.get('answerer')}
답변: {pair.get('answer')}
"""
            )

        qa_text = "\n".join(qa_sections)
        prompt = f"""당신은 국회 회의록 분석 전문가입니다. 다음은 질의-응답 샘플입니다.

=== 질의-응답 샘플 ===
{qa_text}

=== 분석 요청 ===
다음 JSON 형식으로 질의-응답 효과성을 분석해주세요:

{{
  "session_name": "{qa_pairs[0].get('session_name', '')}",
  "total_qa_pairs": {len(qa_pairs)},
  "quality_distribution": {{
    "high": "고품질 응답 비율 (%)",
    "medium": "중품질 응답 비율 (%)",
    "low": "저품질 응답 비율 (%)"
  }},
  "question_types": {{
    "policy_inquiry": "정책 질의 비율 (%)",
    "fact_checking": "사실 확인 비율 (%)",
    "criticism": "비판 질의 비율 (%)",
    "suggestion": "제안 질의 비율 (%)"
  }},
  "answer_quality": {{
    "completeness": "완성도 평균 (1-10)",
    "specificity": "구체성 평균 (1-10)",
    "responsiveness": "응답성 평균 (1-10)"
  }},
  "key_issues": [
    {{
      "issue": "주요 이슈",
      "qa_count": "질의-응답 수",
      "quality": "평균 품질"
    }}
  ],
  "improvement_suggestions": ["개선 제안1", "개선 제안2"]
}}

중요한 점:
1. 응답 품질을 객관적으로 평가
2. 질문 유형을 명확히 분류
3. 구체적인 개선 제안 제공
4. 한국어로 응답하되 JSON 형식 유지"""
        return prompt

    # ------------------------------------------------------------------
    # LLM invocation wrapper
    # ------------------------------------------------------------------

    def _invoke_llm(self, prompt: str) -> dict:
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 국회 회의록 분석 전문가입니다. 정확하고 구조화된 JSON 형식으로 "
                        "분석 결과를 제공합니다."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content
        return json.loads(result_text)


