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
from datetime import datetime
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
        previous_session_summary: Optional[Mapping[str, object]] = None,
    ) -> SessionSummary:
        """Generate a high-level session summary using the LLM.

        Parameters
        ----------
        session_payload:
            Pre-aggregated data (e.g., concatenated transcript, metadata) for the
            target session.
        """
        prompt = self._build_session_summary_prompt(
            session_name, 
            session_payload,
            previous_session_summary=previous_session_summary,
        )

        response_json = self._invoke_llm(prompt)
        
        # meeting_date를 datetime으로 변환
        parsed_meeting_date = None
        if meeting_date:
            try:
                # ISO 형식 문자열을 datetime으로 변환
                if isinstance(meeting_date, str):
                    parsed_meeting_date = datetime.fromisoformat(meeting_date.replace("Z", "+00:00"))
                elif isinstance(meeting_date, datetime):
                    parsed_meeting_date = meeting_date
            except (ValueError, AttributeError):
                # 파싱 실패 시 None 유지
                parsed_meeting_date = None
        
        # LLM 응답에서 key_issues 추출
        key_issues = response_json.get("key_issues", [])
        
        # 정량 통계 계산 (LLM 분석 결과 기반)
        issue_mentions = {
            issue.get("issue", ""): issue.get("mention_count", 0)
            for issue in key_issues
        }
        
        # 기본 통계 (session_payload에서 가져오기)
        total_speeches = session_payload.get("total_speeches", 0)
        party_stats = session_payload.get("party_stats", {})
        qa_pairs_count = session_payload.get("qa_pairs_count", 0)
        avg_speech_length = session_payload.get("avg_speech_length", 0.0)
        
        # 정당별 참여도 비율 계산
        party_participation_ratio = {}
        if total_speeches > 0:
            party_participation_ratio = {
                party: (count / total_speeches) * 100
                for party, count in party_stats.items()
            }
        
        # 정량 통계 구성
        quantitative_stats = {
            "issue_mentions": issue_mentions,
            "party_participation_ratio": party_participation_ratio,
            "total_speeches": total_speeches,
            "qa_pairs_count": qa_pairs_count,
            "avg_speech_length": avg_speech_length,
        }
        
        # 트렌드 분석 추출 (있는 경우만)
        trend_analysis = response_json.get("trend_analysis")
        if trend_analysis is None or (isinstance(trend_analysis, dict) and len(trend_analysis) == 0):
            trend_analysis = None
        
        # 정량적 인사이트 추출
        quantitative_insights = response_json.get("quantitative_insights", {})
        
        summary = SessionSummary(
            session_name=response_json.get("session_name", session_name),
            meeting_date=parsed_meeting_date,
            key_issues=key_issues,
            overall_sentiment=response_json.get("overall_sentiment"),
            raw_summary=response_json.get("session_characteristics"),
            metadata={
                "party_positions_overview": response_json.get("party_positions"),
                "major_conflicts": response_json.get("major_conflicts"),
                "key_events": response_json.get("key_events"),
                "quantitative_stats": quantitative_stats,
                "trend_analysis": trend_analysis,
                "quantitative_insights": quantitative_insights,
            },
        )
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
        self, 
        session_name: str, 
        session_payload: Mapping[str, object],
        previous_session_summary: Optional[Mapping[str, object]] = None,
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

        # 정량 통계 섹션 구성
        total_speeches = session_payload.get("total_speeches", 0)
        qa_pairs_count = session_payload.get("qa_pairs_count", 0)
        avg_speech_length = session_payload.get("avg_speech_length", 0.0)
        party_stats = session_payload.get("party_stats", {})
        
        quantitative_stats_text = f"""
=== 정량적 통계 ===
- 총 발언 수: {total_speeches:,}개
- QA 쌍 수: {qa_pairs_count}개
- 평균 발언 길이: {avg_speech_length:.0f}자
- 정당별 참여도:
"""
        for party, count in sorted(party_stats.items(), key=lambda x: x[1], reverse=True):
            ratio = (count / total_speeches * 100) if total_speeches > 0 else 0
            quantitative_stats_text += f"  - {party}: {count}개 ({ratio:.1f}%)\n"

        # 이전 회차 정보 섹션 구성
        previous_session_text = ""
        if previous_session_summary:
            prev_name = previous_session_summary.get("session_name", "")
            prev_summary = previous_session_summary.get("summary_text", "")
            prev_issues = previous_session_summary.get("key_issues", [])
            prev_party_positions = previous_session_summary.get("party_positions_overview", {})
            prev_quantitative_stats = previous_session_summary.get("quantitative_stats", {})
            
            previous_session_text = f"""
=== 이전 회차 ({prev_name}) 정보 ===
요약: {prev_summary[:500] if prev_summary else "요약 정보 없음"}

주요 이슈 (상위 5개):
"""
            for issue in prev_issues[:5]:
                issue_name = issue.get("issue", "")
                importance = issue.get("importance", "")
                previous_session_text += f"  - {issue_name} ({importance})\n"
            
            if prev_party_positions:
                previous_session_text += "\n정당별 입장 개요:\n"
                for party, data in list(prev_party_positions.items())[:3]:
                    stance = data.get("stance", "") if isinstance(data, dict) else ""
                    previous_session_text += f"  - {party}: {stance}\n"
            
            if prev_quantitative_stats:
                prev_issue_mentions = prev_quantitative_stats.get("issue_mentions", {})
                if prev_issue_mentions:
                    previous_session_text += "\n이슈별 언급 횟수:\n"
                    for issue, count in list(prev_issue_mentions.items())[:5]:
                        previous_session_text += f"  - {issue}: {count}회\n"

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
{quantitative_stats_text}
{previous_session_text}
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
      "mentioned_parties": ["정당1", "정당2"],
      "mention_count": 0
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
  "session_characteristics": "회차의 전반적인 특징 요약"{",\n  \"trend_analysis\": {\n    \"issue_changes\": [\n      {\n        \"issue\": \"이슈명\",\n        \"change\": \"증가/감소/유지\",\n        \"previous_mention_count\": 0,\n        \"current_mention_count\": 0,\n        \"description\": \"변화 설명\"\n      }\n    ],\n    \"party_position_changes\": {\n      \"정당명\": {\n        \"issue\": \"이슈명\",\n        \"change\": \"지지 강화/약화/전환/유지\",\n        \"description\": \"변화 설명\"\n      }\n    },\n    \"quantitative_changes\": {\n      \"speech_count_change\": \"증가/감소/유지 (비율)\",\n      \"question_count_change\": \"증가/감소/유지 (비율)\",\n      \"description\": \"정량적 변화 해석\"\n    }\n  },\n  \"quantitative_insights\": {\n    \"issue_importance_ranking\": [\"이슈1\", \"이슈2\"],\n    \"most_active_party\": \"정당명\",\n    \"key_statistics\": \"주요 통계 해석\"\n  }" if previous_session_summary else ""}
}}

중요한 점:
1. 핵심 이슈는 3-7개 정도로 추출 (의사진행 발언은 제외)
2. 각 이슈에 대해 **mention_count** 필드에 해당 이슈가 발언에서 언급된 횟수를 정확히 계산하여 제공하세요
   - 발언 샘플을 분석하여 해당 이슈와 관련된 발언의 개수를 세어주세요
   - 맥락을 이해하여 정확하게 계산하세요 (단순 키워드 매칭이 아닌 의미 기반)
3. 정당별 관점은 구체적으로 명시 (비판적/지지적/중립적/건의적)
4. 이 회의는 질의-응답 형태이므로 "반대/지지"가 아니라 문제 제기, 건의, 평가 중심으로 분석
5. 주요 쟁점과 협력/대립 관계를 명확히 구분
6. 문맥을 고려하여 실질적인 정책 발언만 포함
7. **정량적 지표 해석**: 이슈별 언급 횟수와 정당별 참여도가 의미하는 바를 해석하여 quantitative_insights에 제공하세요
8. **트렌드 분석** (이전 회차 정보가 있는 경우에만):
   - 의미 있는 비교가 가능한 경우에만 trend_analysis를 제공하세요
   - 완전히 다른 주제를 억지로 비교하거나, 이전 회차 정보가 불충분한 경우 trend_analysis를 null로 설정하세요
   - 의미 있는 경우:
     * 동일하거나 유사한 이슈가 두 회차에서 다뤄진 경우
     * 정당별 입장이 시간에 따라 변화한 경우
     * 정량적 지표의 변화가 의미 있는 경우
   - 이슈별 변화 추이, 정당별 입장 변화, 정량적 변화를 구체적으로 분석하세요
9. 한국어로 응답하되, JSON 형식은 정확히 유지
10. 실제 데이터에 기반한 분석만 제공"""
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


