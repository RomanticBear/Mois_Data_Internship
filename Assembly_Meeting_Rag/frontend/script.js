/* 국회회의록 AI 인사이트 - 프론트 로직 */

const API_BASE = (() => {
    if (typeof window === 'undefined' || !window.location) return '/api';
    const { protocol, hostname, port, origin } = window.location;
    const apiHost = hostname === 'localhost' ? '127.0.0.1' : hostname;
    const apiOrigin = port && port !== '8000' ? `${protocol}//${apiHost}:8000` : origin;
    return apiOrigin + '/api';
})();

const TEMPLATES = {
    summary: '최근 회의의 주요 내용을 요약해주세요.',
    issue: '최근 회의에서 논의된 주요 쟁점들을 정리해주세요.',
    speaker: '발언자별로 주요 발언 내용을 정리해주세요.',
    material: '회의 중 요구된 자료제출요구 사항을 정리해주세요.',
    next: '다음 회의를 준비하기 위한 포인트를 정리해주세요.'
};

let currentMeetings = [];

function $(id) {
    return document.getElementById(id);
}

function escapeHtml(str) {
    if (str == null) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/* 초기화 */
document.addEventListener('DOMContentLoaded', () => {
    bindEvents();
    loadStats();
    loadMeetings();
});

function bindEvents() {
    $('submitBtn').addEventListener('click', handleQuery);
    $('questionInput').addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleQuery();
        }
    });

    document.querySelectorAll('.pill').forEach(btn => {
        btn.addEventListener('click', () => {
            const t = btn.getAttribute('data-template');
            if (TEMPLATES[t]) {
                $('questionInput').value = TEMPLATES[t];
                $('questionInput').focus();
            }
        });
    });

    $('committeeFilter').addEventListener('change', loadMeetings);
    $('assemblyFilter').addEventListener('change', loadMeetings);
    $('activeOnlyFilter').addEventListener('change', loadMeetings);
    $('refreshBtn').addEventListener('click', () => {
        loadStats();
        loadMeetings();
    });
}

/* 통계: 등록 문서 수만 표시 */
async function loadStats() {
    try {
        const res = await fetch(API_BASE + '/stats');
        if (!res.ok) throw new Error();
        const data = await res.json();
        $('totalDocs').textContent = Number(data.total_documents) || 0;
    } catch (e) {
        $('totalDocs').textContent = '0';
    }
}

/* 회의록 목록 */
async function loadMeetings() {
    const activeOnly = $('activeOnlyFilter').checked;
    const committee = ($('committeeFilter').value || '').trim();
    const assembly = ($('assemblyFilter').value || '').trim();

    const params = new URLSearchParams();
    if (activeOnly !== null && activeOnly !== undefined) params.set('is_active', String(activeOnly));
    if (committee) params.set('committee', committee);
    if (assembly) params.set('assembly_number', assembly);

    try {
        setOverlay(true);
        const res = await fetch(API_BASE + '/meetings?' + params.toString());
        if (!res.ok) throw new Error('목록 조회 실패');
        const list = await res.json();
        currentMeetings = Array.isArray(list) ? list : [];
        currentMeetings.sort((a, b) => {
            const aAsm = (a.assembly_number || '').toString();
            const bAsm = (b.assembly_number || '').toString();
            if (aAsm !== bAsm) return aAsm.localeCompare(bAsm, 'ko');
            const aNum = Number.isFinite(Number(a.meeting_number)) ? Number(a.meeting_number) : 0;
            const bNum = Number.isFinite(Number(b.meeting_number)) ? Number(b.meeting_number) : 0;
            if (aNum !== bNum) return aNum - bNum;
            const aDate = (a.date || '').toString();
            const bDate = (b.date || '').toString();
            return aDate.localeCompare(bDate, 'ko');
        });
        renderMeetings(currentMeetings);
        fillFilters(currentMeetings);
        $('meetingsCount').textContent = currentMeetings.length;
    } catch (e) {
        currentMeetings = [];
        $('meetingsList').innerHTML = '<p class="list-empty">회의록 목록을 불러올 수 없습니다.</p>';
        $('meetingsCount').textContent = '0';
    } finally {
        setOverlay(false);
    }
}

function renderMeetings(meetings) {
    const el = $('meetingsList');
    if (!meetings || meetings.length === 0) {
        el.innerHTML = '<p class="list-empty">등록된 회의록이 없습니다.</p>';
        return;
    }
    el.innerHTML = meetings.map(m => {
        const name = escapeHtml(m.committee || '미지정');
        const sub = escapeHtml([m.assembly_number, m.session_type].filter(Boolean).join(' '));
        const date = escapeHtml(m.date || '');
        const nth = m.meeting_number != null ? m.meeting_number + '차' : '';
        const active = m.is_active ? ' active' : '';
        const status = m.is_active ? '검색 대상' : '보관';
        return `<div class="meeting-item${active}" data-id="${escapeHtml(String(m.id))}">
            <h3>${name}</h3>
            <p>${sub}</p>
            <p>${date} ${nth}</p>
            <span class="status-dot">${status}</span>
        </div>`;
    }).join('');
}

function fillFilters(meetings) {
    const committees = [...new Set((meetings || []).map(m => m.committee).filter(Boolean))];
    const assemblies = [...new Set((meetings || []).map(m => m.assembly_number).filter(Boolean))];

    const selCommittee = $('committeeFilter');
    const curC = selCommittee.value;
    selCommittee.innerHTML = '<option value="">전체</option>' + committees.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
    if (committees.includes(curC)) selCommittee.value = curC;

    const selAssembly = $('assemblyFilter');
    const curA = selAssembly.value;
    selAssembly.innerHTML = '<option value="">전체</option>' + assemblies.map(a => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`).join('');
    if (assemblies.includes(curA)) selAssembly.value = curA;
}

/* 질문 제출 */
async function handleQuery() {
    const input = $('questionInput');
    const question = (input.value || '').trim();
    if (!question) {
        alert('질문을 입력해주세요.');
        input.focus();
        return;
    }

    const submitBtn = $('submitBtn');
    const answerArea = $('answerArea');
    const sourcesArea = $('sourcesArea');

    submitBtn.disabled = true;
    submitBtn.textContent = '처리 중...';
    answerArea.innerHTML = '<div class="answer-empty"><div class="spinner" style="margin:0 auto;"></div><p style="margin-top:0.75rem">답변 생성 중...</p></div>';
    sourcesArea.setAttribute('hidden', '');

    setOverlay(true);

    try {
        const res = await fetch(API_BASE + '/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                include_inactive: $('includeInactive').checked
            })
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || '질문 처리 실패');
        }

        const result = await res.json();
        showAnswer(result);
    } catch (e) {
        answerArea.innerHTML = '<p class="answer-empty" style="color:var(--err)">오류: ' + escapeHtml(e.message) + '</p>';
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '질문 제출';
        setOverlay(false);
    }
}

function showAnswer(result) {
    const answerArea = $('answerArea');
    const sourcesArea = $('sourcesArea');
    const sourcesList = $('sourcesList');

    const text = result.answer || '';
    answerArea.innerHTML = '<div class="answer-content">' + formatAnswer(text) + '</div>';

    const sources = result.sources;
    if (sources && sources.length > 0) {
        sourcesList.innerHTML = sources.map((src, i) => {
            const content = escapeHtml(src.content || src.text || '');
            const page = src.page ? `페이지: ${src.page}` : '';
            const file = src.filename ? escapeHtml(src.filename) : '';
            return `<div class="source-item"><p>${content}</p>${page ? '<p style="font-size:0.75rem;color:var(--text2);margin-top:0.35rem">' + page + '</p>' : ''}${file ? '<p style="font-size:0.75rem;color:var(--text2)">' + file + '</p>' : ''}</div>`;
        }).join('');
        sourcesArea.removeAttribute('hidden');
    } else {
        sourcesArea.setAttribute('hidden', '');
    }

    answerArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function formatAnswer(text) {
    if (!text) return '';
    let s = escapeHtml(text);
    s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\*(.*?)\*/g, '<em>$1</em>');
    s = s.replace(/\n/g, '<br>');
    s = s.replace(/###\s+(.+)/g, '<h3>$1</h3>');
    s = s.replace(/##\s+(.+)/g, '<h2>$1</h2>');
    s = s.replace(/#\s+(.+)/g, '<h1>$1</h1>');
    return s;
}

function setOverlay(show) {
    const el = $('loadingOverlay');
    if (!el) return;
    if (show) el.removeAttribute('hidden');
    else el.setAttribute('hidden', '');
}

window.addEventListener('unhandledrejection', e => {
    setOverlay(false);
});
