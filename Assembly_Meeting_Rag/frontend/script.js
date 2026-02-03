// API 기본 URL
const API_BASE_URL = 'http://localhost:8001/api';

// 질문 템플릿
const questionTemplates = {
    summary: "최근 회의의 주요 내용을 요약해주세요.",
    issue: "최근 회의에서 논의된 주요 쟁점들을 정리해주세요.",
    speaker: "발언자별로 주요 발언 내용을 정리해주세요.",
    material: "회의 중 요구된 자료제출요구 사항을 정리해주세요.",
    next: "다음 회의를 준비하기 위한 포인트를 정리해주세요."
};

// 질문 유형 매핑
const questionTypeMap = {
    summary: "summary",
    issue: "issue",
    speaker: "speaker",
    material: "material_request",
    next: "next_meeting_prep"
};

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadMeetings();
});

// 이벤트 리스너 초기화
function initializeEventListeners() {
    // 업로드 버튼
    document.getElementById('uploadBtn').addEventListener('click', handleUpload);
    
    // 질문 제출 버튼
    document.getElementById('submitBtn').addEventListener('click', handleQuery);
    
    // 템플릿 버튼들
    document.querySelectorAll('.template-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const template = e.target.dataset.template;
            if (questionTemplates[template]) {
                document.getElementById('questionInput').value = questionTemplates[template];
            }
        });
    });
    
    // 필터 변경
    document.getElementById('committeeFilter').addEventListener('change', loadMeetings);
    document.getElementById('assemblyFilter').addEventListener('change', loadMeetings);
    document.getElementById('activeOnlyFilter').addEventListener('change', loadMeetings);
    document.getElementById('refreshBtn').addEventListener('click', loadMeetings);
}

// 회의록 목록 로드
async function loadMeetings() {
    try {
        const isActive = document.getElementById('activeOnlyFilter').checked;
        const committee = document.getElementById('committeeFilter').value;
        const assembly = document.getElementById('assemblyFilter').value;
        
        let url = `${API_BASE_URL}/meetings?`;
        if (isActive !== null) url += `is_active=${isActive}&`;
        if (committee) url += `committee=${encodeURIComponent(committee)}&`;
        if (assembly) url += `assembly_number=${encodeURIComponent(assembly)}&`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('회의록 목록 로드 실패');
        
        const meetings = await response.json();
        displayMeetings(meetings);
        updateFilters(meetings);
        
    } catch (error) {
        console.error('Error loading meetings:', error);
        document.getElementById('meetingsList').innerHTML = 
            '<p style="color: red;">회의록 목록을 불러올 수 없습니다.</p>';
    }
}

// 회의록 목록 표시
function displayMeetings(meetings) {
    const listElement = document.getElementById('meetingsList');
    
    if (meetings.length === 0) {
        listElement.innerHTML = '<p style="color: #999;">등록된 회의록이 없습니다.</p>';
        return;
    }
    
    listElement.innerHTML = meetings.map(meeting => `
        <div class="meeting-item ${meeting.is_active ? 'active' : ''}" 
             data-id="${meeting.id}">
            <h3>${meeting.committee}</h3>
            <p>${meeting.assembly_number} ${meeting.session_type}</p>
            <p>${meeting.date} - ${meeting.meeting_number}차</p>
            <p style="font-size: 0.75em; margin-top: 5px;">
                ${meeting.is_active ? '✓ Active' : '✗ Inactive'}
            </p>
        </div>
    `).join('');
}

// 필터 옵션 업데이트
function updateFilters(meetings) {
    const committees = [...new Set(meetings.map(m => m.committee))];
    const assemblies = [...new Set(meetings.map(m => m.assembly_number))];
    
    const committeeFilter = document.getElementById('committeeFilter');
    const currentCommittee = committeeFilter.value;
    committeeFilter.innerHTML = '<option value="">전체</option>' + 
        committees.map(c => `<option value="${c}">${c}</option>`).join('');
    if (committees.includes(currentCommittee)) {
        committeeFilter.value = currentCommittee;
    }
    
    const assemblyFilter = document.getElementById('assemblyFilter');
    const currentAssembly = assemblyFilter.value;
    assemblyFilter.innerHTML = '<option value="">전체</option>' + 
        assemblies.map(a => `<option value="${a}">${a}</option>`).join('');
    if (assemblies.includes(currentAssembly)) {
        assemblyFilter.value = currentAssembly;
    }
}

// 파일 업로드 처리
async function handleUpload() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('파일을 선택해주세요.');
        return;
    }
    
    if (!file.name.endsWith('.pdf') && !file.name.endsWith('.PDF')) {
        alert('PDF 파일만 업로드 가능합니다.');
        return;
    }
    
    const uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.disabled = true;
    uploadBtn.textContent = '업로드 중...';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        // 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        // 파일명에서 정보 추출: "제22대국회 제415회(임시회) 제1차 행정안전위원회(전체회의) (2024.06.13.) (2).PDF"
        // TODO: 더 정교한 파싱 로직 구현
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '업로드 실패');
        }
        
        const result = await response.json();
        alert('업로드 완료!');
        fileInput.value = '';
        loadMeetings();
        
    } catch (error) {
        console.error('Upload error:', error);
        alert(`업로드 실패: ${error.message}`);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '업로드';
    }
}

// 질문 처리
async function handleQuery() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();
    
    if (!question) {
        alert('질문을 입력해주세요.');
        return;
    }
    
    const submitBtn = document.getElementById('submitBtn');
    const answerArea = document.getElementById('answerArea');
    const sourcesArea = document.getElementById('sourcesArea');
    
    submitBtn.disabled = true;
    submitBtn.textContent = '처리 중...';
    answerArea.innerHTML = '<p class="loading">답변을 생성하는 중...</p>';
    sourcesArea.style.display = 'none';
    
    try {
        const includeInactive = document.getElementById('includeInactive').checked;
        
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                include_inactive: includeInactive
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '질문 처리 실패');
        }
        
        const result = await response.json();
        displayAnswer(result);
        
    } catch (error) {
        console.error('Query error:', error);
        answerArea.innerHTML = `<p style="color: red;">오류: ${error.message}</p>`;
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '질문 제출';
    }
}

// 답변 표시
function displayAnswer(result) {
    const answerArea = document.getElementById('answerArea');
    const sourcesArea = document.getElementById('sourcesArea');
    const sourcesList = document.getElementById('sourcesList');
    
    // 답변 표시
    answerArea.innerHTML = `<div class="answer-content">${escapeHtml(result.answer)}</div>`;
    
    // 근거 문서 표시
    if (result.sources && result.sources.length > 0) {
        sourcesList.innerHTML = result.sources.map((source, index) => `
            <div class="source-item">
                <h4>근거 ${index + 1}</h4>
                <p>${escapeHtml(source.content || '')}</p>
                ${source.page ? `<p style="font-size: 0.8em; color: #666;">페이지: ${source.page}</p>` : ''}
            </div>
        `).join('');
        sourcesArea.style.display = 'block';
    } else {
        sourcesArea.style.display = 'none';
    }
}

// HTML 이스케이프
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

