// API 서버 URL
const API_BASE_URL = 'http://localhost:8000';

// DOM 요소
const chatContainer = document.getElementById('chat-container');
const questionInput = document.getElementById('questionInput');
const sendButton = document.getElementById('sendButton');
const sessionSelect = document.getElementById('sessionSelect');

// 세션 목록 로드
async function loadSessions() {
    try {
        const response = await fetch(`${API_BASE_URL}/sessions`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // 중복 제거 및 정렬
        const sessions = [...new Set(data.sessions || [])].sort();
        
        // 기존 옵션 유지 (전체 옵션)
        const currentValue = sessionSelect.value;
        sessionSelect.innerHTML = '<option value="">전체</option>';
        
        if (sessions.length === 0) {
            console.warn('세션 목록이 비어있습니다.');
            return;
        }
        
        sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session;
            option.textContent = session;
            sessionSelect.appendChild(option);
        });
        
        // 이전 선택값 복원
        if (currentValue) {
            sessionSelect.value = currentValue;
        }
        
        console.log(`세션 목록 로드 완료: ${sessions.length}개`);
    } catch (error) {
        console.error('세션 목록 로드 실패:', error);
        // 사용자에게 알림
        addMessage('세션 목록을 불러올 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.', false);
    }
}

// 메시지 추가
function addMessage(text, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    // 스크롤을 맨 아래로
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return messageDiv;
}

// 로딩 메시지 표시
function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message';
    loadingDiv.id = 'loading-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content loading';
    contentDiv.textContent = '답변을 생성하는 중';
    
    loadingDiv.appendChild(contentDiv);
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 로딩 메시지 제거
function removeLoading() {
    const loadingMessage = document.getElementById('loading-message');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

// 질문 전송
async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        return;
    }
    
    // 사용자 메시지 표시
    addMessage(question, true);
    questionInput.value = '';
    
    // 입력 비활성화
    questionInput.disabled = true;
    sendButton.disabled = true;
    
    // 로딩 표시
    showLoading();
    
    try {
        const selectedSession = sessionSelect.value || null;
        
        const response = await fetch(`${API_BASE_URL}/qa`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                session_name: selectedSession,
            }),
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 로딩 제거
        removeLoading();
        
        // 답변 표시
        const messageDiv = addMessage(data.answer || '답변을 생성할 수 없습니다.');
        
        // 참고 문서 표시 (답변 메시지 안에 포함)
        if (data.sources && data.sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            
            const titleDiv = document.createElement('div');
            titleDiv.className = 'sources-title';
            titleDiv.textContent = `참고 문서 (${data.sources.length}개)`;
            sourcesDiv.appendChild(titleDiv);
            
            data.sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                const metadata = source.metadata || {};
                const session = metadata.session_name || 'N/A';
                const sourceType = source.source_type || 'N/A';
                const similarity = source.similarity ? (source.similarity * 100).toFixed(1) : 'N/A';
                sourceItem.textContent = `${session} - ${sourceType} (유사도: ${similarity}%)`;
                sourcesDiv.appendChild(sourceItem);
            });
            
            // 메시지 내용 div 안에 참고 문서 추가
            const messageContent = messageDiv.querySelector('.message-content');
            if (messageContent) {
                messageContent.appendChild(sourcesDiv);
            }
        }
        
    } catch (error) {
        // 로딩 제거
        removeLoading();
        
        // RAG 관련 에러만 채팅창에 표시
        if (error.message.includes('QA 시스템') || 
            error.message.includes('답변 생성') ||
            error.message.includes('초기화')) {
            addMessage(`오류: ${error.message}`);
        } else {
            // 일반적인 네트워크 에러는 콘솔에만 표시
            console.error('질문 전송 실패:', error);
            addMessage('서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
        }
    } finally {
        // 입력 다시 활성화
        questionInput.disabled = false;
        sendButton.disabled = false;
        questionInput.focus();
    }
}

// 이벤트 리스너
sendButton.addEventListener('click', sendQuestion);

questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

// 페이지 로드 시 세션 목록 로드
window.addEventListener('load', () => {
    loadSessions();
    questionInput.focus();
});
