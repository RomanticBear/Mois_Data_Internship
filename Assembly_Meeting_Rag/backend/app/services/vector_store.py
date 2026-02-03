"""
OpenAI Vector Store 관리 서비스
"""
import os
import requests
import time
from typing import Optional
from openai import OpenAI
from app.models.document import DocumentMetadata


class VectorStoreService:
    """OpenAI Vector Store 관리 (HTTP API 사용)"""
    
    def __init__(self, metadata_db=None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        self.vector_store_id: Optional[str] = None
        self.assistant_id: Optional[str] = None
        self.active_file_ids: list[str] = []  # 활성 파일 ID 목록 (캐시)
        self.metadata_db = metadata_db  # 메타DB 참조 (선택적)
        self._initialize_vector_store()
        self._initialize_assistant()
        self._load_active_file_ids()  # 서버 시작 시 Active 파일 ID 로드
    
    def _initialize_vector_store(self):
        """Vector Store 초기화 또는 기존 Vector Store 조회"""
        # 기존 Vector Store가 있는지 확인 (이름으로 검색)
        try:
            # Vector Store 목록 조회
            response = requests.get(
                f"{self.base_url}/vector_stores",
                headers=self.headers,
                params={"limit": 100}
            )
            
            if response.status_code == 200:
                vector_stores = response.json().get("data", [])
                # "Assembly Meeting Vector Store" 이름으로 찾기
                for vs in vector_stores:
                    if vs.get("name") == "Assembly Meeting Vector Store":
                        self.vector_store_id = vs["id"]
                        print(f"✅ 기존 Vector Store 사용: {self.vector_store_id}")
                        return
            
            # 기존 Vector Store가 없으면 생성
            create_data = {
                "name": "Assembly Meeting Vector Store",
                "expires_after": None  # 만료 없음
            }
            
            create_response = requests.post(
                f"{self.base_url}/vector_stores",
                headers=self.headers,
                json=create_data
            )
            
            if create_response.status_code == 200:
                vector_store = create_response.json()
                self.vector_store_id = vector_store["id"]
                print(f"✅ Vector Store 생성 완료: {self.vector_store_id}")
            else:
                print(f"⚠️ Vector Store 생성 실패: {create_response.status_code}")
                print(f"   응답: {create_response.text}")
                
        except Exception as e:
            print(f"⚠️ Vector Store 초기화 실패: {e}")
            # Vector Store 없이도 작동 가능 (기존 방식으로 fallback)
    
    def _initialize_assistant(self):
        """Assistant 초기화 또는 기존 Assistant 조회"""
        # 기존 Assistant가 있는지 확인
        assistants = self.client.beta.assistants.list()
        
        assistant_name = "Assembly Meeting RAG Assistant"
        existing_assistant = None
        
        for assistant in assistants.data:
            if assistant.name == assistant_name:
                existing_assistant = assistant
                break
        
        if existing_assistant:
            # 기존 Assistant 사용
            self.assistant_id = existing_assistant.id
            needs_update = False
            update_data = {}
            
            # 모델이 GPT-4o가 아니면 업데이트
            if hasattr(existing_assistant, 'model') and existing_assistant.model != "gpt-4o":
                needs_update = True
                update_data["model"] = "gpt-4o"
            
            # Vector Store 연결 확인 및 업데이트
            vector_store_connected = False
            if hasattr(existing_assistant, 'tool_resources') and \
               existing_assistant.tool_resources and \
               hasattr(existing_assistant.tool_resources, 'file_search'):
                file_search = existing_assistant.tool_resources.file_search
                if hasattr(file_search, 'vector_store_ids') and file_search.vector_store_ids:
                    if not self.vector_store_id:
                        self.vector_store_id = file_search.vector_store_ids[0]
                    # 현재 Vector Store가 연결되어 있는지 확인
                    if self.vector_store_id in file_search.vector_store_ids:
                        vector_store_connected = True
            
            # Vector Store가 연결되어 있지 않으면 업데이트
            if self.vector_store_id and not vector_store_connected:
                needs_update = True
                if "tool_resources" not in update_data:
                    update_data["tool_resources"] = {}
                update_data["tool_resources"]["file_search"] = {
                    "vector_store_ids": [self.vector_store_id]
                }
            
            # 필요한 업데이트 수행
            if needs_update:
                try:
                    updated_assistant = self.client.beta.assistants.update(
                        self.assistant_id,
                        **update_data
                    )
                    if "model" in update_data:
                        print(f"✅ Assistant 모델을 GPT-4o로 업데이트 완료")
                    if "tool_resources" in update_data:
                        print(f"✅ Assistant를 Vector Store에 연결 완료: {self.vector_store_id}")
                except Exception as e:
                    print(f"⚠️ Assistant 업데이트 실패: {e}")
            
            print(f"✅ 기존 Assistant 사용: {self.assistant_id}")
        else:
            # 새로운 Assistant 생성 (Vector Store 연결)
            assistant_data = {
                "name": assistant_name,
                "instructions": "당신은 국회회의록을 기반으로 질문에 답변하는 전문 AI입니다. 문서에 기반한 정확한 답변을 제공하세요.",
                "model": "gpt-4o",  # GPT-4o 모델 사용
                "tools": [{"type": "file_search"}]
            }
            
            # Vector Store가 있으면 연결
            if self.vector_store_id:
                assistant_data["tool_resources"] = {
                    "file_search": {
                        "vector_store_ids": [self.vector_store_id]
                    }
                }
            
            assistant = self.client.beta.assistants.create(**assistant_data)
            self.assistant_id = assistant.id
            print(f"✅ Assistant 생성 완료: {self.assistant_id}")
            if self.vector_store_id:
                print(f"   Vector Store 연결됨: {self.vector_store_id}")
    
    def _load_active_file_ids(self):
        """서버 시작 시 메타DB에서 Active 파일 ID 목록 로드"""
        if self.metadata_db:
            try:
                active_file_ids = self.metadata_db.get_active_file_ids()
                self.active_file_ids = active_file_ids
                print(f"✅ Active 파일 ID {len(active_file_ids)}개 로드 완료")
            except Exception as e:
                print(f"⚠️ Active 파일 ID 로드 실패: {e}")
    
    def upload_file(self, file_path: str, metadata: DocumentMetadata) -> str:
        """
        PDF 파일을 OpenAI에 업로드하고 Assistant에 연결
        
        Args:
            file_path: 업로드할 파일 경로
            metadata: 문서 메타데이터
            
        Returns:
            OpenAI File ID
        """
        # 파일 업로드 (확장자 소문자 처리)
        import os
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        # 파일을 열어서 업로드
        with open(file_path, "rb") as file:
            # 파일명을 소문자 확장자로 변경하여 업로드
            if file_ext in ['.pdf', '.docx', '.txt', '.md']:
                # OpenAI가 인식할 수 있도록 파일명 처리
                upload_file_name = os.path.splitext(file_name)[0] + file_ext.lower()
            else:
                upload_file_name = file_name
            
            uploaded_file = self.client.files.create(
                file=(upload_file_name, file, f"application/{file_ext[1:] if file_ext else 'octet-stream'}"),
                purpose="assistants"
            )
        
        # 활성 파일 목록에 추가 (캐시 업데이트)
        if uploaded_file.id not in self.active_file_ids:
            self.active_file_ids.append(uploaded_file.id)
        
        # Vector Store에 파일 추가
        if self.vector_store_id:
            try:
                self._add_file_to_vector_store(uploaded_file.id)
            except Exception as e:
                print(f"⚠️ Vector Store에 파일 추가 실패: {e}")
                # 파일은 업로드되었으므로 계속 진행
        
        # 메타DB가 있으면 Active 파일 ID 목록도 업데이트
        if self.metadata_db:
            try:
                # 메타DB의 get_active_file_ids()는 항상 최신 상태를 반환하므로
                # 여기서는 캐시만 업데이트
                pass
            except Exception as e:
                print(f"⚠️ 메타DB 동기화 실패 (계속 진행): {e}")
        
        return uploaded_file.id
    
    def _add_file_to_vector_store(self, file_id: str):
        """파일을 Vector Store에 추가"""
        if not self.vector_store_id:
            return
        
        file_batch_url = f"{self.base_url}/vector_stores/{self.vector_store_id}/file_batches"
        file_batch_data = {
            "file_ids": [file_id]
        }
        
        response = requests.post(file_batch_url, headers=self.headers, json=file_batch_data)
        
        if response.status_code == 200:
            batch = response.json()
            batch_id = batch["id"]
            
            # 배치 완료 대기
            max_wait = 60
            waited = 0
            while waited < max_wait:
                time.sleep(2)
                batch_status_url = f"{self.base_url}/vector_stores/{self.vector_store_id}/file_batches/{batch_id}"
                status_response = requests.get(batch_status_url, headers=self.headers)
                
                if status_response.status_code == 200:
                    batch_status = status_response.json()
                    current_status = batch_status.get("status", "unknown")
                    
                    if current_status == "completed":
                        print(f"✅ 파일이 Vector Store에 추가 완료: {file_id}")
                        return
                    elif current_status in ["failed", "cancelled"]:
                        raise Exception(f"파일 배치 실패: {current_status}")
                
                waited += 2
            
            raise Exception("파일 배치 타임아웃")
        else:
            raise Exception(f"파일 배치 생성 실패: {response.status_code} - {response.text}")
    
    def delete_file(self, file_id: str):
        """파일 삭제"""
        # 활성 파일 목록에서 제거
        if file_id in self.active_file_ids:
            self.active_file_ids.remove(file_id)
        
        # 파일 삭제
        try:
            self.client.files.delete(file_id)
        except Exception as e:
            print(f"파일 삭제 중 오류: {e}")
    
    def query(
        self, 
        question: str, 
        active_files_only: bool = True,
        file_ids: Optional[list[str]] = None
    ) -> dict:
        """
        Vector Store에서 질문 처리
        
        Args:
            question: 사용자 질문
            active_files_only: Active Window 파일만 검색 여부
            file_ids: 특정 파일 ID 목록 (None이면 전체 검색)
            
        Returns:
            답변 및 근거 정보
        """
        # 검색할 파일 ID 목록 결정
        if file_ids:
            search_file_ids = file_ids
        elif active_files_only:
            # Active Window의 파일만 가져오기
            if self.metadata_db:
                # 메타DB에서 Active 파일 ID 목록 조회 (최신 상태)
                search_file_ids = self.metadata_db.get_active_file_ids()
                # 캐시도 업데이트
                self.active_file_ids = search_file_ids.copy() if search_file_ids else []
            else:
                # 메타DB가 없으면 캐시된 목록 사용
                search_file_ids = self.active_file_ids if self.active_file_ids else None
        else:
            search_file_ids = None
        
        # OpenAI Assistants API를 통한 질문 처리
        # Vector Store를 사용하면 파일 첨부 불필요
        thread = self.client.beta.threads.create()
        
        # Vector Store를 사용하는 경우 파일 첨부 불필요
        # Assistant가 이미 Vector Store에 연결되어 있음
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question
        )
        
        # 미리 생성된 Assistant로 실행
        if not self.assistant_id:
            raise ValueError("Assistant가 초기화되지 않았습니다.")
        
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant_id
        )
        
        # 실행 완료 대기
        import time
        max_wait_time = 300  # 최대 5분 대기
        waited_time = 0
        while run.status in ["queued", "in_progress"]:
            if waited_time >= max_wait_time:
                raise Exception(f"Run timeout after {max_wait_time} seconds")
            time.sleep(1)
            waited_time += 1
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
        
        if run.status == "completed":
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            answer = messages.data[0].content[0].text.value
            
            # 근거 추출 (간단한 구현)
            sources = []
            if hasattr(run, "required_action"):
                # File Search 결과에서 근거 추출
                pass
            
            return {
                "answer": answer,
                "sources": sources,
                "thread_id": thread.id,
                "run_id": run.id
            }
        elif run.status == "failed":
            # 실패한 경우 상세 에러 정보 추출
            error_message = f"Run failed with status: {run.status}"
            error_details = []
            
            # last_error 확인
            if hasattr(run, "last_error"):
                if run.last_error:
                    if hasattr(run.last_error, "message"):
                        error_details.append(f"Message: {run.last_error.message}")
                    if hasattr(run.last_error, "code"):
                        error_details.append(f"Code: {run.last_error.code}")
                    # last_error 객체의 모든 속성 출력 (디버깅용)
                    try:
                        error_dict = run.last_error.model_dump() if hasattr(run.last_error, "model_dump") else str(run.last_error)
                        error_details.append(f"Full error: {error_dict}")
                    except:
                        error_details.append(f"Error object: {str(run.last_error)}")
                else:
                    error_details.append("last_error is None")
            else:
                error_details.append("last_error attribute not found")
            
            # run 객체의 다른 유용한 정보도 포함
            if hasattr(run, "thread_id"):
                error_details.append(f"Thread ID: {run.thread_id}")
            if hasattr(run, "id"):
                error_details.append(f"Run ID: {run.id}")
            
            if error_details:
                error_message += f" ({'; '.join(error_details)})"
            
            # 콘솔에도 출력 (디버깅용)
            print(f"❌ Run failed: {error_message}")
            
            raise Exception(error_message)
        elif run.status == "requires_action":
            raise Exception(f"Run requires action: {run.status}. This may indicate a tool call is needed.")
        elif run.status == "cancelled":
            raise Exception(f"Run was cancelled: {run.status}")
        else:
            raise Exception(f"Run ended with unexpected status: {run.status}")
    
    def get_active_file_ids(self) -> list[str]:
        """Active Window의 파일 ID 목록 조회"""
        return self.active_file_ids.copy()

