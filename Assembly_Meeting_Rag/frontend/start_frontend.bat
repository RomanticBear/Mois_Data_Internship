@echo off
cd /d %~dp0
echo 프론트엔드 서버 시작 중...
echo 브라우저에서 http://localhost:8080 접속하세요.
echo.
python -m http.server 8080
pause

