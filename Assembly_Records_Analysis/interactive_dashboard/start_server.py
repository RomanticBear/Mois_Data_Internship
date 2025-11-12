#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 로컬 웹 서버 실행 스크립트
"""

import http.server
import socketserver
import os
import webbrowser

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # 현재 디렉토리로 이동
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 서버가 시작되었습니다!")
        print(f"📊 브라우저에서 http://localhost:{PORT} 로 접속하세요")
        print(f"📁 서버 디렉토리: {script_dir}")
        print(f"\n⏹️  서버를 중지하려면 Ctrl+C를 누르세요\n")
        
        # 자동으로 브라우저 열기
        try:
            webbrowser.open(f'http://localhost:{PORT}')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 서버가 종료되었습니다.")
            httpd.shutdown()

if __name__ == "__main__":
    main()







